from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import UCF101ClipDataset, motion_score
from models import VideoPredictor, count_trainable_parameters


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = 11,
    sigma: float = 1.5,
):
    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype,
    ) - window_size // 2
    gaussian = torch.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    return window_2d.expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return SSIM for each image in a BCHW tensor."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    channels = prediction.shape[1]
    window = gaussian_window(
        channels,
        prediction.device,
        prediction.dtype,
    )
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(
        prediction,
        window,
        padding=padding,
        groups=channels,
    )
    mu_y = F.conv2d(
        target,
        window,
        padding=padding,
        groups=channels,
    )

    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(
            prediction.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(
            target.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(
            prediction * target,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_xy
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = (
        (2.0 * mu_xy + c1)
        * (2.0 * sigma_xy + c2)
        / (
            (mu_x_sq + mu_y_sq + c1)
            * (sigma_x_sq + sigma_y_sq + c2)
        ).clamp_min(1e-8)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def build_dataset(
    args,
    subset: str,
    *,
    train_full: bool = False,
):
    actual_subset = "train_full" if train_full else subset
    return UCF101ClipDataset(
        video_root=args.video_root,
        annotation_root=args.annotation_root,
        subset=actual_subset,
        split_id=args.split_id,
        clip_len=args.clip_len,
        temporal_stride=args.temporal_stride,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        horizontal_flip=not args.no_horizontal_flip,
        max_videos=args.max_videos,
    )


def build_loader(dataset, batch_size: int, num_workers: int, device: torch.device, *, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )


def build_predictor(args) -> VideoPredictor:
    return VideoPredictor(
        core_name=args.model,
        input_channels=3,
        output_channels=3,
        d_model=args.d_model,
        d_state=args.d_state,
        v_range=args.v_range,
        boundary=args.boundary,
        padding_mode=args.padding_mode,
        encoder_base_channels=args.encoder_base_channels,
        decoder_hidden_channels=args.decoder_hidden_channels,
        fernn_activation=args.fernn_activation,
    )


def save_predictor_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    predictor = model
    torch.save(
        {
            "model": predictor.state_dict(),
            "model_config": predictor.config_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )


def load_predictor_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[VideoPredictor, Dict[str, Any]]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model = VideoPredictor.from_config(
        checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def train_predictor_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    input_frames: int,
    teacher_forcing_ratio: float,
    grad_clip: float,
    amp: bool,
    detach_rollout: bool,
):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        video = batch["video"].to(
            device,
            non_blocking=True,
        )
        context = video[:, :input_frames]
        target = video[:, input_frames:]
        pred_len = target.shape[1]

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction = model(
                context,
                pred_len,
                target=target,
                teacher_forcing_ratio=teacher_forcing_ratio,
                detach_rollout=detach_rollout,
            )
            loss = F.l1_loss(prediction, target)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss: {loss.item()}"
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            optimizer.step()

        batch_size = video.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / max(sample_count, 1)



def save_prediction_montage(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
):
    
    import cv2

    context = context.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)
    prediction = prediction.detach().cpu().clamp(0, 1)

    tile_h, tile_w = context.shape[-2:]
    columns = max(context.shape[0], target.shape[0])

    def make_row(frames: torch.Tensor, label: str):
        row = np.zeros(
            (tile_h + 24, columns * tile_w, 3),
            dtype=np.uint8,
        )
        cv2.putText(
            row,
            label,
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for index, frame in enumerate(frames):
            rgb = (
                frame.permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            row[
                24:24 + tile_h,
                index * tile_w:(index + 1) * tile_w,
            ] = bgr
        return row

    canvas = np.concatenate(
        [
            make_row(context, "context"),
            make_row(target, "ground truth"),
            make_row(prediction, "prediction"),
        ],
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


@torch.no_grad()
def evaluate_prediction(
    model: nn.Module,
    loader,
    device: torch.device,
    input_frames: int,
    amp: bool,
    *,
    visual_dir: Optional[Path] = None,
    num_visuals: int = 0,
):
    model.eval()

    all_mse = []
    all_psnr = []
    all_ssim = []
    all_motion = []
    all_last_psnr = []
    per_step_mse_sum = None
    per_step_psnr_sum = None
    per_step_ssim_sum = None
    evaluated_clips = 0
    visuals_saved = 0

    for batch in loader:
        video = batch["video"].to(
            device,
            non_blocking=True,
        )
        context = video[:, :input_frames]
        target = video[:, input_frames:]

        with autocast_context(device, amp):
            prediction = model(
                context,
                target.shape[1],
            )

        prediction = prediction.float()
        target = target.float()

        squared_error = (
            prediction - target
        ).square()
        clip_mse = squared_error.mean(
            dim=(1, 2, 3, 4)
        )
        batch_size, pred_len, channels, height, width = (
            prediction.shape
        )
        frame_ssim = ssim_per_image(
            prediction.reshape(
                batch_size * pred_len,
                channels,
                height,
                width,
            ),
            target.reshape(
                batch_size * pred_len,
                channels,
                height,
                width,
            ),
        ).reshape(batch_size, pred_len)
        clip_ssim = frame_ssim.mean(dim=1)

        frame_mse = squared_error.mean(dim=(2, 3, 4))
        frame_psnr = -10.0 * torch.log10(
            frame_mse.clamp_min(1e-10)
        )
        clip_psnr = frame_psnr.mean(dim=1)
        last_psnr = frame_psnr[:, -1]

        batch_step_mse = frame_mse.sum(dim=0).cpu()
        batch_step_psnr = frame_psnr.sum(dim=0).cpu()
        batch_step_ssim = frame_ssim.sum(dim=0).cpu()

        if per_step_mse_sum is None:
            per_step_mse_sum = batch_step_mse
            per_step_psnr_sum = batch_step_psnr
            per_step_ssim_sum = batch_step_ssim
        else:
            per_step_mse_sum += batch_step_mse
            per_step_psnr_sum += batch_step_psnr
            per_step_ssim_sum += batch_step_ssim
        evaluated_clips += batch_size

        if visual_dir is not None and visuals_saved < num_visuals:
            remaining = num_visuals - visuals_saved
            for local_index in range(
                min(batch_size, remaining)
            ):
                save_prediction_montage(
                    context[local_index],
                    target[local_index],
                    prediction[local_index],
                    visual_dir
                    / f"sample_{visuals_saved:03d}.png",
                )
                visuals_saved += 1

        all_mse.append(clip_mse.cpu())
        all_psnr.append(clip_psnr.cpu())
        all_ssim.append(clip_ssim.cpu())
        all_motion.append(motion_score(context).cpu())
        all_last_psnr.append(last_psnr.cpu())

    mse = torch.cat(all_mse)
    psnr = torch.cat(all_psnr)
    ssim = torch.cat(all_ssim)
    motion = torch.cat(all_motion)
    last_psnr = torch.cat(all_last_psnr)

    q1, q2 = torch.quantile(
        motion,
        torch.tensor([1 / 3, 2 / 3]),
    )

    groups = {
        "low": motion <= q1,
        "medium": (motion > q1) & (motion <= q2),
        "high": motion > q2,
    }

    result = {
        "num_clips": int(mse.numel()),
        "mse": float(mse.mean()),
        "psnr": float(psnr.mean()),
        "psnr_last": float(last_psnr.mean()),
        "ssim": float(ssim.mean()),
        "per_step_mse": (
            per_step_mse_sum / max(evaluated_clips, 1)
        ).tolist(),
        "per_step_psnr": (
            per_step_psnr_sum / max(evaluated_clips, 1)
        ).tolist(),
        "per_step_ssim": (
            per_step_ssim_sum / max(evaluated_clips, 1)
        ).tolist(),
        "motion_thresholds": [
            float(q1),
            float(q2),
        ],
        "motion_groups": {},
    }

    for name, mask in groups.items():
        if not mask.any():
            result["motion_groups"][name] = {"count": 0}
            continue
        result["motion_groups"][name] = {
            "count": int(mask.sum()),
            "mse": float(mse[mask].mean()),
            "psnr": float(psnr[mask].mean()),
            "psnr_last": float(
                last_psnr[mask].mean()
            ),
            "ssim": float(ssim[mask].mean()),
            "motion_score": float(
                motion[mask].mean()
            ),
        }

    return result


class FrozenLinearProbe(nn.Module):
    def __init__(
        self,
        backbone: VideoPredictor,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.head = nn.Linear(
            backbone.d_model,
            num_classes,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.head.train(mode)
        return self

    def forward(self, video: torch.Tensor):
        with torch.no_grad():
            feature = self.backbone.extract_representation(
                video
            )
        return self.head(feature)


def topk_correct(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> int:
    k = min(k, logits.shape[1])
    prediction = logits.topk(k, dim=1).indices
    return int(
        prediction.eq(target[:, None]).any(dim=1).sum()
    )


def train_probe_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool,
):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        video = batch["video"].to(
            device,
            non_blocking=True,
        )
        label = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(video)
            loss = F.cross_entropy(logits, label)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * video.shape[0]
        sample_count += video.shape[0]

    return loss_sum / max(sample_count, 1)


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool,
):
    model.eval()
    loss_sum = 0.0
    count = 0
    top1 = 0
    top5 = 0

    for batch in loader:
        video = batch["video"].to(
            device,
            non_blocking=True,
        )
        label = batch["label"].to(
            device,
            non_blocking=True,
        )

        with autocast_context(device, amp):
            logits = model(video)
            loss = F.cross_entropy(logits, label)

        batch_size = video.shape[0]
        loss_sum += loss.item() * batch_size
        count += batch_size
        top1 += topk_correct(logits, label, 1)
        top5 += topk_correct(logits, label, 5)

    return {
        "loss": loss_sum / max(count, 1),
        "top1": top1 / max(count, 1),
        "top5": top5 / max(count, 1),
        "count": count,
    }



def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)


def append_history(path: Path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def pretrain(args):
    seed_everything(args.seed, args.deterministic)
    device = get_device()
    amp = device.type == "cuda" and not args.no_amp

    train_loader = build_loader(build_dataset(args, "train"), args.batch_size, args.num_workers, device, shuffle=True)
    val_loader = build_loader(build_dataset(args, "val"), args.eval_batch_size, args.num_workers, device)
    test_loader = build_loader(build_dataset(args, "test"), args.eval_batch_size, args.num_workers, device)

    model = build_predictor(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(amp)

    if args.run_name is None:
        args.run_name = f"ucf101_{args.model}_{args.input_frames}to{args.clip_len - args.input_frames}"
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", vars(args))
    (run_dir / "history.jsonl").write_text("", encoding="utf-8")
    print(f"Model={args.model} | parameters={count_trainable_parameters(model):,} | device={device}")

    best_val = math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_l1 = train_predictor_epoch(
            model, train_loader, optimizer, scaler, device,
            args.input_frames, args.teacher_forcing_ratio,
            args.grad_clip, amp, args.detach_rollout,
        )
        scheduler.step()
        val_metrics = evaluate_prediction(model, val_loader, device, args.input_frames, amp)
        val_mse = val_metrics["mse"]
        append_history(run_dir / "history.jsonl", {
            "epoch": epoch, "train_l1": train_l1, "validation": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_L1={train_l1:.6f} | val_MSE={val_mse:.6e} | "
            f"val_PSNR={val_metrics['psnr']:.3f} | val_SSIM={val_metrics['ssim']:.4f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            save_predictor_checkpoint(
                run_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_val, args,
            )
            save_json(run_dir / "best_validation_metrics.json", {"epoch": epoch, "validation": val_metrics})

        save_predictor_checkpoint(
            run_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, best_val, args,
        )

    best_model, _ = load_predictor_checkpoint(run_dir / "best.pt", device)
    test_metrics = evaluate_prediction(best_model, test_loader, device, args.input_frames, amp)
    save_json(run_dir / "best_metrics.json", {
        "epoch": best_epoch, "best_validation_mse": best_val, "test": test_metrics,
    })
    print(f"Best epoch {best_epoch} | test_MSE={test_metrics['mse']:.6e} | "
          f"test_PSNR={test_metrics['psnr']:.3f} | test_SSIM={test_metrics['ssim']:.4f}")


EVAL_DATA_FIELDS = (
    "split_id", "clip_len", "input_frames", "temporal_stride",
    "resize_size", "crop_size", "val_fraction", "split_seed",
    "no_horizontal_flip",
)


def evaluate_command(args):
    device = get_device()
    model, checkpoint = load_predictor_checkpoint(args.checkpoint, device)
    training_args = checkpoint.get("args", {})
    missing = [field for field in EVAL_DATA_FIELDS if field not in training_args]
    if missing:
        raise KeyError(f"Checkpoint is missing evaluation settings: {', '.join(missing)}")
    for field in EVAL_DATA_FIELDS:
        setattr(args, field, training_args[field])
    seed_everything(args.seed)
    amp = device.type == "cuda" and not args.no_amp
    loader = build_loader(build_dataset(args, args.subset), args.eval_batch_size, args.num_workers, device)
    visual_dir = Path(args.visual_dir) if args.save_visuals > 0 else None
    metrics = evaluate_prediction(
        model, loader, device, args.input_frames, amp,
        visual_dir=visual_dir, num_visuals=args.save_visuals,
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["model_config"] = checkpoint["model_config"]
    metrics["data_config"] = {field: getattr(args, field) for field in EVAL_DATA_FIELDS}
    output = (Path(args.output) if args.output is not None else
              Path(args.checkpoint).with_name(f"{args.subset}_prediction_metrics.json"))
    save_json(output, metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved metrics to {output}")


def probe(args):
    seed_everything(args.seed, args.deterministic)
    device = get_device()
    amp = device.type == "cuda" and not args.no_amp
    backbone, checkpoint = load_predictor_checkpoint(args.checkpoint, device)
    if args.random_backbone:
        backbone = VideoPredictor.from_config(checkpoint["model_config"]).to(device)

    train_loader = build_loader(build_dataset(args, "train"), args.batch_size, args.num_workers, device, shuffle=True)
    val_loader = build_loader(build_dataset(args, "val"), args.eval_batch_size, args.num_workers, device)
    test_loader = build_loader(build_dataset(args, "test"), args.eval_batch_size, args.num_workers, device)

    model = FrozenLinearProbe(backbone, num_classes=train_loader.dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(amp)
    if args.run_name is None:
        suffix = "random" if args.random_backbone else "pretrained"
        args.run_name = f"probe_{backbone.core_name}_{suffix}_seed{args.seed}"
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", vars(args))
    (run_dir / "history.jsonl").write_text("", encoding="utf-8")

    best_val = -math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_probe_epoch(model, train_loader, optimizer, scaler, device, amp)
        scheduler.step()
        val_metrics = evaluate_probe(model, val_loader, device, amp)
        append_history(run_dir / "history.jsonl", {
            "epoch": epoch, "train_loss": train_loss, "validation": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train={train_loss:.4f} | "
              f"val_top1={val_metrics['top1']:.4f} | val_top5={val_metrics['top5']:.4f}")
        if val_metrics["top1"] > best_val:
            best_val = val_metrics["top1"]
            best_epoch = epoch
            torch.save({
                "head": model.head.state_dict(),
                "backbone_checkpoint": str(args.checkpoint),
                "random_backbone": args.random_backbone,
                "epoch": epoch,
                "validation": val_metrics,
                "args": vars(args),
            }, run_dir / "best_probe.pt")
            save_json(run_dir / "best_validation_metrics.json", {"epoch": epoch, "validation": val_metrics})

    saved = torch.load(run_dir / "best_probe.pt", map_location=device, weights_only=False)
    model.head.load_state_dict(saved["head"])
    test_metrics = evaluate_probe(model, test_loader, device, amp)
    saved["test"] = test_metrics
    torch.save(saved, run_dir / "best_probe.pt")
    save_json(run_dir / "best_probe_metrics.json", {
        "epoch": best_epoch,
        "validation": saved["validation"],
        "test": test_metrics,
        "random_backbone": args.random_backbone,
    })
    print(f"Best epoch {best_epoch} | test_top1={test_metrics['top1']:.4f} | "
          f"test_top5={test_metrics['top5']:.4f}")


def add_data_arguments(parser):
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--split-id", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--input-frames", type=int, default=8)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--resize-size", type=int, default=72)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--no-horizontal-flip", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)


def add_runtime_arguments(parser):
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--run-name", default=None)


def create_parser():
    parser = argparse.ArgumentParser(description="UCF101 video prediction and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain_parser = subparsers.add_parser("pretrain")
    add_data_arguments(pretrain_parser)
    add_runtime_arguments(pretrain_parser)
    pretrain_parser.add_argument("--model", choices=["mamba", "flow_mamba", "fernn"], default="flow_mamba")
    pretrain_parser.add_argument("--d-model", type=int, default=64)
    pretrain_parser.add_argument("--d-state", type=int, default=16)
    pretrain_parser.add_argument("--v-range", type=int, default=1)
    pretrain_parser.add_argument("--boundary", choices=["zero", "circular"], default="zero")
    pretrain_parser.add_argument("--padding-mode", choices=["zeros", "circular", "reflect", "replicate"], default="zeros")
    pretrain_parser.add_argument("--encoder-base-channels", type=int, default=32)
    pretrain_parser.add_argument("--decoder-hidden-channels", type=int, default=32)
    pretrain_parser.add_argument("--fernn-activation", choices=["tanh", "relu"], default="tanh")
    pretrain_parser.add_argument("--epochs", type=int, default=20)
    pretrain_parser.add_argument("--batch-size", type=int, default=32)
    pretrain_parser.add_argument("--eval-batch-size", type=int, default=32)
    pretrain_parser.add_argument("--lr", type=float, default=3e-4)
    pretrain_parser.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain_parser.add_argument("--teacher-forcing-ratio", type=float, default=0.0)
    pretrain_parser.add_argument("--detach-rollout", action="store_true")
    pretrain_parser.add_argument("--grad-clip", type=float, default=1.0)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--video-root", required=True)
    eval_parser.add_argument("--annotation-root", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--subset", choices=["val", "test"], default="test")
    eval_parser.add_argument("--eval-batch-size", type=int, default=32)
    eval_parser.add_argument("--max-videos", type=int, default=None)
    eval_parser.add_argument("--num-workers", type=int, default=8)
    eval_parser.add_argument("--output", default=None)
    eval_parser.add_argument("--save-visuals", type=int, default=0)
    eval_parser.add_argument("--visual-dir", default="./prediction_visuals")
    eval_parser.add_argument("--seed", type=int, default=1234)
    eval_parser.add_argument("--no-amp", action="store_true")

    probe_parser = subparsers.add_parser("probe")
    add_data_arguments(probe_parser)
    add_runtime_arguments(probe_parser)
    probe_parser.add_argument("--checkpoint", required=True)
    probe_parser.add_argument("--random-backbone", action="store_true")
    probe_parser.add_argument("--epochs", type=int, default=25)
    probe_parser.add_argument("--batch-size", type=int, default=128)
    probe_parser.add_argument("--eval-batch-size", type=int, default=128)
    probe_parser.add_argument("--lr", type=float, default=1e-2)
    probe_parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main():
    args = create_parser().parse_args()
    if args.command != "evaluate" and not 0 < args.input_frames < args.clip_len:
        raise ValueError("Require 0 < input_frames < clip_len")
    if args.command == "pretrain":
        pretrain(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()

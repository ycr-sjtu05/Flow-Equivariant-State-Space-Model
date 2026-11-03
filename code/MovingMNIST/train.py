"""Train Flow-SSM or FERNN for 10→10 or 10→50 Moving MNIST prediction."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiment import architecture_config, build_model
from utils.dataset import MovingMNISTDataset
from utils.train_eval import evaluate_mse, train_epoch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flow_ssm", "fernn"), default="flow_ssm")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Local MNIST dataset/cache")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--pred-frames", type=int, choices=(10, 50), default=10,
                        help="Train directly for this horizon (10->10 or 10->50)")
    parser.add_argument("--input-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--data-v-range", type=int, default=2,
                        help="Digit translation velocity range, ±pixels/frame")
    parser.add_argument("--v-range", type=int, default=2,
                        help="Velocity-branch range, ±pixels/frame")
    parser.add_argument("--num-digits", type=int, default=2)
    parser.add_argument("--mamba-D", dest="mamba_D", type=int, default=64)
    parser.add_argument("--mamba-N", dest="mamba_N", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--decoder-conv-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.0)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit training examples")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit validation examples")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test examples")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test evaluation")
    args = parser.parse_args(argv)
    if (args.input_frames < 1 or args.batch_size < 1 or args.epochs < 1 or
        args.data_v_range < 0 or args.num_workers < 0 or args.num_digits < 1 or
        not (0.0 <= args.teacher_forcing_ratio <= 1.0)):
        parser.error("Invalid input frames, batch size, epochs, workers, digit count or forcing ratio.")
    for field in ("train_limit", "val_limit", "test_limit"):
        if (value := getattr(args, field)) is not None and value < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive.")
    return args


def resolve_device(preference):
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if (preference == "cuda" or
                         (preference == "auto" and torch.cuda.is_available())) else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit(dataset, count):
    return Subset(dataset, range(min(count, len(dataset)))) if count is not None else dataset


def make_dataloaders(args, device):
    kwargs = dict(root=str(args.root), seq_len=args.input_frames + args.pred_frames,
                  image_size=args.image_size,
                  velocity_range_x=(-args.data_v_range, args.data_v_range),
                  velocity_range_y=(-args.data_v_range, args.data_v_range),
                  num_digits=args.num_digits, download=not args.no_download)

    # Use disjoint source images for training and validation.
    initial = MovingMNISTDataset(train=True, random=True, **kwargs)
    shuffled = np.random.default_rng(args.data_seed).permutation(len(initial.mnist))
    n_val = max(1, int(0.1 * len(shuffled)))
    if n_val >= len(shuffled):
        raise ValueError("Not enough MNIST source digits for a train/val split.")
    train_indices, val_indices = shuffled[n_val:], shuffled[:n_val]

    train_ds = MovingMNISTDataset(train=True, random=True,
                                  digit_indices=train_indices, **kwargs)
    val_ds = MovingMNISTDataset(train=True, random=False, seed=args.data_seed + 1,
                                digit_indices=val_indices, **kwargs)
    test_ds = MovingMNISTDataset(train=False, random=False, seed=args.data_seed, **kwargs)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
    train_generator = torch.Generator().manual_seed(args.data_seed)
    return (
        DataLoader(maybe_limit(train_ds, args.train_limit), shuffle=True,
                   generator=train_generator, **common),
        DataLoader(maybe_limit(val_ds, args.val_limit), shuffle=False, **common),
        DataLoader(maybe_limit(test_ds, args.test_limit), shuffle=False, **common),
    )


def run(args):
    device = resolve_device(args.device)
    seed_everything(args.data_seed)
    train_loader, val_loader, test_loader = make_dataloaders(args, device)
    # Model initialization uses an independent seed.
    random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)

    config = architecture_config(args)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = args.output_dir / (f"{args.model}_{args.input_frames}to{args.pred_frames}"
                                  f"_seed{args.model_seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    best_validation = float("inf")
    best_epoch = None
    print(f"Device: {device}; model: {args.model}; parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training directly for {args.input_frames}->{args.pred_frames} frames")

    saved_config = {key: value for key, value in vars(args).items()
                        if key not in ("root", "output_dir")}
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            training_mse = train_epoch(model, train_loader, optimizer, device,
                                       args.input_frames, args.teacher_forcing_ratio,
                                       args.grad_clip)
            validation_mse = evaluate_mse(model, val_loader, device, args.input_frames)
            writer.writerow(dict(epoch=epoch, train_mse=training_mse, val_mse=validation_mse))
            file.flush()
            print(f"Epoch {epoch:03d}/{args.epochs} | train MSE {training_mse:.7f} | "
                  f"val MSE {validation_mse:.7f}")
            if validation_mse < best_validation:
                best_validation, best_epoch = validation_mse, epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "training_config": saved_config,
                    "epoch": epoch,
                    "best_val_mse": validation_mse,
                }, checkpoint_path)

    metrics = {"best_epoch": best_epoch, "best_val_mse": best_validation}
    if not args.skip_test:
        # Evaluate the checkpoint with the lowest validation MSE.
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                         weights_only=True)["model_state_dict"])
        metrics["test_mse"] = evaluate_mse(model, test_loader, device, args.input_frames)
        print(f"Final test MSE: {metrics['test_mse']:.7f}")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved best checkpoint: {checkpoint_path}")
    return metrics


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()

"""Evaluate Flow-SSM and FERNN on Moving MNIST."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiment import architecture_config, build_model
from utils.dataset import FixedVelocityMovingMNIST, MovingMNISTDataset
from utils.train_eval import evaluate_per_horizon
from utils.visualization import save_prediction_figure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--mode", choices=("standard", "fixed-velocity"), default="standard")
    parser.add_argument("--input-frames", type=int, default=None)
    parser.add_argument("--pred-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--data-v-range", type=int, default=None)
    parser.add_argument("--num-digits", type=int, default=None)
    parser.add_argument("--fixed-vx", type=int, default=2)
    parser.add_argument("--fixed-vy", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-figure", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")

    # Model parameters are needed when the checkpoint is a plain state_dict.
    parser.add_argument("--model", choices=("flow_ssm", "fernn"), default="flow_ssm")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--v-range", type=int, default=2)
    parser.add_argument("--mamba-D", dest="mamba_D", type=int, default=64)
    parser.add_argument("--mamba-N", dest="mamba_N", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--decoder-conv-layers", type=int, default=3)
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be non-negative.")
    for key in ("input_frames", "pred_frames", "num_digits", "limit"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    if args.data_v_range is not None and args.data_v_range < 0:
        parser.error("--data-v-range must be non-negative.")
    return args


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]
        training_config = checkpoint.get("training_config", {})
    else:
        config = architecture_config(args)
        state_dict = checkpoint
        training_config = {}

    model = build_model(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config, training_config


def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    model, model_config, saved_settings = load_model(args, device)

    def setting(name, fallback):
        specified = getattr(args, name)
        return specified if specified is not None else saved_settings.get(name, fallback)

    input_frames = setting("input_frames", 10)
    pred_frames = setting("pred_frames", 10)
    data_seed = setting("data_seed", 42)
    data_v_range = setting("data_v_range", 2)
    num_digits = setting("num_digits", 2)

    dataset_args = dict(
        root=str(args.root), train=False,
        seq_len=input_frames + pred_frames,
        image_size=model_config["image_size"],
        num_digits=num_digits, random=False, seed=data_seed,
        download=not args.no_download,
    )
    if args.mode == "fixed-velocity":
        dataset = FixedVelocityMovingMNIST(
            vx=args.fixed_vx, vy=args.fixed_vy, **dataset_args
        )
    else:
        dataset = MovingMNISTDataset(
            velocity_range_x=(-data_v_range, data_v_range),
            velocity_range_y=(-data_v_range, data_v_range),
            **dataset_args,
        )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda,
    )
    mse, horizon_mean, horizon_std = evaluate_per_horizon(model, loader, device, input_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = ("standard" if args.mode == "standard"
              else f"fixed_vx{args.fixed_vx}_vy{args.fixed_vy}")
    prefix = f"{model_config['model']}_{prefix}_{input_frames}to{pred_frames}"
    np.savez(
        args.output_dir / f"{prefix}_per_horizon.npz",
        mean=horizon_mean, std=horizon_std,
    )
    result = {
        "model": model_config["model"],
        "mode": args.mode,
        "input_frames": input_frames,
        "pred_frames": pred_frames,
        "data_seed": data_seed,
        "data_v_range": data_v_range,
        "num_digits": num_digits,
        "num_sequences": len(dataset),
        "test_mse": mse,
    }
    if args.mode == "fixed-velocity":
        result["velocity"] = {"vx": args.fixed_vx, "vy": args.fixed_vy}
    (args.output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if args.save_figure:
        sequence, _ = next(iter(loader))
        sequence = sequence.to(device)
        with torch.no_grad():
            prediction = model(sequence[:, :input_frames], pred_len=pred_frames)
        save_prediction_figure(
            sequence[:, :input_frames], sequence[:, input_frames:], prediction,
            args.output_dir / f"{prefix}_example.png",
        )
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()

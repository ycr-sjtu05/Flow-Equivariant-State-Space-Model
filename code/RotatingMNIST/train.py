import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiment import VELOCITIES, build_model, model_config
from utils.dataset import RotatingMNISTDataset
from utils.train_eval import evaluate_mse, train_epoch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rotating MNIST prediction")
    parser.add_argument("--model", choices=("flow_ssm", "fernn"), default="flow_ssm")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rotmnist"))
    parser.add_argument("--pred-frames", type=int, choices=(10, 50), default=10)
    parser.add_argument("--input-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=34)
    parser.add_argument("--num-digits", type=int, default=2)
    parser.add_argument("--v-list", type=int, nargs="+", default=list(VELOCITIES))
    parser.add_argument("--data-v-list", type=int, nargs="+", default=list(VELOCITIES))
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-rot", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--decoder-conv-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.0)
    parser.add_argument("--batches-per-eval", type=int, default=160)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.input_frames < 1 or args.batch_size < 1 or args.epochs < 1 or args.lr <= 0:
        parser.error("Input frames, batch size, epochs and learning rate must be positive")
    if args.num_digits < 1 or args.image_size < 28 or args.n_rot < 1 or args.d_model < 1:
        parser.error("Invalid image or model dimensions")
    if args.num_workers < 0 or args.batches_per_eval < 0:
        parser.error("Workers and evaluation frequency cannot be negative")
    if not 0 <= args.teacher_forcing_ratio <= 1:
        parser.error("Teacher forcing ratio must be between 0 and 1")
    for name in ("max_train_samples", "val_limit", "test_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(choice):
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    return torch.device("cuda" if choice == "cuda" or choice == "auto" and torch.cuda.is_available() else "cpu")


def limit(dataset, count):
    return Subset(dataset, list(range(min(count, len(dataset))))) if count is not None else dataset


def make_loaders(args, device):
    common = dict(
        root=str(args.root),
        seq_len=args.input_frames + args.pred_frames,
        image_size=args.image_size,
        angular_velocities=args.data_v_list,
        num_digits=args.num_digits,
        download=not args.no_download,
    )
    full_train = RotatingMNISTDataset(train=True, random=True, seed=args.data_seed, **common)
    full_val = RotatingMNISTDataset(train=True, random=False, seed=args.data_seed + 1, **common)
    test = RotatingMNISTDataset(train=False, random=False, seed=args.data_seed + 2, **common)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.data_seed)).tolist()
    n_val = int(0.1 * len(indices))
    if n_val < 1:
        raise ValueError("Dataset is too small for validation")
    train_indices, val_indices = indices[n_val:], indices[:n_val]
    train = Subset(full_train, train_indices)
    val = Subset(full_val, val_indices)
    if args.max_train_samples is not None:
        train = limit(train, args.max_train_samples)
    if args.val_limit is not None:
        val = limit(val, args.val_limit)
    if args.test_limit is not None:
        test = limit(test, args.test_limit)
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.data_seed)
    return (
        DataLoader(train, shuffle=True, generator=generator, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )


def run(args):
    device = get_device(args.device)
    seed_all(args.data_seed)
    train_loader, val_loader, test_loader = make_loaders(args, device)
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000
    seed_all(args.model_seed)
    config = model_config(args)
    model = build_model(config).to(device)
    seed_all(args.data_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = args.output_dir / f"{args.model}_{args.input_frames}to{args.pred_frames}_seed{args.model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    best_val = float("inf")
    best_epoch = 0
    training_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    print(f"Model: {args.model} | Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Protocol: {args.input_frames} -> {args.pred_frames} | data seed: {args.data_seed} | model seed: {args.model_seed}")
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_mse", "val_mse"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_mse = train_epoch(
                model, train_loader, optimizer, device, args.input_frames,
                args.teacher_forcing_ratio, val_loader, args.batches_per_eval,
            )
            val_mse = evaluate_mse(model, val_loader, device, args.input_frames)
            writer.writerow(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
            stream.flush()
            print(f"Epoch {epoch}/{args.epochs} | train MSE: {train_mse:.7f} | val MSE: {val_mse:.7f}")
            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                torch.save(
                    dict(model_state_dict=model.state_dict(), model_config=config,
                         training_config=training_config, epoch=epoch, best_val_mse=val_mse),
                    best_path,
                )
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse = evaluate_mse(model, test_loader, device, args.input_frames)
    result = dict(best_epoch=best_epoch, best_val_mse=best_val, test_mse=test_mse)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Test MSE: {test_mse:.7f} | Checkpoint: {best_path}")
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()

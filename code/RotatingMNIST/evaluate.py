import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiment import build_model
from train import get_device
from utils.dataset import RotatingMNISTDataset
from utils.train_eval import evaluate_mse
from utils.visualization import save_prediction_grid


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Rotating MNIST prediction")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--save-visualization", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def run(args):
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["model_config"]
    training = checkpoint["training_config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch_size = args.batch_size or training["batch_size"]
    root = args.root if args.root is not None else Path(training["root"])
    dataset = RotatingMNISTDataset(
        root=str(root),
        train=False,
        seq_len=training["input_frames"] + training["pred_frames"],
        image_size=training["image_size"],
        angular_velocities=training["data_v_list"],
        num_digits=training["num_digits"],
        random=False,
        seed=training["data_seed"] + 2,
        download=not training.get("no_download", False),
    )
    if args.test_limit is not None:
        dataset = Subset(dataset, range(min(args.test_limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    mse = evaluate_mse(model, loader, device, training["input_frames"])
    print(json.dumps({"test_mse": mse}, indent=2))
    if args.save_visualization is not None:
        seq, _ = next(iter(loader))
        seq = seq.to(device)
        inp, target = seq[:, :training["input_frames"]], seq[:, training["input_frames"]:]
        with torch.inference_mode():
            pred = model(inp, pred_len=target.size(1), teacher_forcing_ratio=0.0)
        save_prediction_grid(inp, target, pred, args.save_visualization)
    return mse


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()

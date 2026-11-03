import json
from pathlib import Path

import torch

from evaluate import main as evaluate_main
from train import main as train_main


def test_end_to_end_train_then_evaluate(fake_mnist, tmp_path):
    result = train_main([
        "--root", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "runs"),
        "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
        "--train-limit", "2", "--val-limit", "2", "--test-limit", "2",
        "--data-seed", "77",
        "--mamba-D", "4", "--mamba-N", "2", "--v-range", "0",
        "--decoder-conv-layers", "0", "--device", "cpu",
    ])
    assert result["test_mse"] >= 0
    best_path = tmp_path / "runs" / "flow_ssm_10to10_seed1234" / "best.pt"
    saved = torch.load(best_path, weights_only=True)
    assert saved["model_config"]["mamba_D"] == 4
    assert saved["training_config"]["data_seed"] == 77
    assert "root" not in saved["training_config"]
    assert "output_dir" not in saved["training_config"]

    evaluated = evaluate_main([
        "--checkpoint", str(best_path),
        "--root", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "evaluations"),
        "--num-workers", "0", "--batch-size", "2",
        "--limit", "2", "--device", "cpu",
    ])
    assert evaluated["test_mse"] >= 0
    assert abs(evaluated["test_mse"] - result["test_mse"]) < 1e-7
    assert evaluated["num_sequences"] == 2
    assert evaluated["data_seed"] == 77
    files = list((tmp_path / "evaluations").glob("*_metrics.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["pred_frames"] == 10

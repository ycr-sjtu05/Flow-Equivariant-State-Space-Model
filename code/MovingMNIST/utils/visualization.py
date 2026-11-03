"""Visualize Moving MNIST predictions."""

from pathlib import Path

import torch


def save_prediction_figure(inputs: torch.Tensor, targets: torch.Tensor,
                           predictions: torch.Tensor, output_path: str | Path,
                           max_frames: int = 10) -> Path:
    """Save one sample: observed input, future ground truth and prediction."""
    import matplotlib.pyplot as plt

    if not 1 <= max_frames:
        raise ValueError("max_frames must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed = inputs[0, :, 0].detach().cpu()
    ground_truth = targets[0, :, 0].detach().cpu()
    predicted = predictions[0, :, 0].detach().cpu()
    input_ids = torch.linspace(0, observed.size(0) - 1,
                               steps=min(max_frames, observed.size(0))).long()
    future_ids = torch.linspace(0, ground_truth.size(0) - 1,
                                steps=min(max_frames, ground_truth.size(0))).long()
    ncols = max(len(input_ids), len(future_ids))
    figure, axes = plt.subplots(3, ncols, figsize=(1.3 * ncols, 4.0), squeeze=False)
    for row, frames, indices, label in (
        (0, observed, input_ids, "Input"),
        (1, ground_truth, future_ids, "Target"),
        (2, predicted, future_ids, "Prediction"),
    ):
        for col in range(ncols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices):
                ax.imshow(frames[int(indices[col])], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path

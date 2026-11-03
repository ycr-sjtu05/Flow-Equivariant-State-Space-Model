from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_prediction_grid(input_seq, target_seq, prediction, output_path, sample=0, stride=1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = target_seq[sample, ::stride, 0].detach().cpu()
    pred = prediction[sample, ::stride, 0].detach().cpu()
    count = target.shape[0]
    fig, axes = plt.subplots(3, count, figsize=(max(count, 4), 3), squeeze=False)
    for t in range(count):
        axes[0, t].imshow(target[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(pred[t], cmap="gray", vmin=0, vmax=1)
        axes[2, t].imshow((pred[t] - target[t]).abs(), cmap="gray", vmin=0, vmax=1)
        for ax in axes[:, t]:
            ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

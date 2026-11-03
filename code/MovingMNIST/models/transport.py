"""Circular spatial transport for integer translation-velocity branches."""

import torch


def translate_branches(state: torch.Tensor, velocities: list[tuple[int, int]]) -> torch.Tensor:
    """Transport each branch by one frame using periodic boundaries.

    Args:
        state: Tensor of shape (batch, velocity_branches, channels, height, width).
        velocities: Branch ordering as (vx, vy) integer pixel/frame pairs.

    Returns:
        State with each branch rolled by (vy, vx) along (height, width).
    """
    return torch.stack(
        [torch.roll(state[:, i], shifts=(vy, vx), dims=(2, 3))
         for i, (vx, vy) in enumerate(velocities)],
        dim=1,
    )

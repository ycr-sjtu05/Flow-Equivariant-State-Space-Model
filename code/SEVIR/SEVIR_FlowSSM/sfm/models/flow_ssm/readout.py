from __future__ import annotations
import torch
import torch.nn as nn
from .specs import PairSpec


class StrictReadout(nn.Module):
    def __init__(self, pair_spec: PairSpec, out_channels: int) -> None:
        super().__init__()
        self.pair_spec = pair_spec
        self.out_proj = nn.LazyLinear(out_channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        y = state.permute(0, 1, 3, 4, 2).contiguous()
        y = self.out_proj(y)
        y = y.permute(0, 1, 4, 2, 3).contiguous()
        y = y.mean(dim=1)
        return y

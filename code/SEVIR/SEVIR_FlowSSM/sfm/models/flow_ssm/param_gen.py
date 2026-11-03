from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LevelParams:
    feat: torch.Tensor
    u_proj: torch.Tensor


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LevelParamGenerator(nn.Module):


    def __init__(
        self,
        in_channels: int,
        feat_dim: int = 64,
        u_dim: int | None = None,
    ) -> None:
        super().__init__()
        if u_dim is None:
            u_dim = in_channels

        self.net = nn.Sequential(
            ConvGNAct(in_channels, feat_dim),
            ConvGNAct(feat_dim, feat_dim),
            ConvGNAct(feat_dim, feat_dim),
        )
        self.feat_head = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.u_head = nn.Conv2d(feat_dim, u_dim, kernel_size=1)

    def forward(self, x_level: torch.Tensor) -> LevelParams:
        h = self.net(x_level)
        feat = self.feat_head(h)
        u_proj = self.u_head(h)
        return LevelParams(feat=feat, u_proj=u_proj)

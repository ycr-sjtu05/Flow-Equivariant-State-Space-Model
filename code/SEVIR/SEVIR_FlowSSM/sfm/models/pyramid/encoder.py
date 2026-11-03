from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutput:
    latents: dict[int, torch.Tensor]
    guides: dict[int, torch.Tensor]


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.block1 = ConvGNAct(channels, channels, num_groups=num_groups)
        self.block2 = ConvGNAct(channels, channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class ScaleEncoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = ConvGNAct(in_channels, mid_channels)
        self.res1 = ResBlock2d(mid_channels)
        self.proj = ConvGNAct(mid_channels, out_channels)
        self.res2 = ResBlock2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.proj(x)
        x = self.res2(x)
        return x


class FuseBlock(nn.Module):
    def __init__(self, in_a: int, in_b: int, out_channels: int) -> None:
        super().__init__()
        self.proj_a = nn.Conv2d(in_a, out_channels, kernel_size=1)
        self.proj_b = nn.Conv2d(in_b, out_channels, kernel_size=1)
        self.res = ResBlock2d(out_channels)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = self.proj_a(a) + self.proj_b(b)
        x = self.res(x)
        return x


class GuideStem(nn.Module):
    def __init__(self, in_channels: int, guide_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_channels, guide_channels),
            ResBlock2d(guide_channels),
            ResBlock2d(guide_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyramidEncoder(nn.Module):


    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: tuple[int, int, int] = (64, 64, 64),
        mid_channels: int = 32,
        guide_channels: int = 16,
        use_guides: bool = True,
    ) -> None:
        super().__init__()
        c0, c1, c2 = latent_channels
        self.use_guides = use_guides

        self.enc0 = ScaleEncoder(in_channels, mid_channels, c0)
        self.enc1 = ScaleEncoder(in_channels, mid_channels, c1)
        self.enc2 = ScaleEncoder(in_channels, mid_channels, c2)

        self.fuse1 = FuseBlock(c1, c2, c1)
        self.fuse0 = FuseBlock(c0, c1, c0)

        if use_guides:
            self.guide_stem = GuideStem(in_channels, guide_channels)
        else:
            self.guide_stem = None

    def _encode_frame(self, x: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        x64 = F.interpolate(x, size=(64, 64), mode="area")
        x32 = F.interpolate(x, size=(32, 32), mode="area")
        x16 = F.interpolate(x, size=(16, 16), mode="area")

        h0 = self.enc0(x64)
        h1 = self.enc1(x32)
        h2 = self.enc2(x16)

        e2 = h2
        e1 = self.fuse1(h1, F.interpolate(e2, size=(32, 32), mode="nearest"))
        e0 = self.fuse0(h0, F.interpolate(e1, size=(64, 64), mode="nearest"))

        latents = {0: e0, 1: e1, 2: e2}

        guides: dict[int, torch.Tensor] = {}
        if self.use_guides:
            g384 = self.guide_stem(x)
            g192 = F.interpolate(g384, size=(192, 192), mode="area")
            g96 = F.interpolate(g384, size=(96, 96), mode="area")
            guides = {96: g96, 192: g192, 384: g384}

        return latents, guides

    def forward(self, x: torch.Tensor) -> EncoderOutput:

        latents_by_level = {0: [], 1: [], 2: []}
        guides_by_res = {96: [], 192: [], 384: []} if self.use_guides else {}

        for t in range(x.shape[1]):
            latents_t, guides_t = self._encode_frame(x[:, t])
            for level in [0, 1, 2]:
                latents_by_level[level].append(latents_t[level])
            if self.use_guides:
                for res in [96, 192, 384]:
                    guides_by_res[res].append(guides_t[res])

        latents = {level: torch.stack(latents_by_level[level], dim=1) for level in [0, 1, 2]}
        guides = {}
        if self.use_guides:
            guides = {res: torch.stack(guides_by_res[res], dim=1) for res in [96, 192, 384]}

        return EncoderOutput(latents=latents, guides=guides)

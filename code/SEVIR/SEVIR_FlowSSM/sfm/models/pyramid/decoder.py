from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block1 = ConvGNAct(channels, channels)
        self.block2 = ConvGNAct(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class LatentDecodeBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj_main = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.proj_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.res = ResBlock2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj_main(x)
        if skip is not None:
            x = x + self.proj_skip(skip)
        x = self.res(x)
        return x


class GatedRefineBlock(nn.Module):
    def __init__(self, feat_channels: int, guide_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj_feat = nn.Conv2d(feat_channels, out_channels, kernel_size=1)
        self.proj_guide = nn.Conv2d(guide_channels, out_channels, kernel_size=1)
        self.gate = nn.Conv2d(guide_channels, out_channels, kernel_size=1)
        self.res = ResBlock2d(out_channels)

    def forward(self, feat: torch.Tensor, guide: torch.Tensor | None = None) -> torch.Tensor:
        feat_p = self.proj_feat(feat)
        if guide is None:
            x = feat_p
        else:
            guide_p = self.proj_guide(guide)
            gate = torch.sigmoid(self.gate(guide))
            x = feat_p + gate * feat_p + guide_p
        x = self.res(x)
        return x


class PyramidDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: tuple[int, int, int] = (64, 64, 64),
        decode_channels: int = 64,
        guide_channels: int = 16,
        use_guides: bool = True,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        c0, c1, c2 = latent_channels
        self.use_guides = use_guides

        self.dec2 = LatentDecodeBlock(c2, c2, decode_channels)
        self.dec1 = LatentDecodeBlock(c1, decode_channels, decode_channels)
        self.dec0 = LatentDecodeBlock(c0, decode_channels, decode_channels)

        self.refine96 = GatedRefineBlock(decode_channels, guide_channels, 64)
        self.refine192 = GatedRefineBlock(64, guide_channels, 48)
        self.refine384 = GatedRefineBlock(48, guide_channels, 32)

        self.head = nn.Sequential(
            ResBlock2d(32),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def _decode_frame(self, z0, z1, z2, guide96=None, guide192=None, guide384=None):
        d2 = self.dec2(z2, skip=None)
        d1 = self.dec1(F.interpolate(d2, size=(32, 32), mode="nearest"), z1)
        d0 = self.dec0(F.interpolate(d1, size=(64, 64), mode="nearest"), z0)

        d96 = F.interpolate(d0, size=(96, 96), mode="bilinear", align_corners=False)
        d96 = self.refine96(d96, guide96 if self.use_guides else None)
        d192 = F.interpolate(d96, size=(192, 192), mode="bilinear", align_corners=False)
        d192 = self.refine192(d192, guide192 if self.use_guides else None)
        d384 = F.interpolate(d192, size=(384, 384), mode="bilinear", align_corners=False)
        d384 = self.refine384(d384, guide384 if self.use_guides else None)
        return self.head(d384)

    def forward(self, z_pred_pyramid: dict[int, torch.Tensor], guide_context: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        z0, z1, z2 = z_pred_pyramid[0], z_pred_pyramid[1], z_pred_pyramid[2]
        ys = []
        for t in range(z0.shape[1]):
            g96 = g192 = g384 = None
            if self.use_guides and guide_context is not None:
                if 96 in guide_context:
                    g = guide_context[96]
                    g96 = g[:, t] if g.ndim == 5 else g
                if 192 in guide_context:
                    g = guide_context[192]
                    g192 = g[:, t] if g.ndim == 5 else g
                if 384 in guide_context:
                    g = guide_context[384]
                    g384 = g[:, t] if g.ndim == 5 else g

            ys.append(self._decode_frame(z0[:, t], z1[:, t], z2[:, t], g96, g192, g384))
        return torch.stack(ys, dim=1)

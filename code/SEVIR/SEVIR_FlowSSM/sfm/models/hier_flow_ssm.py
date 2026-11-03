from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class HierConvBlock2d(nn.Module):


    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        norm_groups: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        g1 = choose_gn_groups(channels, norm_groups)
        g2 = choose_gn_groups(hidden_channels, norm_groups)

        self.net = nn.Sequential(
            nn.GroupNorm(g1, channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(g2, hidden_channels),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyramidHierarchyAdapter(nn.Module):


    def __init__(
        self,
        channels_per_scale: dict[int, int],
        adapter_channels: int,
        gamma_max: float = 0.10,
        gamma_init: float = -2.0,
        norm_groups: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.scales = tuple(sorted(channels_per_scale.keys()))
        self.channels_per_scale = channels_per_scale
        self.gamma_max = float(gamma_max)

        self.blocks = nn.ModuleDict()
        for s in self.scales:
            c = channels_per_scale[s]
            self.blocks[str(s)] = HierConvBlock2d(
                channels=c,
                hidden_channels=adapter_channels,
                norm_groups=norm_groups,
                zero_init=zero_init,
            )

        self.raw_self_gates = nn.ParameterDict()
        for s in self.scales:
            self.raw_self_gates[str(s)] = nn.Parameter(torch.tensor(-1.0))

        self.raw_td_gates = nn.ParameterDict()
        for src in self.scales:
            for dst in self.scales:
                if src > dst:
                    self.raw_td_gates[f"{src}_to_{dst}"] = nn.Parameter(torch.tensor(-2.0))

        self.raw_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_probe: dict[str, float] = {}

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @staticmethod
    def _to_4d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int] | None]:
        if x.dim() == 4:
            return x, None
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b * t, c, h, w), (b, t, c, h, w)
        raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")

    @staticmethod
    def _from_4d(x: torch.Tensor, shape_info: tuple[int, int, int, int, int] | None) -> torch.Tensor:
        if shape_info is None:
            return x
        b, t, c, h, w = shape_info
        return x.reshape(b, t, c, h, w)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, z_pred_pyramid: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        scales = [s for s in self.scales if s in z_pred_pyramid]
        if len(scales) == 0:
            raise ValueError("PyramidHierarchyAdapter received empty pyramid.")

        flat: dict[int, torch.Tensor] = {}
        shape_info: dict[int, tuple[int, int, int, int, int] | None] = {}

        for s in scales:
            x4, info = self._to_4d(z_pred_pyramid[s])
            flat[s] = x4
            shape_info[s] = info

        local_msg: dict[int, torch.Tensor] = {}
        for s in scales:
            local_msg[s] = self.blocks[str(s)](flat[s])

        gamma = self.gamma
        out_flat: dict[int, torch.Tensor] = {}
        probe_tensors: dict[str, torch.Tensor] = {
            "gamma": gamma.detach(),
        }

        eps = 1e-8

        for dst in scales:
            self_gate = torch.sigmoid(self.raw_self_gates[str(dst)])
            correction = self_gate * local_msg[dst]
            probe_tensors[f"self_gate_s{dst}"] = self_gate.detach()

            for src in scales:
                if src <= dst:
                    continue
                edge_name = f"{src}_to_{dst}"
                if edge_name not in self.raw_td_gates:
                    continue

                td_gate = torch.sigmoid(self.raw_td_gates[edge_name])
                msg = self._resize_like(local_msg[src], flat[dst])
                correction = correction + td_gate * msg

                probe_tensors[f"td_gate_{edge_name}"] = td_gate.detach()

            adapted = flat[dst] + gamma * correction
            out_flat[dst] = adapted

            with torch.no_grad():
                base_norm = flat[dst].detach().float().norm().clamp_min(eps)
                corr_norm = (gamma * correction).detach().float().norm()
                probe_tensors[f"base_norm_s{dst}"] = base_norm
                probe_tensors[f"corr_norm_s{dst}"] = corr_norm
                probe_tensors[f"corr_to_base_s{dst}"] = corr_norm / base_norm

        out = dict(z_pred_pyramid)
        for s in scales:
            out[s] = self._from_4d(out_flat[s], shape_info[s])

        self.last_probe = {
            k: float(v.detach().cpu().item())
            for k, v in probe_tensors.items()
            if torch.is_tensor(v) and v.numel() == 1
        }

        return out


class HierPyramidDecoder(nn.Module):


    def __init__(
        self,
        inner_decoder: PyramidDecoder,
        latent_channels: tuple[int, int, int],
        adapter_channels: int,
        gamma_max: float,
        gamma_init: float,
        norm_groups: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.inner = inner_decoder

        channels_per_scale = {
            0: latent_channels[0],
            1: latent_channels[1],
            2: latent_channels[2],
        }

        self.adapter = PyramidHierarchyAdapter(
            channels_per_scale=channels_per_scale,
            adapter_channels=adapter_channels,
            gamma_max=gamma_max,
            gamma_init=gamma_init,
            norm_groups=norm_groups,
            zero_init=zero_init,
        )

    @property
    def last_probe(self) -> dict[str, float]:
        return self.adapter.last_probe

    def forward(
        self,
        z_pred_pyramid: dict[int, torch.Tensor],
        guide_context: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        z_adapted = self.adapter(z_pred_pyramid)
        return self.inner(
            z_pred_pyramid=z_adapted,
            guide_context=guide_context,
        )

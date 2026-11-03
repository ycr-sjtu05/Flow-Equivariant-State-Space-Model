from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaStepAux:
    extra: dict[str, Any] = field(default_factory=dict)


class SimpleMambaBlock(nn.Module):


    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)


        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        N, T, C = x.shape

        xz = self.in_proj(x)
        x_part, z_part = xz.chunk(2, dim=-1)


        u = x_part.transpose(1, 2)
        u = self.conv1d(u)[..., :T]
        u = u.transpose(1, 2)
        u = F.silu(u)

        x_proj = self.x_proj(u)
        dt_raw, B_raw, C_raw = torch.split(
            x_proj,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1,
        )

        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)


        h = torch.zeros(
            N,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        ys = []


        for t in range(T):
            dt_t = dt[:, t]
            u_t = u[:, t]
            B_t = B_raw[:, t]
            C_t = C_raw[:, t]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_u = (
                dt_t.unsqueeze(-1)
                * u_t.unsqueeze(-1)
                * B_t.unsqueeze(1)
            )

            h = dA * h + dB_u

            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y * F.silu(z_part)
        y = self.out_proj(y)
        return y


class TemporalMambaLevel(nn.Module):


    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual_scale: float = 0.5,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.residual_scale = float(residual_scale)

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_blocks):
            self.blocks.append(
                SimpleMambaBlock(
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            self.norms.append(nn.LayerNorm(channels))

        self.out = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        assert C == self.channels, f"Expected C={self.channels}, got C={C}"

        tokens = x_seq.permute(0, 3, 4, 1, 2).contiguous()
        tokens = tokens.view(B * H * W, T, C)

        h = tokens
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h))

        last_h = h[:, -1]
        delta = self.out(last_h)

        delta = delta.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        last_x = x_seq[:, -1]

        z_next = last_x + self.residual_scale * torch.tanh(delta)
        return z_next


class MambaCore(nn.Module):


    def __init__(
        self,
        latent_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_blocks: int = 1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.levels = (0, 1, 2)

        self.level_models = nn.ModuleDict({
            "0": TemporalMambaLevel(
                channels=latent_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                residual_scale=residual_scale,
                num_blocks=num_blocks,
            ),
            "1": TemporalMambaLevel(
                channels=latent_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                residual_scale=residual_scale,
                num_blocks=num_blocks,
            ),
            "2": TemporalMambaLevel(
                channels=latent_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                residual_scale=residual_scale,
                num_blocks=num_blocks,
            ),
        })

    def _predict_next_pyramid(
        self,
        hist: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        z_next = {}

        for level in self.levels:
            z_next[level] = self.level_models[str(level)](hist[level])

        return z_next

    def rollout_pyramid(
        self,
        x0_ctx: torch.Tensor,
        x1_ctx: torch.Tensor | None,
        x2_ctx: torch.Tensor | None,
        pred_steps: int,
        teacher_forcing_pyramid: dict[int, torch.Tensor] | None = None,
        state0=None,
        ablation_cfg=None,
        return_probe: bool = False,
    ) -> tuple[dict[int, torch.Tensor], list[MambaStepAux]]:
        if x1_ctx is None or x2_ctx is None:
            raise ValueError(
                "MambaCore expects x0_ctx, x1_ctx, x2_ctx from PyramidEncoder."
            )

        hist = {
            0: x0_ctx,
            1: x1_ctx,
            2: x2_ctx,
        }

        z0s: list[torch.Tensor] = []
        z1s: list[torch.Tensor] = []
        z2s: list[torch.Tensor] = []
        aux_list: list[MambaStepAux] = []

        for k in range(pred_steps):
            z_next = self._predict_next_pyramid(hist)

            z0s.append(z_next[0])
            z1s.append(z_next[1])
            z2s.append(z_next[2])

            if return_probe:
                aux = MambaStepAux()
                aux.extra["z0_norm"] = z_next[0].detach().pow(2).mean().sqrt()
                aux.extra["z1_norm"] = z_next[1].detach().pow(2).mean().sqrt()
                aux.extra["z2_norm"] = z_next[2].detach().pow(2).mean().sqrt()
                aux_list.append(aux)
            else:
                aux_list.append(MambaStepAux())

            if teacher_forcing_pyramid is not None and teacher_forcing_pyramid.get(0, None) is not None:
                next0 = teacher_forcing_pyramid[0][:, k]
                next1 = teacher_forcing_pyramid[1][:, k]
                next2 = teacher_forcing_pyramid[2][:, k]
            else:
                next0 = z_next[0]
                next1 = z_next[1]
                next2 = z_next[2]

            hist = {
                0: torch.cat([hist[0], next0.unsqueeze(1)], dim=1),
                1: torch.cat([hist[1], next1.unsqueeze(1)], dim=1),
                2: torch.cat([hist[2], next2.unsqueeze(1)], dim=1),
            }

        z_pyr = {
            0: torch.stack(z0s, dim=1),
            1: torch.stack(z1s, dim=1),
            2: torch.stack(z2s, dim=1),
        }

        return z_pyr, aux_list

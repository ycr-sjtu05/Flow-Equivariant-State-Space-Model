from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FERNNState:

    levels: dict[int, torch.Tensor]
    t: int = 0


@dataclass
class FERNNStepAux:
    extra: dict[str, Any] = field(default_factory=dict)


class FlowConvGRUCell(nn.Module):


    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        in_ch = input_channels + hidden_channels

        self.gates = nn.Conv2d(
            in_ch,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        B, V, Ch, H, W = h.shape
        Cu = u.shape[2]

        h2 = h.reshape(B * V, Ch, H, W)
        u2 = u.reshape(B * V, Cu, H, W)

        gate_in = torch.cat([u2, h2], dim=1)
        z_raw, r_raw = self.gates(gate_in).chunk(2, dim=1)
        z = torch.sigmoid(z_raw)
        r = torch.sigmoid(r_raw)

        cand_in = torch.cat([u2, r * h2], dim=1)
        h_tilde = torch.tanh(self.candidate(cand_in))

        h_next = (1.0 - z) * h2 + z * h_tilde
        h_next = h_next.reshape(B, V, Ch, H, W)
        return h_next


class FERNNLevel(nn.Module):


    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        v_axis_size: int = 3,
        kernel_size: int = 3,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.v_size = int(v_axis_size) ** 2
        self.residual_scale = float(residual_scale)

        self.in_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.cell = FlowConvGRUCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )


        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )

    def build_initial_state(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.v_size,
            self.hidden_channels,
            h,
            w,
            device=device,
            dtype=dtype,
        )

    def lift_input(self, x: torch.Tensor) -> torch.Tensor:

        u = self.in_proj(x)
        u = u.unsqueeze(1).expand(-1, self.v_size, -1, -1, -1).contiguous()
        return u

    def step(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_t = self.lift_input(x_t)
        h_next = self.cell(h_t, u_t)

        pooled = h_next.mean(dim=1)
        raw = self.out(pooled)

        z_next = x_t + self.residual_scale * torch.tanh(raw)
        return z_next, h_next


class FERNNCore(nn.Module):


    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int | None = None,
        v_axis_size: int = 3,
        kernel_size: int = 3,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = latent_channels

        self.levels = (0, 1, 2)
        self.hidden_channels = hidden_channels

        self.level_models = nn.ModuleDict({
            "0": FERNNLevel(
                input_channels=latent_channels,
                hidden_channels=hidden_channels,
                v_axis_size=v_axis_size,
                kernel_size=kernel_size,
                residual_scale=residual_scale,
            ),
            "1": FERNNLevel(
                input_channels=latent_channels,
                hidden_channels=hidden_channels,
                v_axis_size=v_axis_size,
                kernel_size=kernel_size,
                residual_scale=residual_scale,
            ),
            "2": FERNNLevel(
                input_channels=latent_channels,
                hidden_channels=hidden_channels,
                v_axis_size=v_axis_size,
                kernel_size=kernel_size,
                residual_scale=residual_scale,
            ),
        })

    def build_initial_state(
        self,
        x0_ctx: torch.Tensor,
        x1_ctx: torch.Tensor,
        x2_ctx: torch.Tensor,
    ) -> FERNNState:
        B = x0_ctx.shape[0]
        device = x0_ctx.device
        dtype = x0_ctx.dtype

        states = {
            0: self.level_models["0"].build_initial_state(
                batch_size=B,
                h=x0_ctx.shape[-2],
                w=x0_ctx.shape[-1],
                device=device,
                dtype=dtype,
            ),
            1: self.level_models["1"].build_initial_state(
                batch_size=B,
                h=x1_ctx.shape[-2],
                w=x1_ctx.shape[-1],
                device=device,
                dtype=dtype,
            ),
            2: self.level_models["2"].build_initial_state(
                batch_size=B,
                h=x2_ctx.shape[-2],
                w=x2_ctx.shape[-1],
                device=device,
                dtype=dtype,
            ),
        }

        return FERNNState(levels=states, t=0)

    def step(
        self,
        x0_t: torch.Tensor,
        x1_t: torch.Tensor,
        x2_t: torch.Tensor,
        state_t: FERNNState,
        return_probe: bool = False,
    ) -> tuple[dict[int, torch.Tensor], FERNNState, FERNNStepAux]:
        x_by_level = {
            0: x0_t,
            1: x1_t,
            2: x2_t,
        }

        z_next: dict[int, torch.Tensor] = {}
        next_states: dict[int, torch.Tensor] = {}
        aux = FERNNStepAux()

        for level in self.levels:
            z_l, h_l = self.level_models[str(level)].step(
                x_t=x_by_level[level],
                h_t=state_t.levels[level],
            )
            z_next[level] = z_l
            next_states[level] = h_l

            if return_probe:
                aux.extra[f"level_{level}_state_norm"] = h_l.detach().pow(2).mean().sqrt()
                aux.extra[f"level_{level}_z_norm"] = z_l.detach().pow(2).mean().sqrt()

        state_next = FERNNState(
            levels=next_states,
            t=state_t.t + 1,
        )

        return z_next, state_next, aux

    def scan_context(
        self,
        x0_ctx: torch.Tensor,
        x1_ctx: torch.Tensor,
        x2_ctx: torch.Tensor,
        state0: FERNNState | None = None,
        return_probe: bool = False,
    ) -> tuple[FERNNState, list[FERNNStepAux]]:
        state = self.build_initial_state(x0_ctx, x1_ctx, x2_ctx) if state0 is None else state0

        aux_list: list[FERNNStepAux] = []
        T = x0_ctx.shape[1]

        for t in range(T):
            _, state, aux = self.step(
                x0_t=x0_ctx[:, t],
                x1_t=x1_ctx[:, t],
                x2_t=x2_ctx[:, t],
                state_t=state,
                return_probe=return_probe,
            )
            aux_list.append(aux)

        return state, aux_list

    def rollout_pyramid(
        self,
        x0_ctx: torch.Tensor,
        x1_ctx: torch.Tensor | None,
        x2_ctx: torch.Tensor | None,
        pred_steps: int,
        teacher_forcing_pyramid: dict[int, torch.Tensor] | None = None,
        state0: FERNNState | None = None,
        ablation_cfg=None,
        return_probe: bool = False,
    ) -> tuple[dict[int, torch.Tensor], list[FERNNStepAux]]:
        if x1_ctx is None or x2_ctx is None:
            raise ValueError("FERNNCore expects x0_ctx, x1_ctx, x2_ctx from PyramidEncoder.")

        state, aux_ctx = self.scan_context(
            x0_ctx=x0_ctx,
            x1_ctx=x1_ctx,
            x2_ctx=x2_ctx,
            state0=state0,
            return_probe=return_probe,
        )

        last_x0 = x0_ctx[:, -1]
        last_x1 = x1_ctx[:, -1]
        last_x2 = x2_ctx[:, -1]

        z0s: list[torch.Tensor] = []
        z1s: list[torch.Tensor] = []
        z2s: list[torch.Tensor] = []
        aux_roll: list[FERNNStepAux] = []

        for k in range(pred_steps):
            z_next, state, aux = self.step(
                x0_t=last_x0,
                x1_t=last_x1,
                x2_t=last_x2,
                state_t=state,
                return_probe=return_probe,
            )

            z0s.append(z_next[0])
            z1s.append(z_next[1])
            z2s.append(z_next[2])
            aux_roll.append(aux)

            if teacher_forcing_pyramid is not None and teacher_forcing_pyramid.get(0, None) is not None:
                last_x0 = teacher_forcing_pyramid[0][:, k]
                last_x1 = teacher_forcing_pyramid[1][:, k]
                last_x2 = teacher_forcing_pyramid[2][:, k]
            else:
                last_x0 = z_next[0]
                last_x1 = z_next[1]
                last_x2 = z_next[2]

        z_pyr = {
            0: torch.stack(z0s, dim=1),
            1: torch.stack(z1s, dim=1),
            2: torch.stack(z2s, dim=1),
        }

        return z_pyr, aux_ctx + aux_roll

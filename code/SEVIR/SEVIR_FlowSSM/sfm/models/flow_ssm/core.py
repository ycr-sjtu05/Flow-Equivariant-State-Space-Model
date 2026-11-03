from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.models.flow_ssm.specs import PairSpec
from sfm.models.flow_ssm.param_gen import LevelParamGenerator
from sfm.models.flow_ssm.strict_update import StrictPairUpdater
from sfm.models.flow_ssm.readout import StrictReadout


@dataclass
class FlowSSMState:

    levels: dict[int, torch.Tensor]
    t: int = 0


@dataclass
class FlowStepAux:
    pair_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def build_pair_specs(
    v_axis_size: int = 3,
    state_dims: dict[int, int] | None = None,
) -> dict[int, PairSpec]:


    if state_dims is None:
        state_dims = {
            0: 64,
            1: 80,
            2: 96,
        }

    v_size = v_axis_size * v_axis_size

    return {
        0: PairSpec(
            name="p00_plain",
            m=0,
            ell=0,
            v_size=v_size,
            h=64,
            w=64,
            state_dim=state_dims[0],
            strict=True,
            parent_name=None,
        ),
        1: PairSpec(
            name="p11_plain",
            m=1,
            ell=1,
            v_size=v_size,
            h=32,
            w=32,
            state_dim=state_dims[1],
            strict=True,
            parent_name=None,
        ),
        2: PairSpec(
            name="p22_plain",
            m=2,
            ell=2,
            v_size=v_size,
            h=16,
            w=16,
            state_dim=state_dims[2],
            strict=True,
            parent_name=None,
        ),
    }


class FlowSSMCore(nn.Module):


    def __init__(
        self,
        input_channels: dict[int, int],
        output_channels: int,
        feat_dim: int = 64,
        v_axis_size: int = 3,
        state_dims: dict[int, int] | None = None,
        residual_latent: bool = True,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.levels = (0, 1, 2)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_latent = residual_latent
        self.residual_scale = residual_scale

        self.pair_specs = build_pair_specs(
            v_axis_size=v_axis_size,
            state_dims=state_dims,
        )

        self.param_gens = nn.ModuleDict()
        self.updaters = nn.ModuleDict()
        self.readouts = nn.ModuleDict()

        for level in self.levels:
            pair = self.pair_specs[level]
            level_key = str(level)

            self.param_gens[level_key] = LevelParamGenerator(
                in_channels=input_channels[level],
                feat_dim=feat_dim,
                u_dim=input_channels[level],
            )

            self.updaters[level_key] = StrictPairUpdater(
                pair_spec=pair,
                feat_dim=feat_dim,
                u_dim=input_channels[level],
            )

            self.readouts[level_key] = StrictReadout(
                pair_spec=pair,
                out_channels=output_channels,
            )

    def build_initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> FlowSSMState:
        levels: dict[int, torch.Tensor] = {}

        for level in self.levels:
            pair = self.pair_specs[level]
            levels[level] = torch.zeros(
                batch_size,
                pair.v_size,
                pair.state_dim,
                pair.h,
                pair.w,
                device=device,
                dtype=dtype,
            )

        return FlowSSMState(levels=levels, t=0)

    def _lift_input(self, u: torch.Tensor, pair: PairSpec) -> torch.Tensor:


        return u.unsqueeze(1).expand(-1, pair.v_size, -1, -1, -1).contiguous()

    def step(
        self,
        x0_t: torch.Tensor,
        state_t: FlowSSMState,
        x1_t: torch.Tensor | None = None,
        x2_t: torch.Tensor | None = None,
        return_probe: bool = True,
    ) -> tuple[dict[int, torch.Tensor], FlowSSMState, FlowStepAux]:
        if x1_t is None:
            x1_t = F.interpolate(x0_t, size=(32, 32), mode="area")
        if x2_t is None:
            x2_t = F.interpolate(x1_t, size=(16, 16), mode="area")

        x_by_level = {
            0: x0_t,
            1: x1_t,
            2: x2_t,
        }

        next_levels: dict[int, torch.Tensor] = {}
        z_pyramid: dict[int, torch.Tensor] = {}
        aux = FlowStepAux()

        for level in self.levels:
            level_key = str(level)
            pair = self.pair_specs[level]
            x_level = x_by_level[level]

            params = self.param_gens[level_key](x_level)
            lift = self._lift_input(params.u_proj, pair)
            msg = torch.zeros_like(state_t.levels[level])

            state_next = self.updaters[level_key](
                prev_state=state_t.levels[level],
                lift=lift,
                msg=msg,
                params=params,
            )

            raw = self.readouts[level_key](state_next)

            if self.residual_latent:


                z_level = x_level + self.residual_scale * torch.tanh(raw)
            else:
                z_level = raw

            next_levels[level] = state_next
            z_pyramid[level] = z_level

            if return_probe:
                aux.pair_outputs[pair.name] = raw.detach()
                aux.extra[f"level_{level}_raw"] = raw.detach()
                aux.extra[f"level_{level}_z"] = z_level.detach()

        state_next_all = FlowSSMState(
            levels=next_levels,
            t=state_t.t + 1,
        )
        return z_pyramid, state_next_all, aux

    def scan_context(
        self,
        x0_ctx: torch.Tensor,
        x1_ctx: torch.Tensor | None = None,
        x2_ctx: torch.Tensor | None = None,
        state0: FlowSSMState | None = None,
        return_probe: bool = True,
    ) -> tuple[FlowSSMState, list[FlowStepAux]]:
        batch_size = x0_ctx.shape[0]
        state = (
            self.build_initial_state(
                batch_size=batch_size,
                device=x0_ctx.device,
                dtype=x0_ctx.dtype,
            )
            if state0 is None
            else state0
        )

        aux_list: list[FlowStepAux] = []
        T = x0_ctx.shape[1]

        for t in range(T):
            x1_t = x1_ctx[:, t] if x1_ctx is not None else None
            x2_t = x2_ctx[:, t] if x2_ctx is not None else None

            _, state, aux = self.step(
                x0_t=x0_ctx[:, t],
                x1_t=x1_t,
                x2_t=x2_t,
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
        state0: FlowSSMState | None = None,
        return_probe: bool = True,
    ) -> tuple[dict[int, torch.Tensor], list[FlowStepAux]]:
        state, aux_ctx = self.scan_context(
            x0_ctx=x0_ctx,
            x1_ctx=x1_ctx,
            x2_ctx=x2_ctx,
            state0=state0,
            return_probe=return_probe,
        )

        last_x0 = x0_ctx[:, -1]
        last_x1 = x1_ctx[:, -1] if x1_ctx is not None else None
        last_x2 = x2_ctx[:, -1] if x2_ctx is not None else None

        z0s: list[torch.Tensor] = []
        z1s: list[torch.Tensor] = []
        z2s: list[torch.Tensor] = []
        aux_roll: list[FlowStepAux] = []

        for k in range(pred_steps):
            z_pyr_t, state, aux = self.step(
                x0_t=last_x0,
                x1_t=last_x1,
                x2_t=last_x2,
                state_t=state,
                return_probe=return_probe,
            )

            z0s.append(z_pyr_t[0])
            z1s.append(z_pyr_t[1])
            z2s.append(z_pyr_t[2])
            aux_roll.append(aux)

            if teacher_forcing_pyramid is not None and teacher_forcing_pyramid.get(0, None) is not None:
                last_x0 = teacher_forcing_pyramid[0][:, k]
                last_x1 = (
                    teacher_forcing_pyramid[1][:, k]
                    if teacher_forcing_pyramid.get(1, None) is not None
                    else None
                )
                last_x2 = (
                    teacher_forcing_pyramid[2][:, k]
                    if teacher_forcing_pyramid.get(2, None) is not None
                    else None
                )
            else:
                last_x0 = z_pyr_t[0]
                last_x1 = z_pyr_t[1]
                last_x2 = z_pyr_t[2]

        z_pyr = {
            0: torch.stack(z0s, dim=1),
            1: torch.stack(z1s, dim=1),
            2: torch.stack(z2s, dim=1),
        }

        return z_pyr, aux_ctx + aux_roll

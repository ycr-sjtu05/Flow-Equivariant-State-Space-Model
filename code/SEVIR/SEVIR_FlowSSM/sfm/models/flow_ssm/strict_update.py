from __future__ import annotations

import torch
import torch.nn as nn

from .specs import PairSpec
from .param_gen import LevelParams


class StrictPairUpdater(nn.Module):


    def __init__(
        self,
        pair_spec: PairSpec,
        feat_dim: int,
        u_dim: int,
    ) -> None:
        super().__init__()
        self.pair_spec = pair_spec
        self.state_dim = pair_spec.state_dim

        self.gate_head = nn.Conv2d(feat_dim, 2 * self.state_dim, kernel_size=1)
        self.u_proj = nn.LazyLinear(self.state_dim)

    def _project_u(self, lift: torch.Tensor) -> torch.Tensor:

        y = lift.permute(0, 1, 3, 4, 2).contiguous()
        y = self.u_proj(y)
        y = y.permute(0, 1, 4, 2, 3).contiguous()
        return y

    def forward(
        self,
        prev_state: torch.Tensor,
        lift: torch.Tensor,
        msg: torch.Tensor,
        params: LevelParams,
    ) -> torch.Tensor:
        gates = self.gate_head(params.feat)
        a_logit, b_logit = gates.chunk(2, dim=1)

        a = torch.sigmoid(a_logit).unsqueeze(1)
        b = torch.sigmoid(b_logit).unsqueeze(1)

        u_term = self._project_u(lift)
        next_state = a * prev_state + b * u_term + msg
        return next_state

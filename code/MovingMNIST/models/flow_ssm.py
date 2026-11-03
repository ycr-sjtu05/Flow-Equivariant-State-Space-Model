"""Flow-equivariant state-space predictor for Moving MNIST."""

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .transport import translate_branches


class FlowSSMCell(nn.Module):
    """Transport branch memory before an input-selective state-space update.

    Input features: (B, D, H, W); state: (B, V, N, H, W).
    Branch order is (vx outer, vy inner).
    """

    def __init__(self, d_model: int, d_state: int, v_range: int = 2):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 (GroupNorm).")
        if d_state <= 0 or v_range < 0:
            raise ValueError("d_state must be positive and v_range non-negative.")

        self.D = d_model
        self.N = d_state
        self.v_list = [(vx, vy) for vx in range(-v_range, v_range + 1)
                       for vy in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        self.norm = nn.GroupNorm(4, d_model)
        self.conv_delta = nn.Conv2d(d_model, d_state, 3, padding=1, padding_mode="circular")
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1)))
        self.conv_B = nn.Conv2d(d_model, d_state * d_model, 3, padding=1,
                                padding_mode="circular", bias=False)
        self.conv_C = nn.Conv2d(d_model, d_model * d_state, 3, padding=1,
                                padding_mode="circular", bias=False)
        self.state_mix = nn.Conv2d(d_state, d_state, 3, padding=1,
                                   groups=d_state, padding_mode="circular", bias=False)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_state)))
        self.D_param = nn.Parameter(torch.ones(d_model))

    def forward(self, u_t: torch.Tensor, s_prev: torch.Tensor):
        batch, _, height, width = u_t.shape
        u_t_norm = self.norm(u_t)

        delta = F.softplus(self.conv_delta(u_t_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=1e-4, max=5.0)
        b_val = self.conv_B(u_t_norm).view(batch, self.N, self.D, height, width)
        c_val = self.conv_C(u_t_norm).view(batch, self.D, self.N, height, width)

        a = -torch.exp(self.log_A_real)
        a_bar = torch.exp(delta * a.view(1, self.N, 1, 1))
        bu = torch.einsum("bndhw,bdhw->bnhw", b_val, u_t)
        b_bar_u = delta * bu

        transported = translate_branches(s_prev, self.v_list)
        mixed = self.state_mix(transported.reshape(batch * self.num_v, self.N, height, width))
        mixed = mixed.reshape(batch, self.num_v, self.N, height, width)
        s_new = a_bar.unsqueeze(1) * mixed + b_bar_u.unsqueeze(1)

        y_state = torch.einsum("bdnhw,bvnhw->bvdhw", c_val, s_new)
        y_skip = u_t.unsqueeze(1) * self.D_param.view(1, 1, self.D, 1, 1)
        return y_state + y_skip, s_new


class FlowSSMPredictor(nn.Module):
    """Autoregressive predictor with transported selective SSM states."""

    def __init__(self, input_channels: int = 1, D: int = 64, N: int = 32,
                 height: int = 28, width: int = 28, output_channels: int = 1,
                 v_range: int = 2, pool_type: str = "max", decoder_conv_layers: int = 3):
        super().__init__()
        if pool_type not in ("max", "mean"):
            raise ValueError("pool_type must be 'max' or 'mean'.")
        if decoder_conv_layers < 0:
            raise ValueError("decoder_conv_layers must be non-negative.")
        self.height = height
        self.width = width
        self.pool_type = pool_type
        self.N = N
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, D, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(4, D),
            nn.ReLU(),
        )
        self.cell = FlowSSMCell(D, N, v_range)
        self.num_v = self.cell.num_v
        decoder = []
        for _ in range(decoder_conv_layers):
            decoder.extend([
                nn.Conv2d(D, D, 3, padding=1, padding_mode="circular", bias=False),
                nn.GroupNorm(4, D),
                nn.ReLU(),
            ])
        decoder.append(nn.Conv2d(D, output_channels, 3, padding=1,
                                 padding_mode="circular", bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input_seq: torch.Tensor, pred_len: int,
                teacher_forcing_ratio: float = 0.0, target_seq=None,
                return_hidden: bool = False) -> torch.Tensor:
        del return_hidden
        if input_seq.ndim != 5 or input_seq.size(1) < 1 or pred_len < 1:
            raise ValueError("Expected (B,T,C,H,W) input with T>=1 and pred_len>=1.")
        batch, t_in, _, height, width = input_seq.shape
        state = input_seq.new_zeros(batch, self.num_v, self.N, height, width)

        for t in range(t_in):
            _, state = self.cell(self.encoder(input_seq[:, t]), state)

        previous = input_seq[:, -1]
        predictions = []
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = previous.detach()
            y_branch, state = self.cell(self.encoder(frame), state)
            features = y_branch.max(dim=1)[0] if self.pool_type == "max" else y_branch.mean(dim=1)
            previous = torch.sigmoid(self.decoder(features))
            predictions.append(previous)
        return torch.stack(predictions, dim=1)

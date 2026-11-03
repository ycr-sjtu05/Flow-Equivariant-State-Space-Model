import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces, nn as e2nn

def build_rot_equivariant_conv(in_type, out_channels: int, kernel_size: int, bias: bool = False):
    gspace = in_type.gspace
    out_type = e2nn.FieldType(gspace, out_channels * [gspace.regular_repr])
    padding = kernel_size // 2
    conv = e2nn.R2Conv(
        in_type,
        out_type,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
    )
    return conv, out_type

def regular_field_type(gspace, channels: int):
    return e2nn.FieldType(gspace, channels * [gspace.regular_repr])

def scalar_field_type(gspace, channels: int):
    return e2nn.FieldType(gspace, channels * [gspace.trivial_repr])

class RotationFlowSSMCell(nn.Module):

    def __init__(
        self,
        d_model: int,
        v_list: List[int],
        N_rot: int,
        h_kernel_size: int = 3,
        u_kernel_size: int = 3,
        delta_min: float = 1e-4,
        delta_max: float = 5.0,
    ):
        super().__init__()
        self.D = d_model
        self.N_rot = N_rot
        self.v_list = list(v_list)
        self.num_v = len(self.v_list)

        self.delta_min = delta_min
        self.delta_max = delta_max

        self.gspace = gspaces.rot2dOnR2(N=N_rot)
        self.reg_type = regular_field_type(self.gspace, self.D)
        self.scalar_type = scalar_field_type(self.gspace, self.D)

        self.total_hidden_channels = self.D * self.N_rot

        self.input_norm = e2nn.InnerBatchNorm(self.reg_type)

        self.delta_head = e2nn.R2Conv(
            self.reg_type, self.scalar_type,
            kernel_size=u_kernel_size, padding=u_kernel_size // 2, bias=True
        )
        self.b_head = e2nn.R2Conv(
            self.reg_type, self.scalar_type,
            kernel_size=u_kernel_size, padding=u_kernel_size // 2, bias=True
        )
        self.c_head = e2nn.R2Conv(
            self.reg_type, self.scalar_type,
            kernel_size=u_kernel_size, padding=u_kernel_size // 2, bias=True
        )

        self.mix_conv1 = e2nn.R2Conv(
            self.reg_type, self.reg_type,
            kernel_size=h_kernel_size, padding=h_kernel_size // 2, bias=False
        )
        self.mix_bn1 = e2nn.InnerBatchNorm(self.reg_type)
        self.mix_relu1 = e2nn.ReLU(self.reg_type, inplace=True)

        self.mix_conv2 = e2nn.R2Conv(
            self.reg_type, self.reg_type,
            kernel_size=h_kernel_size, padding=h_kernel_size // 2, bias=False
        )
        self.mix_bn2 = e2nn.InnerBatchNorm(self.reg_type)

        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(self.D)))

        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self.skip_scale = nn.Parameter(torch.ones(self.D))

    def _flat_to_group(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_flat.shape
        assert C == self.total_hidden_channels, (
            f"Expected {self.total_hidden_channels} channels, got {C}"
        )
        return x_flat.view(B, self.D, self.N_rot, H, W)

    def _group_to_flat(self, x_group: torch.Tensor) -> torch.Tensor:
        B, D, N, H, W = x_group.shape
        assert D == self.D and N == self.N_rot
        return x_group.reshape(B, self.total_hidden_channels, H, W)

    def _scalar_times_regular(self, scalar_gate: torch.Tensor, reg_flat: torch.Tensor) -> torch.Tensor:
        reg_group = self._flat_to_group(reg_flat)
        out = scalar_gate.unsqueeze(2) * reg_group
        return self._group_to_flat(out)

    def _channel_scale_regular(self, scale: torch.Tensor, reg_flat: torch.Tensor) -> torch.Tensor:
        reg_group = self._flat_to_group(reg_flat)
        out = scale.view(1, self.D, 1, 1, 1) * reg_group
        return self._group_to_flat(out)

    def act_on_hidden(self, tensor: torch.Tensor, k_deg: float) -> torch.Tensor:
        B, C, H, W = tensor.shape
        assert C == self.total_hidden_channels

        C0 = C // self.N_rot
        tensor = tensor.view(B, C0, self.N_rot, H, W)

        g_idx = int(k_deg * self.N_rot / 360.0)
        tensor = torch.roll(tensor, shifts=g_idx, dims=2)
        tensor = tensor.view(B, C, H, W)

        if abs(k_deg % 90) < 1e-8:
            k = int(round(k_deg / 90.0)) % 4
            if k == 0:
                return tensor
            return torch.rot90(tensor, k=k, dims=(2, 3))

        angle_rad = math.radians(k_deg)
        rot_mat = tensor.new_tensor([
            [ math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [ math.sin(angle_rad),  math.cos(angle_rad), 0.0],
        ])
        rot_mat = rot_mat.unsqueeze(0).expand(B, -1, -1)

        grid = F.affine_grid(rot_mat, tensor.size(), align_corners=False)
        return F.grid_sample(
            tensor,
            grid,
            align_corners=False,
            padding_mode="zeros",
        )

    def _mix_state(self, s_flat: torch.Tensor) -> torch.Tensor:
        s_gt = e2nn.GeometricTensor(s_flat, self.reg_type)
        residual = s_gt.tensor

        x = self.mix_conv1(s_gt)
        x = self.mix_bn1(x)
        x = self.mix_relu1(x)
        x = self.mix_conv2(x)
        x = self.mix_bn2(x)

        return residual + x.tensor

    def forward(self, u_t_flat: torch.Tensor, s_prev: torch.Tensor):
        B, C, H, W = u_t_flat.shape
        assert C == self.total_hidden_channels
        assert s_prev.shape == (B, self.num_v, self.total_hidden_channels, H, W)

        u_gt = e2nn.GeometricTensor(u_t_flat, self.reg_type)
        u_gt = self.input_norm(u_gt)
        u_t_flat = u_gt.tensor

        delta = self.delta_head(u_gt).tensor
        delta = F.softplus(delta + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=self.delta_min, max=self.delta_max)

        b_gate = torch.sigmoid(self.b_head(u_gt).tensor)
        c_gate = torch.sigmoid(self.c_head(u_gt).tensor)

        A = -torch.exp(self.log_A_real)
        alpha = torch.exp(delta * A.view(1, self.D, 1, 1))

        Bu_flat = self._scalar_times_regular(b_gate, u_t_flat)
        skip_flat = self._channel_scale_regular(self.skip_scale, u_t_flat)

        y_branches = []
        s_branches = []

        for i, w in enumerate(self.v_list):
            s_i = s_prev[:, i]

            s_shift = self.act_on_hidden(s_i, w)

            s_mixed = self._mix_state(s_shift)

            s_new_i = self._scalar_times_regular(alpha, s_mixed) + Bu_flat

            y_i = self._scalar_times_regular(c_gate, s_new_i) + skip_flat

            s_branches.append(s_new_i)
            y_branches.append(y_i)

        s_new = torch.stack(s_branches, dim=1)
        y_all = torch.stack(y_branches, dim=1)
        return y_all, s_new

class RotationFlowSSM(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        D: int = 32,

        height: int = 34,
        width: int = 34,
        output_channels: int = 1,
        v_list: List[int] = (-40, -20, 0, 20, 40),
        N_rot: int = 8,
        pool_type: str = "max",
        decoder_conv_layers: int = 1,
        kernel_size: int = 3,
        delta_min: float = 1e-4,
        delta_max: float = 5.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.D = D
        self.N_rot = N_rot
        self.pool_type = pool_type
        self.output_channels = output_channels

        self.gspace = gspaces.rot2dOnR2(N=N_rot)

        self.input_type = scalar_field_type(self.gspace, input_channels)
        self.reg_type = regular_field_type(self.gspace, D)
        self.total_hidden_channels = D * N_rot

        self.enc_conv, enc_type = build_rot_equivariant_conv(
            self.input_type, D, kernel_size, bias=False
        )
        self.encoder = e2nn.SequentialModule(
            self.enc_conv,
            e2nn.InnerBatchNorm(enc_type),
            e2nn.ReLU(enc_type, inplace=True),
        )

        self.cell = RotationFlowSSMCell(
            d_model=D,
            v_list=list(v_list),
            N_rot=N_rot,
            h_kernel_size=kernel_size,
            u_kernel_size=kernel_size,
            delta_min=delta_min,
            delta_max=delta_max,
        )
        self.num_v = self.cell.num_v

        layers = []
        for _ in range(decoder_conv_layers):
            layers += [
                nn.Conv2d(
                    self.total_hidden_channels,
                    self.total_hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(4, self.total_hidden_channels),
                nn.ReLU(),
            ]
        layers += [
            nn.Conv2d(
                self.total_hidden_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        ]
        self.decoder_conv = nn.Sequential(*layers)

    def _pool_branches(self, y_all: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "max":
            return y_all.max(dim=1)[0]
        elif self.pool_type == "mean":
            return y_all.mean(dim=1)
        elif self.pool_type == "sum":
            return y_all.sum(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def forward(
        self,
        input_seq: torch.Tensor,
        pred_len: int,
        teacher_forcing_ratio: float = 0.0,
        target_seq: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        batch, T_in, C_in, H, W = input_seq.size()
        device = input_seq.device

        s_t = torch.zeros(
            batch,
            self.num_v,
            self.total_hidden_channels,
            H,
            W,
            device=device,
        )

        for t in range(T_in):
            x_t = input_seq[:, t]
            u_t = self.encoder(e2nn.GeometricTensor(x_t, self.input_type)).tensor
            _, s_t = self.cell(u_t, s_t)

        prev_x = input_seq[:, -1]
        outputs = []

        for t in range(pred_len):
            if self.training and (target_seq is not None) and (random.random() < teacher_forcing_ratio):
                frame = target_seq[:, t]
            else:
                frame = prev_x.detach()

            u_t = self.encoder(e2nn.GeometricTensor(frame, self.input_type)).tensor
            y_t, s_t = self.cell(u_t, s_t)

            feat = self._pool_branches(y_t)
            out = self.decoder_conv(feat)

            out = torch.sigmoid(out)

            outputs.append(out)
            prev_x = out

        outputs = torch.stack(outputs, dim=1)

        if return_hidden:
            return outputs, s_t
        return outputs

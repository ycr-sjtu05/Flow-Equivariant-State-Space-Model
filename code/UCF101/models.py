from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Velocity = Tuple[int, int]
CoreName = Literal["mamba", "flow_mamba", "fernn"]
BoundaryMode = Literal["zero", "circular"]


def make_velocity_grid(v_range: int) -> Tuple[Velocity, ...]:
    if v_range < 0:
        raise ValueError(f"v_range must be nonnegative, got {v_range}.")
    return tuple(
        (vx, vy)
        for vx in range(-v_range, v_range + 1)
        for vy in range(-v_range, v_range + 1)
    )


def choose_group_count(channels: int, max_groups: int = 8) -> int:
    if channels <= 0:
        raise ValueError("channels must be positive.")
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def shift_2d(
    x: torch.Tensor,
    vx: int,
    vy: int,
    boundary: BoundaryMode,
) -> torch.Tensor:
    """
    Shift content by (vx, vy).

    
    """
    if boundary == "circular":
        return torch.roll(x, shifts=(vy, vx), dims=(-2, -1))
    if boundary != "zero":
        raise ValueError(f"Unknown boundary mode: {boundary}.")

    height, width = x.shape[-2:]
    out = torch.zeros_like(x)

    dst_x0 = max(vx, 0)
    dst_x1 = width + min(vx, 0)
    src_x0 = max(-vx, 0)
    src_x1 = width - max(vx, 0)

    dst_y0 = max(vy, 0)
    dst_y1 = height + min(vy, 0)
    src_y0 = max(-vy, 0)
    src_y1 = height - max(vy, 0)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return out

    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = (
        x[..., src_y0:src_y1, src_x0:src_x1]
    )
    return out


class FrameEncoder(nn.Module):
    """64x64 RGB -> D x 16 x 16 by default."""

    def __init__(
        self,
        input_channels: int,
        d_model: int,
        base_channels: int = 32,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=False,
            ),
            nn.GroupNorm(
                choose_group_count(base_channels),
                base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                bias=False,
            ),
            nn.GroupNorm(
                choose_group_count(d_model),
                d_model,
            ),
            nn.SiLU(),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.net(frame)


class FrameDecoder(nn.Module):
    """D x H/4 x W/4 -> RGB frame."""

    def __init__(
        self,
        d_model: int,
        output_channels: int,
        hidden_channels: int = 32,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            d_model,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            choose_group_count(hidden_channels),
            hidden_channels,
        )
        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            choose_group_count(hidden_channels),
            hidden_channels,
        )
        self.output = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )

    def forward(
        self,
        feature: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            feature,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        x = F.silu(self.norm2(self.conv2(x)))
        return torch.sigmoid(self.output(x))


class SelectiveFields(nn.Module):
    """
    Generate input-selective Delta, B, and C fields.

    Delta: (B, N, H, W)
    B:     (B, N, D, H, W)
    C:     (B, D, N, H, W)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dt_init: float = 1.0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = nn.GroupNorm(
            choose_group_count(d_model),
            d_model,
        )
        self.delta_proj = nn.Conv2d(
            d_model,
            d_state,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.B_proj = nn.Conv2d(
            d_model,
            d_state * d_model,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.C_proj = nn.Conv2d(
            d_model,
            d_model * d_state,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )

        inv_softplus = math.log(math.expm1(dt_init))
        self.delta_bias = nn.Parameter(
            torch.full((d_state,), inv_softplus)
        )

    def forward(self, u: torch.Tensor):
        batch, _, height, width = u.shape
        x = self.norm(u)

        delta = F.softplus(
            self.delta_proj(x)
            + self.delta_bias.view(1, self.d_state, 1, 1)
        ).clamp_(1e-4, 5.0)

        B_value = self.B_proj(x).reshape(
            batch,
            self.d_state,
            self.d_model,
            height,
            width,
        )
        C_value = self.C_proj(x).reshape(
            batch,
            self.d_model,
            self.d_state,
            height,
            width,
        )
        return delta, B_value, C_value


class SpatialMambaCore(nn.Module):
    """
    Fixed-coordinate spatial selective SSM without state transport.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.fields = SelectiveFields(
            d_model,
            d_state,
            padding_mode=padding_mode,
        )
        self.log_A = nn.Parameter(
            torch.full((d_state,), math.log(0.5))
        )
        self.state_mix = nn.Conv2d(
            d_state,
            d_state,
            kernel_size=3,
            padding=1,
            groups=d_state,
            padding_mode=padding_mode,
            bias=False,
        )
        self.D_skip = nn.Parameter(torch.ones(d_model))
        self._identity_init()

    def _identity_init(self):
        with torch.no_grad():
            self.state_mix.weight.zero_()
            self.state_mix.weight[:, 0, 1, 1] = 1.0

    def initial_state(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.d_state,
            height,
            width,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
        return_aux: bool = False,
    ):
        delta, B_value, C_value = self.fields(u)

        A = -torch.exp(self.log_A).view(
            1, self.d_state, 1, 1
        )
        A_bar = torch.exp(delta * A)

        Bu = torch.einsum(
            "bndhw,bdhw->bnhw",
            B_value,
            u,
        )
        B_bar_u = delta * Bu

        state = (
            A_bar * self.state_mix(state)
            + B_bar_u
        )
        y_state = torch.einsum(
            "bdnhw,bnhw->bdhw",
            C_value,
            state,
        )
        feature = (
            y_state
            + u * self.D_skip.view(1, self.d_model, 1, 1)
        )

        aux = {}
        if return_aux:
            aux["state_energy"] = state.abs().mean(
                dim=(1, 2, 3)
            )
        return feature, state, aux


class FlowMambaCore(nn.Module):
    """
    Velocity-lifted selective SSM with transported recurrent states.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        velocities: Sequence[Velocity],
        boundary: BoundaryMode = "zero",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if len(velocities) == 0:
            raise ValueError("At least one velocity is required.")

        self.d_model = d_model
        self.d_state = d_state
        self.velocity_list = tuple(
            (int(vx), int(vy)) for vx, vy in velocities
        )
        self.num_velocities = len(self.velocity_list)
        self.boundary = boundary

        self.register_buffer(
            "velocities",
            torch.tensor(
                self.velocity_list,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.fields = SelectiveFields(
            d_model,
            d_state,
            padding_mode=padding_mode,
        )
        self.log_A = nn.Parameter(
            torch.full((d_state,), math.log(0.5))
        )
        self.state_mix = nn.Conv2d(
            d_state,
            d_state,
            kernel_size=3,
            padding=1,
            groups=d_state,
            padding_mode=padding_mode,
            bias=False,
        )
        self.D_skip = nn.Parameter(torch.ones(d_model))
        self._identity_init()

    def _identity_init(self):
        with torch.no_grad():
            self.state_mix.weight.zero_()
            self.state_mix.weight[:, 0, 1, 1] = 1.0

    def initial_state(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.num_velocities,
            self.d_state,
            height,
            width,
            device=device,
            dtype=dtype,
        )

    def _transport(self, state: torch.Tensor) -> torch.Tensor:
        branches = [
            shift_2d(
                state[:, branch],
                vx=vx,
                vy=vy,
                boundary=self.boundary,
            )
            for branch, (vx, vy) in enumerate(
                self.velocity_list
            )
        ]
        return torch.stack(branches, dim=1)

    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
        return_aux: bool = False,
    ):
        batch, _, height, width = u.shape
        delta, B_value, C_value = self.fields(u)

        A = -torch.exp(self.log_A).view(
            1, self.d_state, 1, 1
        )
        A_bar = torch.exp(delta * A)

        Bu = torch.einsum(
            "bndhw,bdhw->bnhw",
            B_value,
            u,
        )
        B_bar_u = delta * Bu

        transported = self._transport(state)
        mixed = self.state_mix(
            transported.reshape(
                batch * self.num_velocities,
                self.d_state,
                height,
                width,
            )
        ).reshape(
            batch,
            self.num_velocities,
            self.d_state,
            height,
            width,
        )

        state = (
            A_bar.unsqueeze(1) * mixed
            + B_bar_u.unsqueeze(1)
        )

        branch_output = torch.einsum(
            "bdnhw,bvnhw->bvdhw",
            C_value,
            state,
        )
        branch_output = (
            branch_output
            + u.unsqueeze(1)
            * self.D_skip.view(1, 1, self.d_model, 1, 1)
        )
        feature = branch_output.max(dim=1).values

        aux = {}
        if return_aux:
            aux["branch_energy"] = branch_output.abs().mean(
                dim=(2, 3, 4)
            )
            aux["velocities"] = self.velocities
        return feature, state, aux


class FERNNCore(nn.Module):
    """
    Translation FERNN with independent velocity branches.

    For each velocity v:
        h_t^v = act(W_h * T_v(h_{t-1}^v) + W_x * u_t)

    The recurrent and input kernels are shared across all branches, the
    input lift is trivial, and the outputs are max-pooled over V.
    """

    def __init__(
        self,
        d_model: int,
        velocities: Sequence[Velocity],
        boundary: BoundaryMode = "zero",
        padding_mode: str = "zeros",
        activation: Literal["tanh", "relu"] = "tanh",
    ):
        super().__init__()
        if len(velocities) == 0:
            raise ValueError("At least one velocity is required.")
        if activation not in {"tanh", "relu"}:
            raise ValueError(f"Unknown activation: {activation}.")

        self.d_model = d_model
        self.velocity_list = tuple(
            (int(vx), int(vy)) for vx, vy in velocities
        )
        self.num_velocities = len(self.velocity_list)
        self.boundary = boundary
        self.activation = activation

        self.register_buffer(
            "velocities",
            torch.tensor(
                self.velocity_list,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.input_conv = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.recurrent_conv = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            choose_group_count(d_model),
            d_model,
        )

    def initial_state(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.num_velocities,
            self.d_model,
            height,
            width,
            device=device,
            dtype=dtype,
        )

    def _transport(self, state: torch.Tensor) -> torch.Tensor:
        branches = [
            shift_2d(
                state[:, branch],
                vx=vx,
                vy=vy,
                boundary=self.boundary,
            )
            for branch, (vx, vy) in enumerate(
                self.velocity_list
            )
        ]
        return torch.stack(branches, dim=1)

    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
        return_aux: bool = False,
    ):
        batch, channels, height, width = u.shape
        transported = self._transport(state)

        recurrent = self.recurrent_conv(
            transported.reshape(
                batch * self.num_velocities,
                channels,
                height,
                width,
            )
        )
        recurrent = recurrent.reshape(
            batch,
            self.num_velocities,
            channels,
            height,
            width,
        )
        injected = self.input_conv(u).unsqueeze(1)

        preactivation = recurrent + injected
        normalized = self.norm(
            preactivation.reshape(
                batch * self.num_velocities,
                channels,
                height,
                width,
            )
        ).reshape_as(preactivation)

        if self.activation == "tanh":
            state = torch.tanh(normalized)
        else:
            state = F.relu(normalized)

        feature = state.max(dim=1).values

        aux = {}
        if return_aux:
            aux["branch_energy"] = state.abs().mean(
                dim=(2, 3, 4)
            )
            aux["velocities"] = self.velocities
        return feature, state, aux


class VideoPredictor(nn.Module):
    """
    Shared encoder-decoder video predictor with one of three cores:
      - mamba
      - flow_mamba
      - fernn
    """

    def __init__(
        self,
        core_name: CoreName = "flow_mamba",
        input_channels: int = 3,
        output_channels: int = 3,
        d_model: int = 64,
        d_state: int = 16,
        v_range: int = 1,
        boundary: BoundaryMode = "zero",
        padding_mode: str = "zeros",
        encoder_base_channels: int = 32,
        decoder_hidden_channels: int = 32,
        fernn_activation: Literal["tanh", "relu"] = "tanh",
    ):
        super().__init__()
        if core_name not in {"mamba", "flow_mamba", "fernn"}:
            raise ValueError(f"Unknown core: {core_name}.")
        if input_channels != output_channels:
            raise ValueError(
                "Autoregressive prediction requires equal input/output channels."
            )

        self.core_name = core_name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.d_model = d_model
        self.d_state = d_state
        self.v_range = v_range
        self.boundary = boundary
        self.padding_mode = padding_mode
        self.encoder_base_channels = encoder_base_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.fernn_activation = fernn_activation

        self.encoder = FrameEncoder(
            input_channels=input_channels,
            d_model=d_model,
            base_channels=encoder_base_channels,
            padding_mode=padding_mode,
        )
        self.decoder = FrameDecoder(
            d_model=d_model,
            output_channels=output_channels,
            hidden_channels=decoder_hidden_channels,
            padding_mode=padding_mode,
        )

        velocities = make_velocity_grid(v_range)
        if core_name == "mamba":
            self.core = SpatialMambaCore(
                d_model=d_model,
                d_state=d_state,
                padding_mode=padding_mode,
            )
        elif core_name == "flow_mamba":
            self.core = FlowMambaCore(
                d_model=d_model,
                d_state=d_state,
                velocities=velocities,
                boundary=boundary,
                padding_mode=padding_mode,
            )
        else:
            self.core = FERNNCore(
                d_model=d_model,
                velocities=velocities,
                boundary=boundary,
                padding_mode=padding_mode,
                activation=fernn_activation,
            )

    def config_dict(self) -> Dict[str, object]:
        return {
            "core_name": self.core_name,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "v_range": self.v_range,
            "boundary": self.boundary,
            "padding_mode": self.padding_mode,
            "encoder_base_channels": self.encoder_base_channels,
            "decoder_hidden_channels": self.decoder_hidden_channels,
            "fernn_activation": self.fernn_activation,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, object],
    ) -> "VideoPredictor":
        return cls(**config)

    def _initial_state(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = latent.shape
        return self.core.initial_state(
            batch,
            height,
            width,
            device=latent.device,
            dtype=latent.dtype,
        )

    def forward(
        self,
        context: torch.Tensor,
        pred_len: int,
        *,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        detach_rollout: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            context: (B, T_in, C, H, W)
            target:  (B, T_pred, C, H, W), used only for teacher forcing
        """
        if context.ndim != 5:
            raise ValueError(
                "context must have shape (B,T,C,H,W)."
            )
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]."
            )

        batch, context_len, channels, out_h, out_w = context.shape
        if context_len < 1:
            raise ValueError("At least one context frame is required.")
        if channels != self.input_channels:
            raise ValueError("Unexpected input channel count.")

        first_latent = self.encoder(context[:, 0])
        state = self._initial_state(first_latent)

        # Process context frames 0 ... T_in-2.
        for t in range(context_len - 1):
            latent = (
                first_latent
                if t == 0
                else self.encoder(context[:, t])
            )
            _, state, _ = self.core.step(
                latent,
                state,
                return_aux=False,
            )

        decoder_input = context[:, -1]
        outputs = []
        auxiliary = []

        for step in range(pred_len):
            latent = self.encoder(decoder_input)
            feature, state, aux = self.core.step(
                latent,
                state,
                return_aux=return_aux,
            )
            prediction = self.decoder(
                feature,
                output_size=(out_h, out_w),
            )
            outputs.append(prediction)
            if return_aux:
                auxiliary.append(aux)

            next_input = prediction
            if (
                self.training
                and target is not None
                and teacher_forcing_ratio > 0.0
            ):
                mask = (
                    torch.rand(
                        batch, 1, 1, 1,
                        device=context.device,
                    )
                    < teacher_forcing_ratio
                )
                next_input = torch.where(
                    mask,
                    target[:, step],
                    prediction,
                )

            if detach_rollout:
                next_input = next_input.detach()
            decoder_input = next_input

        predictions = torch.stack(outputs, dim=1)
        if return_aux:
            return predictions, auxiliary
        return predictions

    def extract_representation(
        self,
        clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all frames and return the final temporally processed feature
        after spatial global average pooling.

        Args:
            clip: (B,T,C,H,W)

        Returns:
            representation: (B,D)
        """
        if clip.ndim != 5:
            raise ValueError("clip must have shape (B,T,C,H,W).")
        first_latent = self.encoder(clip[:, 0])
        state = self._initial_state(first_latent)
        feature = None

        for t in range(clip.shape[1]):
            latent = (
                first_latent
                if t == 0
                else self.encoder(clip[:, t])
            )
            feature, state, _ = self.core.step(
                latent,
                state,
                return_aux=False,
            )

        if feature is None:
            raise RuntimeError("Empty video clip.")
        return feature.mean(dim=(-2, -1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

"""Flow-equivariant recurrent neural network baseline for Moving MNIST."""

import random

import torch
from torch import nn

from .transport import translate_branches


class FERNNCell(nn.Module):
    """Shared convolutional recurrent cell over translated hidden-state branches."""

    def __init__(self, input_channels: int, hidden_channels: int,
                 h_kernel_size: int = 3, u_kernel_size: int = 3,
                 v_range: int = 2):
        super().__init__()
        if v_range < 0:
            raise ValueError("v_range must be non-negative.")
        if h_kernel_size % 2 != 1 or u_kernel_size % 2 != 1:
            raise ValueError("Convolution kernel sizes must be odd.")
        self.hidden_channels = hidden_channels
        self.v_list = [(vx, vy) for vx in range(-v_range, v_range + 1)
                       for vy in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                padding=u_kernel_size // 2, padding_mode="circular", bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                padding=h_kernel_size // 2, padding_mode="circular", bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = u_t.shape
        u_conv = self.conv_u(u_t).unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)
        transported = translate_branches(h_prev, self.v_list)
        h_conv = self.conv_h(transported.reshape(batch * self.num_v, self.hidden_channels,
                                                  height, width))
        h_conv = h_conv.reshape(batch, self.num_v, self.hidden_channels, height, width)
        return self.activation(u_conv + h_conv)


class FERNNPredictor(nn.Module):
    """Autoregressive predictor with shared convolutional branch recurrence."""

    def __init__(self, input_channels: int = 1, hidden_channels: int = 128,
                 height: int = 28, width: int = 28, output_channels=None,
                 h_kernel_size: int = 3, u_kernel_size: int = 3,
                 v_range: int = 2, pool_type: str = "max", decoder_conv_layers: int = 3):
        super().__init__()
        if pool_type not in ("max", "mean", "sum"):
            raise ValueError("pool_type must be 'max', 'mean' or 'sum'.")
        if decoder_conv_layers < 0:
            raise ValueError("decoder_conv_layers must be non-negative.")
        self.height = height
        self.width = width
        self.pool_type = pool_type
        self.output_channels = output_channels or input_channels
        self.cell = FERNNCell(input_channels, hidden_channels,
                               h_kernel_size, u_kernel_size, v_range)
        self.hidden_channels = hidden_channels
        self.num_v = self.cell.num_v
        decoder = []
        for _ in range(decoder_conv_layers):
            decoder.extend([
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                          padding_mode="circular", bias=False),
                nn.ReLU(),
            ])
        decoder.append(nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1,
                                 padding_mode="circular", bias=False))
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq: torch.Tensor, pred_len: int,
                teacher_forcing_ratio: float = 0.0, target_seq=None,
                return_hidden: bool = False):
        if input_seq.ndim != 5 or input_seq.size(1) < 1 or pred_len < 1:
            raise ValueError("Expected (B,T,C,H,W) input with T>=1 and pred_len>=1.")
        batch, t_in, _, height, width = input_seq.shape
        state = input_seq.new_zeros(batch, self.num_v, self.hidden_channels, height, width)
        input_states, output_states = [], []

        for t in range(t_in):
            state = self.cell(input_seq[:, t], state)
            if return_hidden:
                input_states.append(state.detach().clone())

        previous = input_seq[:, -1]
        predictions = []
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = previous.detach()
            state = self.cell(frame, state)
            if return_hidden:
                output_states.append(state.detach().clone())

            if self.pool_type == "max":
                features = state.max(dim=1)[0]
            elif self.pool_type == "mean":
                features = state.mean(dim=1)
            else:
                features = state.sum(dim=1)
            previous = self.decoder_conv(features)
            predictions.append(previous)

        output = torch.stack(predictions, dim=1)
        if return_hidden:
            return output, torch.stack(input_states, 1), torch.stack(output_states, 1)
        return output

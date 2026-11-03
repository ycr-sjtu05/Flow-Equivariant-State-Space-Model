from __future__ import annotations

import torch
import torch.nn as nn

from .core import MambaCore


class MambaForecaster(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        core: MambaCore,
        decoder: nn.Module,
        use_last_context_guides: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.decoder = decoder
        self.use_last_context_guides = use_last_context_guides

    def forward(
        self,
        x: torch.Tensor,
        pred_steps: int,
        teacher_forcing_latents: dict[int, torch.Tensor] | None = None,
        state0=None,
        ablation_cfg=None,
        return_probe: bool = False,
    ):
        enc = self.encoder(x)
        latents = enc.latents
        guides = enc.guides

        z_pred_pyramid, aux_list = self.core.rollout_pyramid(
            x0_ctx=latents[0],
            x1_ctx=latents[1],
            x2_ctx=latents[2],
            pred_steps=pred_steps,
            teacher_forcing_pyramid=teacher_forcing_latents,
            state0=state0,
            ablation_cfg=ablation_cfg,
            return_probe=return_probe,
        )

        guide_context = None
        if self.use_last_context_guides and len(guides) > 0:
            guide_context = {res: guides[res][:, -1] for res in guides.keys()}

        y_pred = self.decoder(
            z_pred_pyramid=z_pred_pyramid,
            guide_context=guide_context,
        )
        return y_pred, aux_list

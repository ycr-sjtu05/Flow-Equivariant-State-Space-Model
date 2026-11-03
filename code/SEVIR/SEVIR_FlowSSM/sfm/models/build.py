from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sfm.models.pyramid import PyramidEncoder, PyramidDecoder
from sfm.models.mamba import MambaCore, MambaForecaster
from sfm.models.fernn import FERNNCore, FERNNForecaster
from sfm.models.flow_ssm import FlowSSMCore, FlowSSMForecaster
from sfm.models.hier_flow_ssm import HierPyramidDecoder


def get_arg(cfg: Mapping[str, Any], name: str, default: Any) -> Any:
    value = cfg.get(name)
    return default if value is None else value


def build_model(model_name: str, cfg: Mapping[str, Any]):
    c = int(get_arg(cfg, 'latent_channels', 64))
    latent_channels = (c, c, c)
    no_guides = bool(get_arg(cfg, 'no_guides', False))
    residual_scale = float(get_arg(cfg, 'residual_scale', 0.5))

    encoder = PyramidEncoder(
        in_channels=1,
        latent_channels=latent_channels,
        mid_channels=int(get_arg(cfg, 'encoder_mid_channels', 32)),
        guide_channels=int(get_arg(cfg, 'guide_channels', 8)),
        use_guides=not no_guides,
    )
    decoder = PyramidDecoder(
        latent_channels=latent_channels,
        decode_channels=int(get_arg(cfg, 'decoder_channels', c)),
        guide_channels=int(get_arg(cfg, 'guide_channels', 8)),
        use_guides=not no_guides,
        out_channels=1,
    )

    if model_name == 'mamba':
        core = MambaCore(
            latent_channels=c,
            d_state=int(get_arg(cfg, 'mamba_d_state', 16)),
            d_conv=int(get_arg(cfg, 'mamba_d_conv', 4)),
            expand=int(get_arg(cfg, 'mamba_expand', 2)),
            num_blocks=int(get_arg(cfg, 'mamba_num_blocks', 1)),
            residual_scale=residual_scale,
        )
        return MambaForecaster(encoder, core, decoder, use_last_context_guides=not no_guides)

    if model_name == 'fernn':
        core = FERNNCore(
            latent_channels=c,
            hidden_channels=get_arg(cfg, 'fernn_hidden_channels', c),
            v_axis_size=int(get_arg(cfg, 'fernn_v_axis_size', 3)),
            kernel_size=int(get_arg(cfg, 'fernn_kernel_size', 3)),
            residual_scale=residual_scale,
        )
        return FERNNForecaster(encoder, core, decoder, use_last_context_guides=not no_guides)

    if model_name in ('flow_ssm', 'hier_flow_ssm'):
        core = FlowSSMCore(
            input_channels={0: c, 1: c, 2: c},
            output_channels=c,
            feat_dim=int(get_arg(cfg, 'core_feat_dim', 64)),
            v_axis_size=int(get_arg(cfg, 'plain_v_axis_size', 3)),
            state_dims={
                0: int(get_arg(cfg, 'plain_state_dim0', 64)),
                1: int(get_arg(cfg, 'plain_state_dim1', 80)),
                2: int(get_arg(cfg, 'plain_state_dim2', 96)),
            },
            residual_latent=not bool(get_arg(cfg, 'no_residual_latent', False)),
            residual_scale=residual_scale,
        )
        if model_name == 'hier_flow_ssm':
            decoder = HierPyramidDecoder(
                inner_decoder=decoder,
                latent_channels=latent_channels,
                adapter_channels=int(get_arg(cfg, 'hier_adapter_channels', 16)),
                gamma_max=float(get_arg(cfg, 'hier_gamma_max', 0.10)),
                gamma_init=float(get_arg(cfg, 'hier_gamma_init', -2.0)),
                norm_groups=int(get_arg(cfg, 'hier_norm_groups', 8)),
                zero_init=not bool(get_arg(cfg, 'no_zero_init_adapter', False)),
            )
        return FlowSSMForecaster(encoder, core, decoder, use_last_context_guides=not no_guides)

    raise ValueError(f'Unknown model: {model_name}')


def initialize_hierarchy(model, flow_state: Mapping[str, Any]) -> None:
    state = flow_state.get('model_state', flow_state.get('model', flow_state))
    mapped = {
        'decoder.inner.' + k[len('decoder.'): ] if k.startswith('decoder.') else k: value
        for k, value in state.items()
    }
    result = model.load_state_dict(mapped, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith('decoder.adapter.')]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f'Flow-SSM checkpoint mismatch: missing={missing[:10]}, '
            f'unexpected={result.unexpected_keys[:10]}'
        )


def freeze_except_hierarchy(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.adapter.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in model.decoder.adapter.parameters() if parameter.requires_grad]

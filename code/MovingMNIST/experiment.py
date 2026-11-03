"""Model definitions and architecture configurations."""

from models.fernn import FERNNPredictor
from models.flow_ssm import FlowSSMPredictor


def build_model(config: dict):
    """Instantiate a predictor from architecture settings."""
    name = config["model"]
    common = dict(input_channels=1, output_channels=1,
                  height=config["image_size"], width=config["image_size"],
                  v_range=config["v_range"], pool_type="max",
                  decoder_conv_layers=config["decoder_conv_layers"])
    if name == "flow_ssm":
        return FlowSSMPredictor(D=config["mamba_D"], N=config["mamba_N"], **common)
    if name == "fernn":
        return FERNNPredictor(hidden_channels=config["hidden_size"], **common)
    raise ValueError(f"Unknown model: {name!r}")


def architecture_config(args) -> dict:
    return {key: getattr(args, key) for key in (
        "model", "image_size", "v_range", "decoder_conv_layers",
        "mamba_D", "mamba_N", "hidden_size"
    )}

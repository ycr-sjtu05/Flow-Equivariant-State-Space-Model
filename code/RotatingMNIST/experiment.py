VELOCITIES = (-40, -20, 0, 20, 40)


def model_config(args):
    return {
        "model": args.model,
        "image_size": args.image_size,
        "v_list": list(args.v_list),
        "n_rot": args.n_rot,
        "d_model": args.d_model,
        "hidden_size": args.hidden_size,
        "kernel_size": args.kernel_size,
        "decoder_conv_layers": args.decoder_conv_layers,
    }


def build_model(config):
    from models import RotationFERNN, RotationFlowSSM

    common = {
        "input_channels": 1,
        "height": config["image_size"],
        "width": config["image_size"],
        "v_list": config["v_list"],
        "pool_type": "max",
        "decoder_conv_layers": config["decoder_conv_layers"],
    }
    if config["model"] == "flow_ssm":
        return RotationFlowSSM(
            D=config["d_model"],
            N_rot=config["n_rot"],
            kernel_size=config["kernel_size"],
            **common,
        )
    if config["model"] == "fernn":
        return RotationFERNN(
            hidden_channels=config["hidden_size"],
            N=config["n_rot"],
            h_kernel_size=config["kernel_size"],
            u_kernel_size=config["kernel_size"],
            **common,
        )
    raise ValueError(f"Unknown model: {config['model']}")

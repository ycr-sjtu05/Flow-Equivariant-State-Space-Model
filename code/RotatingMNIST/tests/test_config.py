from experiment import VELOCITIES, model_config
from train import parse_args


def test_paper_defaults():
    args = parse_args([])
    assert args.v_list == list(VELOCITIES) == [-40, -20, 0, 20, 40]
    assert args.data_v_list == list(VELOCITIES)
    assert args.d_model == 32
    assert args.n_rot == 8
    assert args.epochs == 200
    assert args.lr == 1e-4
    assert args.hidden_size == 32
    assert args.decoder_conv_layers == 4
    assert args.batch_size == 16
    assert args.teacher_forcing_ratio == 0
    cfg = model_config(args)
    assert cfg["model"] == "flow_ssm"
    assert cfg["n_rot"] * cfg["d_model"] == 256

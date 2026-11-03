import pytest
import torch

pytest.importorskip("escnn")
from experiment import build_model


@pytest.mark.parametrize("model_name", ["flow_ssm", "fernn"])
def test_forward_and_gradient(model_name):
    torch.manual_seed(123)
    config = dict(
        model=model_name, image_size=16, v_list=[-40, 0, 40],
        n_rot=4, d_model=2, hidden_size=2, kernel_size=3,
        decoder_conv_layers=1,
    )
    model = build_model(config)
    x = torch.rand(1, 2, 1, 16, 16)
    output = model(x, pred_len=2, teacher_forcing_ratio=0.0)
    assert output.shape == (1, 2, 1, 16, 16)
    output.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

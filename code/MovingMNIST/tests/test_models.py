import torch

from models.fernn import FERNNCell, FERNNPredictor
from models.flow_ssm import FlowSSMCell, FlowSSMPredictor
from models.transport import translate_branches


def test_transport_branch_order_and_periodic_boundaries():
    previous = torch.zeros(1, 3, 1, 5, 5)
    previous[:, :, :, 0, 0] = 1
    result = translate_branches(previous, [(-1, 0), (0, 2), (1, -1)])
    assert result[0, 0, 0, 0, 4] == 1
    assert result[0, 1, 0, 2, 0] == 1
    assert result[0, 2, 0, 4, 1] == 1


def test_flow_cell_has_shared_selective_fields():
    torch.manual_seed(3)
    cell = FlowSSMCell(d_model=4, d_state=2, v_range=1)
    feature = torch.rand(1, 4, 8, 8)
    state = torch.zeros(1, 9, 2, 8, 8)
    output, next_state = cell(feature, state)
    assert output.shape == (1, 9, 4, 8, 8)
    assert next_state.shape == state.shape
    # Equal initial memories and branch-shared gates imply identical outputs.
    assert torch.allclose(next_state[:, 0], next_state[:, 1])
    assert torch.allclose(output[:, 0], output[:, 1])


def test_models_output_shapes_and_gradients():
    for model in (
        FlowSSMPredictor(D=4, N=2, v_range=1, decoder_conv_layers=1),
        FERNNPredictor(hidden_channels=4, v_range=1, decoder_conv_layers=1),
    ):
        model.train()
        context = torch.rand(1, 2, 1, 28, 28)
        target = torch.rand(1, 2, 1, 28, 28)
        output = model(context, pred_len=2, target_seq=target)
        assert output.shape == target.shape
        output.square().mean().backward()
        assert any(p.grad is not None and p.grad.isfinite().all()
                   for p in model.parameters())


def test_fernn_return_hidden_signatures():
    model = FERNNPredictor(hidden_channels=4, v_range=0, decoder_conv_layers=0)
    frames = torch.rand(2, 2, 1, 28, 28)
    predicted, encoded_states, decoded_states = model(frames, pred_len=3, return_hidden=True)
    assert predicted.shape == (2, 3, 1, 28, 28)
    assert encoded_states.shape == (2, 2, 1, 4, 28, 28)
    assert decoded_states.shape == (2, 3, 1, 4, 28, 28)


def test_cell_state_dict_parameter_keys():
    flow = FlowSSMCell(4, 2, 1)
    fernn = FERNNCell(1, 4, v_range=1)
    assert "state_mix.weight" in flow.state_dict()
    assert "conv_delta.weight" in flow.state_dict()
    assert "conv_B.weight" in flow.state_dict()
    assert "conv_h.weight" in fernn.state_dict()

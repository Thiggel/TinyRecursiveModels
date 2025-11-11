from typing import Tuple

import pytest

torch = pytest.importorskip("torch")
from torch import nn

pytest.importorskip("transformers")

from models.recursive_reasoning.depth_recurrence import (
    LSTMDepthCell,
    MambaDepthCell,
    RNNDepthCell,
    XLSTMDepthCell,
)


def _copy_rnn_weights(cell: RNNDepthCell, reference: nn.RNN) -> None:
    reference.weight_ih_l0.data.copy_(cell.first_layer.weight_ih)
    reference.weight_hh_l0.data.copy_(cell.first_layer.weight_hh)
    reference.bias_ih_l0.data.copy_(cell.first_layer.bias_ih)
    reference.bias_hh_l0.data.copy_(cell.first_layer.bias_hh)

    for layer_idx, layer in enumerate(cell.additional_layers, start=1):
        getattr(reference, f"weight_ih_l{layer_idx}").data.copy_(layer.weight_ih)
        getattr(reference, f"weight_hh_l{layer_idx}").data.copy_(layer.weight_hh)
        getattr(reference, f"bias_ih_l{layer_idx}").data.copy_(layer.bias_ih)
        getattr(reference, f"bias_hh_l{layer_idx}").data.copy_(layer.bias_hh)


def _copy_lstm_weights(cell: LSTMDepthCell, reference: nn.LSTM) -> None:
    reference.weight_ih_l0.data.copy_(cell.first_layer.weight_ih)
    reference.weight_hh_l0.data.copy_(cell.first_layer.weight_hh)
    reference.bias_ih_l0.data.copy_(cell.first_layer.bias_ih)
    reference.bias_hh_l0.data.copy_(cell.first_layer.bias_hh)

    for layer_idx, layer in enumerate(cell.additional_layers, start=1):
        getattr(reference, f"weight_ih_l{layer_idx}").data.copy_(layer.weight_ih)
        getattr(reference, f"weight_hh_l{layer_idx}").data.copy_(layer.weight_hh)
        getattr(reference, f"bias_ih_l{layer_idx}").data.copy_(layer.bias_ih)
        getattr(reference, f"bias_hh_l{layer_idx}").data.copy_(layer.bias_hh)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_rnn_depth_cell_matches_pytorch_rnn(nonlinearity: str) -> None:
    torch.manual_seed(0)
    hidden_size = 4
    batch, positions, steps = 2, 3, 4
    cell = RNNDepthCell(
        input_size=hidden_size,
        hidden_size=hidden_size,
        nonlinearity=nonlinearity,
        num_layers=2,
    )
    reference = nn.RNN(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=2,
        nonlinearity=nonlinearity,
        batch_first=False,
    )
    _copy_rnn_weights(cell, reference)

    h = torch.randn(batch, positions, hidden_size)
    initial_top = h.clone()
    inputs = torch.randn(steps, batch, positions, hidden_size)

    state = cell.init_state(batch, positions, device=h.device, dtype=h.dtype)
    outputs = []
    for t in range(steps):
        h, state = cell(inputs[t], h, state)
        outputs.append(h)
    stacked = torch.stack(outputs).permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)

    flat_inputs = inputs.reshape(steps, batch * positions, hidden_size)
    initial_hidden = torch.zeros(reference.num_layers, batch * positions, hidden_size)
    initial_hidden[0] = initial_top.reshape(batch * positions, hidden_size)
    ref_outputs, ref_hidden = reference(flat_inputs, initial_hidden)

    torch.testing.assert_close(stacked, ref_outputs.permute(1, 0, 2))

    first_hidden, extra_hidden = state
    torch.testing.assert_close(first_hidden.reshape(batch * positions, hidden_size), ref_hidden[0])
    for idx, layer_state in enumerate(extra_hidden):
        torch.testing.assert_close(layer_state.reshape(batch * positions, hidden_size), ref_hidden[idx + 1])
    torch.testing.assert_close(h.reshape(batch * positions, hidden_size), ref_hidden[-1])


def test_lstm_depth_cell_matches_pytorch_lstm() -> None:
    torch.manual_seed(0)
    hidden_size = 4
    batch, positions, steps = 2, 3, 4
    cell = LSTMDepthCell(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=2,
    )
    reference = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=False,
    )
    _copy_lstm_weights(cell, reference)

    h = torch.randn(batch, positions, hidden_size)
    initial_top = h.clone()
    inputs = torch.randn(steps, batch, positions, hidden_size)

    state = cell.init_state(batch, positions, device=h.device, dtype=h.dtype)
    outputs = []
    for t in range(steps):
        h, state = cell(inputs[t], h, state)
        outputs.append(h)
    stacked = torch.stack(outputs).permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)

    flat_inputs = inputs.reshape(steps, batch * positions, hidden_size)
    initial_hidden = torch.zeros(reference.num_layers, batch * positions, hidden_size)
    initial_hidden[0] = initial_top.reshape(batch * positions, hidden_size)
    initial_cell = torch.zeros_like(initial_hidden)
    ref_outputs, (ref_hidden, ref_cell) = reference(flat_inputs, (initial_hidden, initial_cell))

    torch.testing.assert_close(stacked, ref_outputs.permute(1, 0, 2))

    first_hidden, first_cell, extra = state
    torch.testing.assert_close(first_hidden.reshape(batch * positions, hidden_size), ref_hidden[0])
    torch.testing.assert_close(first_cell.reshape(batch * positions, hidden_size), ref_cell[0])
    for idx, (layer_hidden, layer_cell) in enumerate(extra):
        torch.testing.assert_close(layer_hidden.reshape(batch * positions, hidden_size), ref_hidden[idx + 1])
        torch.testing.assert_close(layer_cell.reshape(batch * positions, hidden_size), ref_cell[idx + 1])
    torch.testing.assert_close(h.reshape(batch * positions, hidden_size), ref_hidden[-1])


def _sequential_outputs(
    cell: torch.nn.Module,
    u_sequence: torch.Tensor,
    initial_hidden: torch.Tensor,
    batch: int,
    positions: int,
) -> Tuple[torch.Tensor, Tuple]:
    state = cell.init_state(batch, positions, device=u_sequence.device, dtype=u_sequence.dtype)
    h = initial_hidden
    outputs = []
    for step in range(u_sequence.shape[0]):
        h, state = cell(u_sequence[step], h, state)
        outputs.append(h)
    stacked = torch.stack(outputs) if outputs else torch.empty(0)
    return stacked, state


@pytest.mark.parametrize(
    "cell_cls, kwargs",
    [
        (
            XLSTMDepthCell,
            dict(
                num_layers=2,
                chunkwise_kernel="chunkwise--native_autograd",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
                num_heads=1,
            ),
        ),
        (MambaDepthCell, dict(state_size=8, expand=1, conv_kernel=2, num_layers=2, implementation="pytorch")),
    ],
)
def test_transformer_backed_cells_match_full_sequence(cell_cls, kwargs) -> None:
    torch.manual_seed(0)
    hidden_size = 6
    batch, positions, steps = 2, 2, 3
    inputs = torch.randn(steps, batch, positions, hidden_size)
    initial_hidden = torch.zeros(batch, positions, hidden_size)

    cell = cell_cls(hidden_size, **kwargs)
    if hasattr(cell, "model"):
        cell.model.eval()
    sequential, state = _sequential_outputs(cell, inputs, initial_hidden, batch, positions)
    sequential = sequential.permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)

    model = cell.model
    flat_inputs = inputs.permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)
    reference = model(inputs_embeds=flat_inputs, use_cache=False).last_hidden_state

    torch.testing.assert_close(sequential, reference)

    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], torch.Tensor):
        cache_position = state[1]
        expected = torch.full_like(cache_position, steps)
        torch.testing.assert_close(cache_position, expected)
    elif hasattr(state, "seqlen_offset"):
        assert state.seqlen_offset == steps

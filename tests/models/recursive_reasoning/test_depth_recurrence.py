from typing import Tuple

import pytest

torch = pytest.importorskip("torch")
from torch import nn

pytest.importorskip("transformers")

from models.recursive_reasoning.depth_recurrence import LSTMDepthCell, MambaDepthCell, RNNDepthCell, XLSTMDepthCell


def _copy_rnn_weights(cell: RNNDepthCell, reference: nn.RNN) -> None:
    first = cell.cells[0]
    reference.weight_ih_l0.data.copy_(first.weight_ih)
    reference.weight_hh_l0.data.copy_(first.weight_hh)
    reference.bias_ih_l0.data.copy_(first.bias_ih)
    reference.bias_hh_l0.data.copy_(first.bias_hh)


def _copy_lstm_weights(cell: LSTMDepthCell, reference: nn.LSTM) -> None:
    first = cell.cells[0]
    reference.weight_ih_l0.data.copy_(first.weight_ih)
    reference.weight_hh_l0.data.copy_(first.weight_hh)
    reference.bias_ih_l0.data.copy_(first.bias_ih)
    reference.bias_hh_l0.data.copy_(first.bias_hh)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_rnn_depth_cell_matches_pytorch_rnn(nonlinearity: str) -> None:
    torch.manual_seed(0)
    hidden_size = 4
    batch, positions, steps = 2, 3, 4
    cell = RNNDepthCell(
        input_size=hidden_size,
        hidden_size=hidden_size,
        nonlinearity=nonlinearity,
        num_layers=1,
    )
    reference = nn.RNN(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity=nonlinearity,
        batch_first=False,
    )
    _copy_rnn_weights(cell, reference)

    h = torch.randn(batch, positions, hidden_size)
    initial_top = h.clone()
    inputs = torch.randn(steps, batch, positions, hidden_size)

    state = cell.init_state(batch, positions, device=h.device, dtype=h.dtype, initial_hidden=h)
    stacked_layers = getattr(cell, "stacked_layers", cell.num_layers)
    outputs = []
    for t in range(steps):
        next_states = []
        layer_input = h
        for layer_idx in range(stacked_layers):
            layer_output, layer_state = cell.forward_layer(layer_idx, inputs[t], layer_input, state[layer_idx])
            next_states.append(layer_state)
            layer_input = layer_output
        h = layer_input
        state = next_states
        outputs.append(h)
    stacked = torch.stack(outputs).permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)
    assert stacked.dtype == inputs.dtype

    flat_inputs = inputs.reshape(steps, batch * positions, hidden_size)
    initial_hidden = torch.zeros(reference.num_layers, batch * positions, hidden_size)
    initial_hidden[0] = initial_top.reshape(batch * positions, hidden_size)
    ref_outputs, ref_hidden = reference(flat_inputs, initial_hidden)

    torch.testing.assert_close(stacked, ref_outputs.permute(1, 0, 2))

    torch.testing.assert_close(state[0].reshape(batch * positions, hidden_size), ref_hidden[0])
    torch.testing.assert_close(h.reshape(batch * positions, hidden_size), ref_hidden[-1])


def test_lstm_depth_cell_matches_pytorch_lstm() -> None:
    torch.manual_seed(0)
    hidden_size = 4
    batch, positions, steps = 2, 3, 4
    cell = LSTMDepthCell(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
    )
    reference = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=False,
    )
    _copy_lstm_weights(cell, reference)

    h = torch.randn(batch, positions, hidden_size)
    initial_top = h.clone()
    inputs = torch.randn(steps, batch, positions, hidden_size)

    state = cell.init_state(batch, positions, device=h.device, dtype=h.dtype, initial_hidden=h)
    stacked_layers = getattr(cell, "stacked_layers", cell.num_layers)
    outputs = []
    for t in range(steps):
        next_states = []
        layer_input = h
        for layer_idx in range(stacked_layers):
            layer_output, layer_state = cell.forward_layer(layer_idx, inputs[t], layer_input, state[layer_idx])
            next_states.append(layer_state)
            layer_input = layer_output
        h = layer_input
        state = next_states
        outputs.append(h)
    stacked = torch.stack(outputs).permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)
    assert stacked.dtype == inputs.dtype

    flat_inputs = inputs.reshape(steps, batch * positions, hidden_size)
    initial_hidden = torch.zeros(reference.num_layers, batch * positions, hidden_size)
    initial_hidden[0] = initial_top.reshape(batch * positions, hidden_size)
    initial_cell = torch.zeros_like(initial_hidden)
    ref_outputs, (ref_hidden, ref_cell) = reference(flat_inputs, (initial_hidden, initial_cell))

    torch.testing.assert_close(stacked, ref_outputs.permute(1, 0, 2))

    first_hidden, first_cell = state[0]
    torch.testing.assert_close(first_hidden.reshape(batch * positions, hidden_size), ref_hidden[0])
    torch.testing.assert_close(first_cell.reshape(batch * positions, hidden_size), ref_cell[0])
    torch.testing.assert_close(h.reshape(batch * positions, hidden_size), ref_hidden[-1])


def _sequential_outputs(
    cell: torch.nn.Module,
    u_sequence: torch.Tensor,
    initial_hidden: torch.Tensor,
    batch: int,
    positions: int,
) -> Tuple[torch.Tensor, Tuple]:
    state = cell.init_state(batch, positions, device=u_sequence.device, dtype=u_sequence.dtype, initial_hidden=initial_hidden)
    h = initial_hidden
    stacked_layers = getattr(cell, "stacked_layers", cell.num_layers)
    outputs = []
    for step in range(u_sequence.shape[0]):
        next_states = []
        layer_input = h
        for layer_idx in range(stacked_layers):
            layer_output, layer_state = cell.forward_layer(layer_idx, u_sequence[step], layer_input, state[layer_idx])
            next_states.append(layer_state)
            layer_input = layer_output
        h = layer_input
        state = next_states
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
    assert sequential.dtype == inputs.dtype

    model = cell.model
    flat_inputs = inputs.permute(1, 2, 0, 3).reshape(batch * positions, steps, hidden_size)
    reference = model(inputs_embeds=flat_inputs, use_cache=False).last_hidden_state

    torch.testing.assert_close(sequential, reference)

    if isinstance(state, (list, tuple)):
        for entry in state:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], torch.Tensor):
                cache_position = entry[1]
                expected = torch.full_like(cache_position, steps)
                torch.testing.assert_close(cache_position, expected)
            elif hasattr(entry, "seqlen_offset"):
                assert entry.seqlen_offset == steps

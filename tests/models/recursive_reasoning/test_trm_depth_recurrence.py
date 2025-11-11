import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("transformers")

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)


def build_config(**overrides):
    base = dict(
        batch_size=2,
        seq_len=4,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=32,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=2,
        hidden_size=8,
        expansion=2.0,
        num_heads=2,
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
    )
    base.update(overrides)
    return TinyRecursiveReasoningModel_ACTV1Config(**base)


@pytest.mark.parametrize(
    "recurrence, extra",
    [
        ("transformer", {}),
        ("rnn", {"depth_cell_layers": 2}),
        ("lstm", {"depth_cell_layers": 2}),
        (
            "xlstm",
            {
                "depth_cell_layers": 2,
                "depth_cell_xlstm_num_heads": 2,
            },
        ),
        (
            "mamba",
            {
                "depth_cell_layers": 2,
                "depth_cell_state_size": 8,
                "depth_cell_expand": 1,
                "depth_cell_conv_kernel": 2,
                "depth_cell_mamba_impl": "pytorch",
            },
        ),
    ],
)
def test_reasoning_module_depth_modes(recurrence, extra, monkeypatch):
    torch.manual_seed(0)
    kwargs = dict(depth_recurrence=recurrence, depth_recurrence_steps=3)
    kwargs.update(extra)
    config = build_config(**kwargs)

    module = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config)
    hidden = torch.randn(config.batch_size, config.seq_len, config.hidden_size)
    injection = torch.randn_like(hidden)

    if module.mode == "recurrent":
        call_counter = {"calls": 0}
        real_forward = module.depth_block.cell.forward

        def counted(u, h, state):
            call_counter["calls"] += 1
            return real_forward(u, h, state)

        monkeypatch.setattr(module.depth_block.cell, "forward", counted)
        output = module(hidden, injection, cos_sin=None)
        assert call_counter["calls"] == config.depth_recurrence_steps
    else:
        output = module(hidden, injection, cos_sin=None)

    assert output.shape == hidden.shape


def test_depth_steps_default_matches_transformer_layers():
    torch.manual_seed(0)
    config = build_config(depth_recurrence="rnn", depth_cell_layers=1, depth_recurrence_steps=None)
    module = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config)
    assert module.depth_block.config.depth_steps == config.L_layers

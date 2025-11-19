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
        ("rnn", {}),
        ("lstm", {}),
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
def test_reasoning_module_depth_modes(recurrence, extra):
    torch.manual_seed(0)
    kwargs = dict(depth_recurrence=recurrence)
    kwargs.update(extra)
    config = build_config(**kwargs)

    module = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config)
    hidden = torch.randn(config.batch_size, config.seq_len, config.hidden_size)
    injection = torch.randn_like(hidden)

    recurrence_state = None
    output, recurrence_state = module(
        hidden,
        injection,
        recurrence_state=recurrence_state,
        cos_sin=None,
    )

    assert output.shape == hidden.shape
    if module.mode == "transformer":
        assert recurrence_state is None
    else:
        assert recurrence_state is not None


@pytest.mark.parametrize("recurrence", ["rnn", "lstm"])
def test_depth_recurrent_layers_default_to_transformer_block_depth(recurrence):
    torch.manual_seed(0)
    config = build_config(depth_recurrence=recurrence)
    module = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config)
    assert module.depth_block is not None
    assert module.depth_block.stacked_layers == config.L_layers

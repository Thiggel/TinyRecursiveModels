from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from transformers import MambaConfig, MambaModel, xLSTMConfig, xLSTMModel
from transformers.models.mamba.modeling_mamba import MambaCache
from transformers.models.xlstm.modeling_xlstm import xLSTMCache

from models.layers import Attention, rms_norm


@dataclass
class DepthRecurrentConfig:
    hidden_size: int
    num_heads: int
    expansion: float
    rms_norm_eps: float
    depth_steps: int
    cell_type: str
    cell_hidden_size: Optional[int] = None
    cell_state_size: int = 64
    cell_expand: int = 1
    cell_conv_kernel: int = 4
    cell_layers: int = 1
    cell_nonlinearity: str = "tanh"
    xlstm_chunkwise_kernel: str = "chunkwise--native_autograd"
    xlstm_sequence_kernel: str = "native_sequence__native"
    xlstm_step_kernel: str = "native"
    xlstm_num_heads: Optional[int] = None
    mamba_impl: str = "mamba2"


class DepthRecurrentCell(nn.Module):
    """Base interface for depth recurrent cells."""

    num_layers: int
    stacked_layers: int

    def init_state(
        self,
        batch: int,
        positions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def forward_layer(self, layer_idx: int, u: torch.Tensor, h: torch.Tensor, state):
        """Update a specific stacked recurrent layer using the attention-driven input."""

        raise NotImplementedError


class RNNDepthCell(DepthRecurrentCell):
    """Vanilla Elman RNN update driven by the attention output."""

    def __init__(
        self, input_size: int, hidden_size: int, *, nonlinearity: str = "tanh", num_layers: int = 1
    ) -> None:
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")

        self.hidden_size = hidden_size
        self.num_layers = max(1, num_layers)
        self.stacked_layers = self.num_layers
        self.stacked_layers = self.num_layers
        self.cells = nn.ModuleList(
            [nn.RNNCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size, nonlinearity=nonlinearity) for i in range(self.num_layers)]
        )

    def init_state(
        self,
        batch: int,
        positions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        if initial_hidden is None:
            return [None for _ in range(self.num_layers)]
        states = []
        for _ in range(self.num_layers):
            states.append(initial_hidden.to(device=device, dtype=dtype))
        return states

    def forward_layer(self, layer_idx: int, u: torch.Tensor, h: torch.Tensor, state: Optional[torch.Tensor]):
        batch, positions, _ = u.shape
        cell = self.cells[layer_idx]
        orig_dtype = u.dtype
        compute_dtype = cell.weight_ih.dtype
        flat_u = u.reshape(batch * positions, -1).to(compute_dtype)

        if state is None:
            flat_h_prev = h.reshape(batch * positions, -1).to(compute_dtype)
        else:
            flat_h_prev = state.reshape(batch * positions, -1).to(compute_dtype)

        updated_flat = cell(flat_u, flat_h_prev)
        next_hidden = updated_flat.view(batch, positions, self.hidden_size)

        return next_hidden.to(orig_dtype), next_hidden.to(compute_dtype)


class LSTMDepthCell(DepthRecurrentCell):
    """Standard LSTM update that keeps the hidden sequence explicit."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = max(1, num_layers)
        self.cells = nn.ModuleList(
            [nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)]
        )

    def init_state(
        self,
        batch: int,
        positions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        states = []
        for idx in range(self.num_layers):
            cell = self.cells[idx]
            compute_dtype = cell.weight_ih.dtype
            zeros = torch.zeros(batch, positions, self.hidden_size, device=device, dtype=compute_dtype)
            if initial_hidden is not None:
                h_state = initial_hidden.to(device=device, dtype=dtype)
            else:
                h_state = None
            states.append((h_state, zeros))
        return states

    def forward_layer(
        self,
        layer_idx: int,
        u: torch.Tensor,
        h: torch.Tensor,
        state: Tuple[Optional[torch.Tensor], torch.Tensor],
    ):
        h_prev, c_prev = state
        batch, positions, _ = u.shape

        cell = self.cells[layer_idx]
        orig_dtype = u.dtype
        compute_dtype = cell.weight_ih.dtype
        flat_u = u.reshape(batch * positions, -1).to(compute_dtype)
        if h_prev is None:
            flat_h_prev = h.reshape(batch * positions, -1).to(compute_dtype)
        else:
            flat_h_prev = h_prev.reshape(batch * positions, -1).to(compute_dtype)
        flat_c_prev = c_prev.reshape(batch * positions, -1).to(compute_dtype)
        flat_h_next, flat_c_next = cell(flat_u, (flat_h_prev, flat_c_prev))

        next_hidden = flat_h_next.view(batch, positions, self.hidden_size)
        next_cell = flat_c_next.view(batch, positions, self.hidden_size)

        return next_hidden.to(orig_dtype), (
            next_hidden.to(compute_dtype),
            next_cell.to(compute_dtype),
        )


class XLSTMDepthCell(DepthRecurrentCell):
    """xLSTM update backed by the official package or Transformers integration."""

    def __init__(
        self,
        input_size: int,
        *,
        num_layers: int,
        chunkwise_kernel: str,
        sequence_kernel: str,
        step_kernel: str,
        num_heads: Optional[int],
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("XLSTMDepthCell expects at least one layer")

        heads = num_heads if num_heads is not None else 1
        self.num_layers = num_layers
        self.stacked_layers = 1
        self.cache_class = xLSTMCache

        config = xLSTMConfig(
            hidden_size=input_size,
            num_hidden_layers=num_layers,
            num_heads=heads,
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
            mode="inference",
            return_last_states=True,
        )
        self.model = xLSTMModel(config)

    def init_state(
        self,
        batch: int,
        positions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        max_batch = batch * positions
        compute_dtype = self.model.dtype
        cache = self.cache_class(
            config=self.model.config,
            max_batch_size=max_batch,
            dtype=compute_dtype,
            device=device,
        )
        cache.reset()
        cache.seqlen_offset = 0
        return [cache]

    def forward_layer(self, layer_idx: int, u: torch.Tensor, h: torch.Tensor, state):
        if layer_idx != 0:
            raise IndexError("XLSTMDepthCell expects a single stacked layer")
        model = self.model
        cache = state
        batch, positions, hidden = u.shape
        orig_dtype = u.dtype
        compute_dtype = model.dtype
        inputs_embeds = u.view(batch * positions, 1, hidden).to(compute_dtype)
        outputs = model(
            inputs_embeds=inputs_embeds,
            cache_params=cache,
            use_cache=True,
        )
        updated_cache = outputs.cache_params
        next_hidden = outputs.last_hidden_state.view(batch, positions, hidden)
        return next_hidden.to(orig_dtype), updated_cache


class MambaDepthCell(DepthRecurrentCell):
    """Selective state-space update that treats depth as time."""

    def __init__(
        self,
        input_size: int,
        state_size: int,
        expand: int,
        conv_kernel: int,
        num_layers: int,
        implementation: str,
    ) -> None:
        super().__init__()
        if state_size <= 0 or expand <= 0:
            raise ValueError("MambaDepthCell expects positive state size and expansion")
        if num_layers < 1:
            raise ValueError("MambaDepthCell expects at least one layer")

        self.num_layers = num_layers
        self.stacked_layers = 1
        self.cache_class = MambaCache
        config = MambaConfig(
            hidden_size=input_size,
            state_size=state_size,
            num_hidden_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            implementation=implementation,
        )
        self.model = MambaModel(config)

    def init_state(
        self,
        batch: int,
        positions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        max_batch = batch * positions
        compute_dtype = self.model.dtype
        cache = self.cache_class(
            config=self.model.config,
            max_batch_size=max_batch,
            dtype=compute_dtype,
            device=device,
        )
        cache_position = torch.zeros(max_batch, dtype=torch.long, device=device)
        return [(cache, cache_position)]

    def forward_layer(self, layer_idx: int, u: torch.Tensor, h: torch.Tensor, state):
        if layer_idx != 0:
            raise IndexError("MambaDepthCell expects a single stacked layer")
        cache, cache_position = state
        model = self.model
        batch, positions, hidden = u.shape
        orig_dtype = u.dtype
        compute_dtype = model.dtype
        inputs_embeds = u.view(batch * positions, 1, hidden).to(compute_dtype)
        outputs = model(
            inputs_embeds=inputs_embeds,
            cache_params=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        updated_cache = outputs.cache_params
        next_hidden = outputs.last_hidden_state.view(batch, positions, hidden)
        return next_hidden.to(orig_dtype), (updated_cache, cache_position + 1)


class DepthRecurrentBlock(nn.Module):
    def __init__(self, config: DepthRecurrentConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_type = config.cell_type.lower()
        self.cell_layers = max(1, config.cell_layers)

        cell_hidden_size = config.cell_hidden_size or config.hidden_size
        cell_type = config.cell_type.lower()

        if cell_hidden_size != config.hidden_size and cell_type in {"rnn", "lstm", "xlstm"}:
            raise ValueError(
                "When manipulating the TRM hidden state directly the cell hidden size must match the model hidden size."
            )

        if cell_type == "rnn":
            self.cell: DepthRecurrentCell = RNNDepthCell(
                config.hidden_size,
                cell_hidden_size,
                nonlinearity=config.cell_nonlinearity,
                num_layers=self.cell_layers,
            )
        elif cell_type == "lstm":
            self.cell = LSTMDepthCell(
                config.hidden_size,
                cell_hidden_size,
                num_layers=self.cell_layers,
            )
        elif cell_type == "xlstm":
            self.cell = XLSTMDepthCell(
                config.hidden_size,
                num_layers=self.cell_layers,
                chunkwise_kernel=config.xlstm_chunkwise_kernel,
                sequence_kernel=config.xlstm_sequence_kernel,
                step_kernel=config.xlstm_step_kernel,
                num_heads=config.xlstm_num_heads if config.xlstm_num_heads is not None else config.num_heads,
            )
        elif cell_type == "mamba":
            state_size = max(1, config.cell_state_size)
            self.cell = MambaDepthCell(
                config.hidden_size,
                state_size,
                expand=max(1, config.cell_expand),
                conv_kernel=max(1, config.cell_conv_kernel),
                num_layers=self.cell_layers,
                implementation=config.mamba_impl,
            )
        else:
            raise ValueError(f"Unsupported depth recurrence cell: {config.cell_type}")

        self.stacked_layers = (
            self.cell.stacked_layers if hasattr(self.cell, "stacked_layers") else 1
        )
        use_layered_attention = self.cell_type in {"rnn", "lstm"}
        if use_layered_attention:
            self.attn_layers = nn.ModuleList(
                [
                    Attention(
                        hidden_size=config.hidden_size,
                        head_dim=config.hidden_size // config.num_heads,
                        num_heads=config.num_heads,
                        num_key_value_heads=config.num_heads,
                        causal=False,
                    )
                    for _ in range(self.stacked_layers)
                ]
            )
        else:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.norm_eps = config.rms_norm_eps

    def init_state(self, hidden_states: torch.Tensor):
        batch, positions, _ = hidden_states.shape
        return self.cell.init_state(
            batch,
            positions,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            initial_hidden=hidden_states,
        )

    def forward(self, hidden_states: torch.Tensor, *, state=None, cos_sin=None):
        if state is None:
            state = self.init_state(hidden_states)

        next_states = []
        layer_input = hidden_states
        for layer_idx in range(self.stacked_layers):
            attn_input = rms_norm(layer_input, variance_epsilon=self.norm_eps)
            if self.cell_type in {"rnn", "lstm"}:
                u = self.attn_layers[layer_idx](cos_sin=cos_sin, hidden_states=attn_input)
            else:
                u = self.attn(cos_sin=cos_sin, hidden_states=attn_input)
            if isinstance(state, (list, tuple)):
                layer_state = state[layer_idx] if layer_idx < len(state) else None
            else:
                layer_state = state
            layer_output, new_layer_state = self.cell.forward_layer(layer_idx, u, layer_input, layer_state)
            next_states.append(new_layer_state)
            layer_input = layer_output

        return layer_input, next_states

    def step(self, hidden_states: torch.Tensor, state, *, cos_sin=None):
        return self.forward(hidden_states, state=state, cos_sin=cos_sin)

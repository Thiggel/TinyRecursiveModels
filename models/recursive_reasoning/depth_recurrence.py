from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

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

    def init_state(self, batch: int, positions: int, *, device: torch.device, dtype: torch.dtype):
        raise NotImplementedError

    def forward(self, u: torch.Tensor, h: torch.Tensor, state):
        """Update the hidden state given the attention-driven input.

        Args:
            u: The attention output at the current depth step, shape ``[B, N, D_u]``.
            h: The current hidden state sequence, shape ``[B, N, D_h]``.
            state: Additional recurrent state carried by the cell.

        Returns:
            A tuple ``(new_h, new_state)`` with ``new_h`` shaped ``[B, N, D_h]``.
        """

        raise NotImplementedError


class RNNDepthCell(DepthRecurrentCell):
    """Vanilla Elman RNN update driven by the attention output."""

    def __init__(self, input_size: int, hidden_size: int, *, nonlinearity: str = "tanh", num_layers: int = 1) -> None:
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")
        if num_layers < 1:
            raise ValueError("RNNDepthCell expects at least one layer")

        self.first_layer = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
        self.additional_layers = nn.ModuleList(
            nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
            for _ in range(num_layers - 1)
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_state(self, batch: int, positions: int, *, device: torch.device, dtype: torch.dtype):
        if self.num_layers == 1:
            return None, tuple()
        return None, tuple(
            None for _ in range(len(self.additional_layers))
        )

    def forward(self, u: torch.Tensor, h: torch.Tensor, state: Tuple[Optional[torch.Tensor], Sequence[Optional[torch.Tensor]]]):
        first_prev, extra_prev = state
        batch, positions, _ = u.shape
        orig_dtype = u.dtype
        compute_dtype = self.first_layer.weight_ih.dtype
        flat_u = u.reshape(batch * positions, -1).to(compute_dtype)

        if first_prev is None:
            flat_h_prev = h.reshape(batch * positions, -1).to(compute_dtype)
        else:
            flat_h_prev = first_prev.reshape(batch * positions, -1).to(compute_dtype)

        updated_flat = self.first_layer(flat_u, flat_h_prev)
        next_first = updated_flat.view(batch, positions, self.hidden_size)

        layer_inputs = updated_flat
        new_states = []
        for idx, layer in enumerate(self.additional_layers):
            prev_state = extra_prev[idx] if idx < len(extra_prev) else None
            if prev_state is None:
                prev_flat = torch.zeros(
                    batch * positions,
                    self.hidden_size,
                    device=layer_inputs.device,
                    dtype=compute_dtype,
                )
            else:
                prev_flat = prev_state.reshape(batch * positions, -1).to(compute_dtype)
            layer_outputs = layer(layer_inputs, prev_flat)
            new_states.append(layer_outputs.view(batch, positions, self.hidden_size))
            layer_inputs = layer_outputs

        next_hidden = layer_inputs.view(batch, positions, self.hidden_size)
        return next_hidden.to(orig_dtype), (
            next_first.to(compute_dtype),
            tuple(state_tensor.to(compute_dtype) for state_tensor in new_states),
        )


class LSTMDepthCell(DepthRecurrentCell):
    """Standard LSTM update that keeps the hidden sequence explicit."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("LSTMDepthCell expects at least one layer")

        self.first_layer = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.additional_layers = nn.ModuleList(
            nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size) for _ in range(num_layers - 1)
        )
        self.hidden_size = hidden_size

    def init_state(self, batch: int, positions: int, *, device: torch.device, dtype: torch.dtype):
        compute_dtype = self.first_layer.weight_ih.dtype
        zeros = torch.zeros(batch, positions, self.hidden_size, device=device, dtype=compute_dtype)
        additional = tuple(
            (None, None) for _ in range(len(self.additional_layers))
        )
        return None, zeros, additional

    def forward(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        state: Tuple[Optional[torch.Tensor], torch.Tensor, Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], ...]],
    ):
        h_prev, c_prev, extra_prev = state
        batch, positions, _ = u.shape

        orig_dtype = u.dtype
        compute_dtype = self.first_layer.weight_ih.dtype
        flat_u = u.reshape(batch * positions, -1).to(compute_dtype)
        if h_prev is None:
            flat_h_prev = h.reshape(batch * positions, -1).to(compute_dtype)
        else:
            flat_h_prev = h_prev.reshape(batch * positions, -1).to(compute_dtype)
        flat_c_prev = c_prev.reshape(batch * positions, -1).to(compute_dtype)
        flat_h_next, flat_c_next = self.first_layer(flat_u, (flat_h_prev, flat_c_prev))

        next_hidden = flat_h_next.view(batch, positions, self.hidden_size)
        next_cell = flat_c_next.view(batch, positions, self.hidden_size)

        new_extra: list[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_input = flat_h_next
        for idx, layer in enumerate(self.additional_layers):
            if idx < len(extra_prev):
                prev_h, prev_c = extra_prev[idx]
            else:
                prev_h = prev_c = None

            if prev_h is None:
                prev_h_flat = torch.zeros(
                    batch * positions,
                    self.hidden_size,
                    device=layer_input.device,
                    dtype=compute_dtype,
                )
            else:
                prev_h_flat = prev_h.reshape(batch * positions, -1).to(compute_dtype)
            if prev_c is None:
                prev_c_flat = torch.zeros(
                    batch * positions,
                    self.hidden_size,
                    device=layer_input.device,
                    dtype=compute_dtype,
                )
            else:
                prev_c_flat = prev_c.reshape(batch * positions, -1).to(compute_dtype)

            layer_h, layer_c = layer(layer_input, (prev_h_flat, prev_c_flat))
            new_extra.append(
                (
                    layer_h.view(batch, positions, self.hidden_size),
                    layer_c.view(batch, positions, self.hidden_size),
                )
            )
            layer_input = layer_h

        next_top = layer_input.view(batch, positions, self.hidden_size)
        return next_top.to(orig_dtype), (
            next_hidden.to(compute_dtype),
            next_cell.to(compute_dtype),
            tuple((h_state.to(compute_dtype), c_state.to(compute_dtype)) for h_state, c_state in new_extra),
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

        self.config = xLSTMConfig(
            hidden_size=input_size,
            num_hidden_layers=num_layers,
            num_heads=heads,
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
            mode="inference",
            return_last_states=True,
        )
        self.model = xLSTMModel(self.config)
        self.cache_class = xLSTMCache

    def init_state(self, batch: int, positions: int, *, device: torch.device, dtype: torch.dtype):
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
        return cache

    def forward(self, u: torch.Tensor, h: torch.Tensor, state):
        cache = state
        batch, positions, hidden = u.shape
        orig_dtype = u.dtype
        compute_dtype = self.model.dtype
        inputs_embeds = u.view(batch * positions, 1, hidden).to(compute_dtype)
        outputs = self.model(
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

        self.config = MambaConfig(
            hidden_size=input_size,
            state_size=state_size,
            num_hidden_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            implementation=implementation,
        )
        self.model = MambaModel(self.config)
        self.cache_class = MambaCache

    def init_state(self, batch: int, positions: int, *, device: torch.device, dtype: torch.dtype):
        max_batch = batch * positions
        compute_dtype = self.model.dtype
        cache = self.cache_class(
            config=self.model.config,
            max_batch_size=max_batch,
            dtype=compute_dtype,
            device=device,
        )
        cache_position = torch.zeros(max_batch, dtype=torch.long, device=device)
        return cache, cache_position

    def forward(self, u: torch.Tensor, h: torch.Tensor, state):
        cache, cache_position = state
        batch, positions, hidden = u.shape
        orig_dtype = u.dtype
        compute_dtype = self.model.dtype
        inputs_embeds = u.view(batch * positions, 1, hidden).to(compute_dtype)
        outputs = self.model(
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

        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )

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
                num_layers=max(1, config.cell_layers),
            )
        elif cell_type == "lstm":
            self.cell = LSTMDepthCell(
                config.hidden_size,
                cell_hidden_size,
                num_layers=max(1, config.cell_layers),
            )
        elif cell_type == "xlstm":
            self.cell = XLSTMDepthCell(
                config.hidden_size,
                num_layers=max(1, config.cell_layers),
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
                num_layers=max(1, config.cell_layers),
                implementation=config.mamba_impl,
            )
        else:
            raise ValueError(f"Unsupported depth recurrence cell: {config.cell_type}")

        self.norm_eps = config.rms_norm_eps

    def init_state(self, hidden_states: torch.Tensor):
        batch, positions, _ = hidden_states.shape
        return self.cell.init_state(batch, positions, device=hidden_states.device, dtype=hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor, *, state=None, cos_sin=None):
        if state is None:
            state = self.init_state(hidden_states)

        attn_input = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        u = self.attn(cos_sin=cos_sin, hidden_states=attn_input)
        new_hidden, new_state = self.cell(u, hidden_states, state)
        return new_hidden, new_state

    def step(self, hidden_states: torch.Tensor, state, *, cos_sin=None):
        return self.forward(hidden_states, state=state, cos_sin=cos_sin)

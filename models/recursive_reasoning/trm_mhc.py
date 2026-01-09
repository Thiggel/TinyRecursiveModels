from typing import Tuple, Dict
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

try:  # optional Triton path
    from models.recursive_reasoning import mhc_triton
except Exception:  # pragma: no cover
    mhc_triton = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return (x.to(input_dtype) * self.weight.to(input_dtype))


def sinkhorn_log_doubly_stochastic(logits: torch.Tensor, n_iters: int = 20, tau: float = 0.05) -> torch.Tensor:
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(n_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class MHCAttentionSubLayer(nn.Module):
    def __init__(self, attention: Attention) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, x: torch.Tensor, *, cos_sin: CosSin | None = None) -> torch.Tensor:
        return self.attention(cos_sin=cos_sin, hidden_states=x)


class MHCFFNSubLayer(nn.Module):
    def __init__(self, mlp: SwiGLU) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ManifoldHyperConnectionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_streams: int,
        sublayer: nn.Module,
        *,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        sinkhorn_tau: float = 0.05,
        alpha_init: float = 1e-2,
        use_triton: bool = False,
        triton_backward: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.sublayer = sublayer
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.sinkhorn_tau = sinkhorn_tau
        self.use_triton = use_triton and (mhc_triton is not None)
        self.triton_backward = triton_backward

        d = hidden_dim * n_streams
        self.rms = RMSNorm(d)

        self.fc_pre = nn.Linear(d, n_streams, bias=False)
        self.fc_post = nn.Linear(d, n_streams, bias=False)
        self.fc_res = nn.Linear(d, n_streams * n_streams, bias=False)

        self.bias_pre = nn.Parameter(torch.zeros(n_streams))
        self.bias_post = nn.Parameter(torch.zeros(n_streams))
        self.bias_res = nn.Parameter(torch.zeros(n_streams * n_streams))

        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

        with torch.no_grad():
            self.alpha_pre.fill_(alpha_init)
            self.alpha_post.fill_(alpha_init)
            self.alpha_res.fill_(alpha_init)

    def forward(self, x_streams: torch.Tensor, **sublayer_kwargs) -> torch.Tensor:
        B, S, n, C = x_streams.shape
        assert n == self.n_streams and C == self.hidden_dim
        d = n * C

        x_flat = x_streams.reshape(B * S, d).to(torch.float32)
        x_norm = self.rms(x_flat)

        H_pre_tilde = self.alpha_pre * self.fc_pre(x_norm) + self.bias_pre
        H_post_tilde = self.alpha_post * self.fc_post(x_norm) + self.bias_post
        H_res_tilde = self.alpha_res * self.fc_res(x_norm) + self.bias_res

        H_pre = torch.sigmoid(H_pre_tilde)
        H_post = 2.0 * torch.sigmoid(H_post_tilde)

        H_res = H_res_tilde.view(B * S, n, n)
        if self.use_triton and x_streams.is_cuda:
            H_res = mhc_triton.sinkhorn_log_triton_autograd(
                H_res,
                n_iters=self.sinkhorn_iters,
                tau=self.sinkhorn_tau,
                use_triton_backward=self.triton_backward,
            )
        else:
            H_res = sinkhorn_log_doubly_stochastic(
                H_res, n_iters=self.sinkhorn_iters, tau=self.sinkhorn_tau
            )

        H_pre = H_pre.view(B, S, n)
        H_post = H_post.view(B, S, n)
        H_res = H_res.view(B, S, n, n)

        H_pre_weights = H_pre / (H_pre.sum(dim=-1, keepdim=True) + self.eps)
        H_post_weights = H_post / (H_post.sum(dim=-1, keepdim=True) + self.eps)

        dtype = x_streams.dtype
        H_pre_weights = H_pre_weights.to(torch.float32)
        H_post_weights = H_post_weights.to(torch.float32)
        H_res = H_res.to(torch.float32)

        if self.use_triton and x_streams.is_cuda:
            x_layer = mhc_triton.pre_aggregate_triton_autograd(x_streams, H_pre_weights)
        else:
            x_layer = torch.einsum("bsnc,bsn->bsc", x_streams, H_pre_weights.to(dtype))
        x_layer = x_layer.to(dtype)

        y = self.sublayer(x_layer, **sublayer_kwargs)
        y_expanded = H_post_weights.to(dtype).unsqueeze(-1) * y.unsqueeze(2)

        if self.use_triton and x_streams.is_cuda:
            x_mixed = mhc_triton.residual_mix_triton_autograd(x_streams, H_res)
        else:
            x_mixed = torch.einsum("bsic,bsoi->bsoc", x_streams, H_res.to(dtype))
        x_mixed = x_mixed.to(dtype)
        return x_mixed + y_expanded

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    tbptt_steps: int = 0  # If > 0, detach carry every N ACT steps (truncated BPTT across steps)

    # mHC config
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_eps: float = 1e-6
    mhc_sinkhorn_tau: float = 0.05
    mhc_alpha_init: float = 1e-2
    mhc_use_triton: bool = False
    mhc_sinkhorn_triton_backward: bool = False

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
            self.mhc_attn = ManifoldHyperConnectionLayer(
                hidden_dim=config.hidden_size,
                n_streams=config.mhc_streams,
                sublayer=MHCAttentionSubLayer(self.self_attn),
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                eps=config.mhc_eps,
                sinkhorn_tau=config.mhc_sinkhorn_tau,
                alpha_init=config.mhc_alpha_init,
                use_triton=config.mhc_use_triton,
                triton_backward=config.mhc_sinkhorn_triton_backward,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        if not self.config.mlp_t:
            self.mhc_mlp = ManifoldHyperConnectionLayer(
                hidden_dim=config.hidden_size,
                n_streams=config.mhc_streams,
                sublayer=MHCFFNSubLayer(self.mlp),
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                eps=config.mhc_eps,
                sinkhorn_tau=config.mhc_sinkhorn_tau,
                alpha_init=config.mhc_alpha_init,
                use_triton=config.mhc_use_triton,
                triton_backward=config.mhc_sinkhorn_triton_backward,
            )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            return hidden_states

        hidden_states = self.mhc_attn(hidden_states, cos_sin=cos_sin)
        hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        hidden_states = self.mhc_mlp(hidden_states)
        hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [TinyRecursiveReasoningModel_ACTV1Block(config) for _ in range(config.L_layers)]
        )
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        *,
        recurrence_state=None,
        **kwargs,
    ):
        hidden_states = hidden_states + input_injection
        if self.config.mlp_t:
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
            return hidden_states, None

        B, S, C = hidden_states.shape
        n_streams = self.config.mhc_streams
        hidden_streams = hidden_states.unsqueeze(2).expand(B, S, n_streams, C).contiguous()
        for layer in self.layers:
            hidden_streams = layer(hidden_states=hidden_streams, **kwargs)
        hidden_states = hidden_streams.mean(dim=2)
        return hidden_states, None


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(self.config)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _apply_detach(self, tensor: torch.Tensor, detach_mask: torch.Tensor | None):
        if detach_mask is None:
            return tensor.detach()
        mask = detach_mask.view(-1, 1, 1)
        return torch.where(mask, tensor.detach(), tensor)

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        *,
        detach_mask: torch.Tensor | None = None,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                recurrence_state = None
                for _L_step in range(self.config.L_cycles):
                    z_L, recurrence_state = self.L_level(
                        z_L,
                        z_H + input_embeddings,
                        recurrence_state=recurrence_state,
                        **seq_info,
                    )
                recurrence_state = None
                z_H, _ = self.L_level(z_H, z_L, recurrence_state=None, **seq_info)
        # 1 with grad
        recurrence_state = None
        for _L_step in range(self.config.L_cycles):
            z_L, recurrence_state = self.L_level(
                z_L,
                z_H + input_embeddings,
                recurrence_state=recurrence_state,
                **seq_info,
            )
        z_H, _ = self.L_level(z_H, z_L, recurrence_state=None, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self._apply_detach(z_H, detach_mask),
            z_L=self._apply_detach(z_L, detach_mask),
        )  # New carry (detached by default; tbptt can keep grads)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Determine detach mask for truncated BPTT across ACT steps
        if self.training and self.config.tbptt_steps > 0:
            step_after = torch.where(carry.halted, 0, carry.steps) + 1
            detach_mask = (step_after % self.config.tbptt_steps == 0)
        else:
            detach_mask = None

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data,
            detach_mask=detach_mask,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and self.config.act_enabled and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

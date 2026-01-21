from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def _check_triton_available():
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")


@triton.jit
def _sinkhorn_log_kernel_with_uv(
    logits_ptr,
    out_ptr,
    u_hist_ptr,
    v_hist_ptr,
    stride_b,
    stride_r,
    stride_c,
    stride_u,
    stride_uiter,
    stride_udur,
    stride_v,
    stride_viter,
    stride_vdur,
    n_iters: tl.constexpr,
    tau: tl.constexpr,
    n: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_b
    rows = tl.arange(0, n)[:, None]
    cols = tl.arange(0, n)[None, :]
    offsets = base + rows * stride_r + cols * stride_c
    mask = (rows < n) & (cols < n)

    Z = tl.load(logits_ptr + offsets, mask=mask, other=0.0)
    Z = (Z / tau).to(tl.float32)

    u = tl.zeros((n,), dtype=tl.float32)
    v = tl.zeros((n,), dtype=tl.float32)

    for it in range(n_iters):
        Zv = Z + v[None, :]
        row_max = tl.max(Zv, axis=1)
        row_sum = tl.sum(tl.exp(Zv - row_max[:, None]), axis=1)
        u = - (row_max + tl.log(row_sum))

        Zu = Z + u[:, None]
        col_max = tl.max(Zu, axis=0)
        col_sum = tl.sum(tl.exp(Zu - col_max[None, :]), axis=0)
        v = - (col_max + tl.log(col_sum))

        u_off = pid * stride_u + it * stride_uiter + tl.arange(0, n) * stride_udur
        v_off = pid * stride_v + it * stride_viter + tl.arange(0, n) * stride_vdur
        tl.store(u_hist_ptr + u_off, u)
        tl.store(v_hist_ptr + v_off, v)

    out = tl.exp(Z + u[:, None] + v[None, :])
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _sinkhorn_log_backward_kernel(
    logits_ptr,
    grad_out_ptr,
    u_hist_ptr,
    v_hist_ptr,
    grad_logits_ptr,
    stride_b,
    stride_r,
    stride_c,
    stride_u,
    stride_uiter,
    stride_udur,
    stride_v,
    stride_viter,
    stride_vdur,
    n_iters: tl.constexpr,
    tau: tl.constexpr,
    n: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_b
    rows = tl.arange(0, n)[:, None]
    cols = tl.arange(0, n)[None, :]
    offsets = base + rows * stride_r + cols * stride_c
    mask = (rows < n) & (cols < n)

    Z = tl.load(logits_ptr + offsets, mask=mask, other=0.0)
    Z = (Z / tau).to(tl.float32)

    u_last_off = pid * stride_u + (n_iters - 1) * stride_uiter + tl.arange(0, n) * stride_udur
    v_last_off = pid * stride_v + (n_iters - 1) * stride_viter + tl.arange(0, n) * stride_vdur
    u = tl.load(u_hist_ptr + u_last_off).to(tl.float32)
    v = tl.load(v_hist_ptr + v_last_off).to(tl.float32)

    out = tl.exp(Z + u[:, None] + v[None, :])
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = grad_out * out

    dZ = g
    du = tl.sum(g, axis=1)
    dv = tl.sum(g, axis=0)

    for it in range(n_iters - 1, -1, -1):
        du_current = du
        dv_current = dv
        u_off = pid * stride_u + it * stride_uiter + tl.arange(0, n) * stride_udur
        v_off = pid * stride_v + it * stride_viter + tl.arange(0, n) * stride_vdur
        u = tl.load(u_hist_ptr + u_off).to(tl.float32)
        v = tl.load(v_hist_ptr + v_off).to(tl.float32)

        A = Z + u[:, None]
        col_max = tl.max(A, axis=0)
        col_sum = tl.sum(tl.exp(A - col_max[None, :]), axis=0)
        softmax_col = tl.exp(A - col_max[None, :]) / col_sum[None, :]
        dA = - dv_current[None, :] * softmax_col
        dZ = dZ + dA
        du_from_v = tl.sum(dA, axis=1)
        du_total = du_current + du_from_v

        if it > 0:
            v_prev_off = pid * stride_v + (it - 1) * stride_viter + tl.arange(0, n) * stride_vdur
            v_prev = tl.load(v_hist_ptr + v_prev_off).to(tl.float32)
        else:
            v_prev = tl.zeros((n,), dtype=tl.float32)

        B = Z + v_prev[None, :]
        row_max = tl.max(B, axis=1)
        row_sum = tl.sum(tl.exp(B - row_max[:, None]), axis=1)
        softmax_row = tl.exp(B - row_max[:, None]) / row_sum[:, None]
        dB = - du_total[:, None] * softmax_row
        dZ = dZ + dB
        dv = tl.sum(dB, axis=0)
        du = tl.zeros((n,), dtype=tl.float32)

    grad_logits = dZ / tau
    tl.store(grad_logits_ptr + offsets, grad_logits, mask=mask)


def sinkhorn_log_triton_with_uv(
    logits: torch.Tensor, n_iters: int = 20, tau: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_triton_available()
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("logits must be [..., n, n]")
    n = logits.shape[-1]
    if n > 16:
        raise ValueError("triton sinkhorn supports n <= 16")

    logits = logits.contiguous()
    out = torch.empty_like(logits, dtype=torch.float32)

    batch = logits.numel() // (n * n)
    logits_2d = logits.view(batch, n, n)
    out_2d = out.view(batch, n, n)

    u_hist = torch.empty((batch, n_iters, n), device=logits.device, dtype=torch.float32)
    v_hist = torch.empty((batch, n_iters, n), device=logits.device, dtype=torch.float32)

    _sinkhorn_log_kernel_with_uv[(batch,)](
        logits_2d,
        out_2d,
        u_hist,
        v_hist,
        logits_2d.stride(0),
        logits_2d.stride(1),
        logits_2d.stride(2),
        u_hist.stride(0),
        u_hist.stride(1),
        u_hist.stride(2),
        v_hist.stride(0),
        v_hist.stride(1),
        v_hist.stride(2),
        n_iters=n_iters,
        tau=tau,
        n=n,
    )
    return out, u_hist, v_hist


def sinkhorn_log_triton_backward(
    logits: torch.Tensor,
    grad_out: torch.Tensor,
    u_hist: torch.Tensor,
    v_hist: torch.Tensor,
    n_iters: int = 20,
    tau: float = 0.05,
) -> torch.Tensor:
    _check_triton_available()
    n = logits.shape[-1]
    if n > 16:
        raise ValueError("triton sinkhorn supports n <= 16")

    logits = logits.contiguous()
    grad_out = grad_out.contiguous()
    grad_logits = torch.empty_like(logits, dtype=torch.float32)

    batch = logits.numel() // (n * n)
    logits_2d = logits.view(batch, n, n)
    grad_out_2d = grad_out.view(batch, n, n)
    grad_logits_2d = grad_logits.view(batch, n, n)

    _sinkhorn_log_backward_kernel[(batch,)](
        logits_2d,
        grad_out_2d,
        u_hist,
        v_hist,
        grad_logits_2d,
        logits_2d.stride(0),
        logits_2d.stride(1),
        logits_2d.stride(2),
        u_hist.stride(0),
        u_hist.stride(1),
        u_hist.stride(2),
        v_hist.stride(0),
        v_hist.stride(1),
        v_hist.stride(2),
        n_iters=n_iters,
        tau=tau,
        n=n,
    )
    return grad_logits


class TritonSinkhornLogFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, n_iters: int, tau: float) -> torch.Tensor:
        out, u_hist, v_hist = sinkhorn_log_triton_with_uv(logits, n_iters=n_iters, tau=tau)
        ctx.save_for_backward(logits, u_hist, v_hist)
        ctx.n_iters = n_iters
        ctx.tau = tau
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, u_hist, v_hist = ctx.saved_tensors
        grad_logits = sinkhorn_log_triton_backward(
            logits, grad_output, u_hist, v_hist, n_iters=ctx.n_iters, tau=ctx.tau
        )
        return grad_logits, None, None


def sinkhorn_log_triton_autograd(
    logits: torch.Tensor, n_iters: int = 20, tau: float = 0.05
) -> torch.Tensor:
    return TritonSinkhornLogFn.apply(logits, n_iters, tau)

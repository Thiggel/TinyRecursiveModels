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


def sinkhorn_log_reference(logits: torch.Tensor, n_iters: int = 20, tau: float = 0.05) -> torch.Tensor:
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    for _ in range(n_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


@triton.jit
def _sinkhorn_log_kernel(
    logits_ptr,
    out_ptr,
    stride_b,
    stride_r,
    stride_c,
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
    Z = Z / tau

    u = tl.zeros((n,), dtype=tl.float32)
    v = tl.zeros((n,), dtype=tl.float32)

    Z = Z.to(tl.float32)

    for _ in range(n_iters):
        # u = -logsumexp(Z + v)
        Zv = Z + v[None, :]
        row_max = tl.max(Zv, axis=1)
        row_sum = tl.sum(tl.exp(Zv - row_max[:, None]), axis=1)
        u = - (row_max + tl.log(row_sum))

        # v = -logsumexp(Z + u)
        Zu = Z + u[:, None]
        col_max = tl.max(Zu, axis=0)
        col_sum = tl.sum(tl.exp(Zu - col_max[None, :]), axis=0)
        v = - (col_max + tl.log(col_sum))

    out = tl.exp(Z + u[:, None] + v[None, :])
    tl.store(out_ptr + offsets, out, mask=mask)


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

        # v_t = -logsumexp(Z + u_t) over rows
        A = Z + u[:, None]
        col_max = tl.max(A, axis=0)
        col_sum = tl.sum(tl.exp(A - col_max[None, :]), axis=0)
        softmax_col = tl.exp(A - col_max[None, :]) / col_sum[None, :]
        dA = - dv_current[None, :] * softmax_col
        dZ = dZ + dA
        du_from_v = tl.sum(dA, axis=1)
        du_total = du_current + du_from_v

        # u_t = -logsumexp(Z + v_{t-1}) over cols
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


def sinkhorn_log_triton(logits: torch.Tensor, n_iters: int = 20, tau: float = 0.05) -> torch.Tensor:
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

    _sinkhorn_log_kernel[(batch,)](
        logits_2d,
        out_2d,
        logits_2d.stride(0),
        logits_2d.stride(1),
        logits_2d.stride(2),
        n_iters=n_iters,
        tau=tau,
        n=n,
    )
    return out


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
    logits = logits.contiguous()
    grad_out = grad_out.contiguous()
    grad_logits = torch.empty_like(logits, dtype=torch.float32)

    batch = logits.numel() // (n * n)
    logits_2d = logits.view(batch, n, n)
    grad_out_2d = grad_out.view(batch, n, n)
    grad_logits_2d = grad_logits.view(batch, n, n)
    u_hist = u_hist.contiguous()
    v_hist = v_hist.contiguous()

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


@triton.jit
def _pre_aggregate_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_hm,
    stride_hn,
    stride_om,
    stride_oc,
    C: tl.constexpr,
    n: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * block_c + tl.arange(0, block_c)
    x_base = pid_m * stride_xm
    h_base = pid_m * stride_hm
    out_base = pid_m * stride_om

    acc = tl.zeros((block_c,), dtype=tl.float32)
    mask_c = c_offsets < C
    for k in range(n):
        h = tl.load(h_ptr + h_base + k * stride_hn).to(tl.float32)
        x = tl.load(
            x_ptr + x_base + k * stride_xn + c_offsets * stride_xc,
            mask=mask_c,
            other=0.0,
        )
        acc += x.to(tl.float32) * h

    tl.store(out_ptr + out_base + c_offsets * stride_oc, acc, mask=mask_c)


@triton.jit
def _residual_mix_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_hm,
    stride_ho,
    stride_hi,
    stride_om,
    stride_on,
    stride_oc,
    C: tl.constexpr,
    n: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_c = tl.program_id(2)

    c_offsets = pid_c * block_c + tl.arange(0, block_c)
    x_base = pid_m * stride_xm
    h_base = pid_m * stride_hm + pid_o * stride_ho
    out_base = pid_m * stride_om + pid_o * stride_on

    acc = tl.zeros((block_c,), dtype=tl.float32)
    mask_c = c_offsets < C
    for k in range(n):
        h = tl.load(h_ptr + h_base + k * stride_hi).to(tl.float32)
        x = tl.load(
            x_ptr + x_base + k * stride_xn + c_offsets * stride_xc,
            mask=mask_c,
            other=0.0,
        )
        acc += x.to(tl.float32) * h

    tl.store(out_ptr + out_base + c_offsets * stride_oc, acc, mask=mask_c)


def pre_aggregate_triton(x_streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    _check_triton_available()
    if x_streams.ndim != 4 or h_pre.ndim != 3:
        raise ValueError("x_streams must be [B,S,n,C] and h_pre [B,S,n]")

    B, S, n, C = x_streams.shape
    x = x_streams.reshape(B * S, n, C).contiguous()
    h = h_pre.reshape(B * S, n).contiguous()

    out = torch.empty((B * S, C), device=x.device, dtype=torch.float32)

    block_c = 128
    grid = (B * S, triton.cdiv(C, block_c))
    _pre_aggregate_kernel[grid](
        x,
        h,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        h.stride(0),
        h.stride(1),
        out.stride(0),
        out.stride(1),
        C=C,
        n=n,
        block_c=block_c,
    )

    return out.view(B, S, C)


def residual_mix_triton(x_streams: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    _check_triton_available()
    if x_streams.ndim != 4 or h_res.ndim != 4:
        raise ValueError("x_streams must be [B,S,n,C] and h_res [B,S,n,n]")

    B, S, n, C = x_streams.shape
    x = x_streams.reshape(B * S, n, C).contiguous()
    h = h_res.reshape(B * S, n, n).contiguous()

    out = torch.empty((B * S, n, C), device=x.device, dtype=torch.float32)

    block_c = 128
    grid = (B * S, n, triton.cdiv(C, block_c))
    _residual_mix_kernel[grid](
        x,
        h,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        h.stride(0),
        h.stride(1),
        h.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        C=C,
        n=n,
        block_c=block_c,
    )

    return out.view(B, S, n, C)


def maybe_cast_back(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype)


class TritonSinkhornLogFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, n_iters: int, tau: float, use_triton_backward: bool) -> torch.Tensor:
        ctx.n_iters = n_iters
        ctx.tau = tau
        ctx.use_triton_backward = use_triton_backward
        if use_triton_backward:
            out, u_hist, v_hist = sinkhorn_log_triton_with_uv(logits, n_iters=n_iters, tau=tau)
            ctx.save_for_backward(logits, u_hist, v_hist)
            return out
        ctx.save_for_backward(logits)
        return sinkhorn_log_triton(logits, n_iters=n_iters, tau=tau)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.use_triton_backward:
            logits, u_hist, v_hist = ctx.saved_tensors
            grad_logits = sinkhorn_log_triton_backward(
                logits, grad_out, u_hist, v_hist, n_iters=ctx.n_iters, tau=ctx.tau
            )
            return grad_logits, None, None, None
        (logits,) = ctx.saved_tensors
        logits_ref = logits.detach().requires_grad_(True)
        with torch.enable_grad():
            out_ref = sinkhorn_log_reference(logits_ref, n_iters=ctx.n_iters, tau=ctx.tau)
            grad_logits = torch.autograd.grad(
                out_ref, logits_ref, grad_out, retain_graph=False, create_graph=False
            )[0]
        return grad_logits, None, None, None


class TritonPreAggregateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x_streams, h_pre)
        return pre_aggregate_triton(x_streams, h_pre)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_streams, h_pre = ctx.saved_tensors
        grad_out_f = grad_out.to(torch.float32)
        grad_x = grad_out_f.unsqueeze(2) * h_pre.unsqueeze(-1)
        grad_h = torch.einsum("bsc,bsnc->bsn", grad_out_f, x_streams.to(torch.float32))
        return grad_x.to(x_streams.dtype), grad_h.to(h_pre.dtype)


class TritonResidualMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_streams: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x_streams, h_res)
        return residual_mix_triton(x_streams, h_res)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_streams, h_res = ctx.saved_tensors
        grad_out_f = grad_out.to(torch.float32)
        grad_x = torch.einsum("bsoc,bsoi->bsic", grad_out_f, h_res)
        grad_h = torch.einsum("bsoc,bsic->bsoi", grad_out_f, x_streams.to(torch.float32))
        return grad_x.to(x_streams.dtype), grad_h.to(h_res.dtype)


def sinkhorn_log_triton_autograd(
    logits: torch.Tensor,
    n_iters: int = 20,
    tau: float = 0.05,
    use_triton_backward: bool = False,
) -> torch.Tensor:
    return TritonSinkhornLogFn.apply(logits, n_iters, tau, use_triton_backward)


def pre_aggregate_triton_autograd(x_streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    return TritonPreAggregateFn.apply(x_streams, h_pre)


def residual_mix_triton_autograd(x_streams: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    return TritonResidualMixFn.apply(x_streams, h_res)

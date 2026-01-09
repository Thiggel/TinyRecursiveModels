import math

import torch

try:
    import triton
    import triton.testing
except Exception:  # pragma: no cover
    triton = None

from models.recursive_reasoning import mhc_triton
from models.recursive_reasoning.trm_mhc import sinkhorn_log_doubly_stochastic


def _require_cuda():
    if not torch.cuda.is_available():
        return False
    if triton is None:
        return False
    return True


def test_sinkhorn_log_triton_matches_reference():
    if not _require_cuda():
        return
    torch.manual_seed(0)
    B, n = 128, 4
    logits = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
    ref = sinkhorn_log_doubly_stochastic(logits, n_iters=10, tau=0.05)
    out = mhc_triton.sinkhorn_log_triton(logits, n_iters=10, tau=0.05)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_sinkhorn_log_triton_backward_close():
    if not _require_cuda():
        return
    torch.manual_seed(0)
    B, n = 64, 4
    logits = torch.randn(B, n, n, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(B, n, n, device="cuda", dtype=torch.float32)

    out_ref = sinkhorn_log_doubly_stochastic(logits, n_iters=8, tau=0.05)
    grad_ref = torch.autograd.grad(out_ref, logits, grad_out, retain_graph=False, create_graph=False)[0]

    out_triton = mhc_triton.sinkhorn_log_triton_autograd(
        logits, n_iters=8, tau=0.05, use_triton_backward=True
    )
    grad_triton = torch.autograd.grad(out_triton, logits, grad_out, retain_graph=False, create_graph=False)[0]

    torch.testing.assert_close(grad_triton, grad_ref, rtol=1e-3, atol=1e-3)


def test_pre_aggregate_triton_matches_einsum():
    if not _require_cuda():
        return
    torch.manual_seed(0)
    B, S, n, C = 8, 16, 4, 256
    x = torch.randn(B, S, n, C, device="cuda", dtype=torch.bfloat16)
    h = torch.rand(B, S, n, device="cuda", dtype=torch.float32)
    ref = torch.einsum("bsnc,bsn->bsc", x.to(torch.float32), h).to(torch.float32)
    out = mhc_triton.pre_aggregate_triton(x, h)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_residual_mix_triton_matches_einsum():
    if not _require_cuda():
        return
    torch.manual_seed(0)
    B, S, n, C = 4, 8, 4, 128
    x = torch.randn(B, S, n, C, device="cuda", dtype=torch.bfloat16)
    h = torch.rand(B, S, n, n, device="cuda", dtype=torch.float32)
    ref = torch.einsum("bsic,bsoi->bsoc", x.to(torch.float32), h).to(torch.float32)
    out = mhc_triton.residual_mix_triton(x, h)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


if triton is not None:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B", "S", "C"],
            x_vals=[(4, 32, 256), (8, 64, 512)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["torch", "triton"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="mhc_stream_ops",
            args={"n": 4},
        )
    )
    def benchmark_stream_ops(B, S, C, n, provider):
        if not _require_cuda():
            return math.nan
        x = torch.randn(B, S, n, C, device="cuda", dtype=torch.bfloat16)
        h_pre = torch.rand(B, S, n, device="cuda", dtype=torch.float32)
        h_res = torch.rand(B, S, n, n, device="cuda", dtype=torch.float32)

        if provider == "torch":
            def fn():
                _ = torch.einsum("bsnc,bsn->bsc", x, h_pre.to(x.dtype))
                _ = torch.einsum("bsic,bsoi->bsoc", x, h_res.to(x.dtype))
        else:
            def fn():
                _ = mhc_triton.pre_aggregate_triton(x, h_pre)
                _ = mhc_triton.residual_mix_triton(x, h_res)

        return triton.testing.do_bench(fn) * 1e3

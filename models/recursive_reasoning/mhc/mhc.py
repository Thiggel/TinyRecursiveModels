import math
import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# UTILS & CONSTANTS
# ============================================================================

def next_multiple_of_16(n):
    return ((n + 15) // 16) * 16

EPS_R = 1e-6
T_SINKHORN = 20

def _get_sinkhorn_iters():
    return int(os.environ.get("MHC_SINKHORN_ITERS", T_SINKHORN))

def _get_block_m():
    return int(os.environ.get("MHC_BLOCK_M", 64))

def _get_block_k():
    return int(os.environ.get("MHC_BLOCK_K", 64))

# ============================================================================
# TRITON KERNELS
# ============================================================================

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def _sinkhorn_fwd_flat(M, iters: tl.constexpr, N: tl.constexpr):
    # Flattened Sinkhorn for N<=4 (Register-based)
    idx = tl.arange(0, 16)
    row_idx = idx // N
    col_idx = idx % N
    valid_mask = idx < N*N
    M_curr = M
    
    for _ in range(iters):
        # Row Norm
        for r in range(N):
            r_mask = (row_idx == r) & valid_mask
            row_sum = tl.sum(tl.where(r_mask[None, :], M_curr, 0.0), axis=1)
            M_curr = tl.where(r_mask[None, :], M_curr / (row_sum[:, None] + 1e-12), M_curr)
        # Col Norm
        for c in range(N):
            c_mask = (col_idx == c) & valid_mask
            col_sum = tl.sum(tl.where(c_mask[None, :], M_curr, 0.0), axis=1)
            M_curr = tl.where(c_mask[None, :], M_curr / (col_sum[:, None] + 1e-12), M_curr)
    return M_curr

@triton.jit
def _sinkhorn_bwd_step_flat(logits, grad_out, iters: tl.constexpr, N: tl.constexpr):
    idx = tl.arange(0, 16)
    row_idx = idx // N
    col_idx = idx % N
    valid_mask = idx < N*N
    g = grad_out
    
    for t in range(iters):
        curr_iter = iters - 1 - t
        m_curr = tl.exp(logits)
        m_row_normed = m_curr 
        
        # --- Forward Replay ---
        for k in range(curr_iter + 1):
            for r in range(N):
                r_mask = (row_idx == r) & valid_mask
                row_sum = tl.sum(tl.where(r_mask[None, :], m_curr, 0.0), axis=1)
                m_curr = tl.where(r_mask[None, :], m_curr / (row_sum[:, None] + 1e-12), m_curr)
            if k == curr_iter: m_row_normed = m_curr 
            if k < curr_iter:
                for c in range(N):
                    c_mask = (col_idx == c) & valid_mask
                    col_sum = tl.sum(tl.where(c_mask[None, :], m_curr, 0.0), axis=1)
                    m_curr = tl.where(c_mask[None, :], m_curr / (col_sum[:, None] + 1e-12), m_curr)
        
        # --- VJP Col Norm ---
        for c in range(N):
            c_mask = (col_idx == c) & valid_mask
            col_sum = tl.sum(tl.where(c_mask[None, :], m_curr, 0.0), axis=1)
            inv_c = 1.0 / (col_sum[:, None] + 1e-12)
            m_out_c = tl.where(c_mask[None, :], m_curr * inv_c, 0.0)
            dot = tl.sum(tl.where(c_mask[None, :], g * m_out_c, 0.0), axis=1)
            g = tl.where(c_mask[None, :], (g - dot[:, None]) * inv_c, g)
        
        # --- Forward Replay (Pre-Row) ---
        m_prev = tl.exp(logits)
        if curr_iter > 0:
            for k in range(curr_iter):
                for r in range(N):
                    r_mask = (row_idx == r) & valid_mask
                    row_sum = tl.sum(tl.where(r_mask[None, :], m_prev, 0.0), axis=1)
                    m_prev = tl.where(r_mask[None, :], m_prev / (row_sum[:, None] + 1e-12), m_prev)
                for c in range(N):
                    c_mask = (col_idx == c) & valid_mask
                    col_sum = tl.sum(tl.where(c_mask[None, :], m_prev, 0.0), axis=1)
                    m_prev = tl.where(c_mask[None, :], m_prev / (col_sum[:, None] + 1e-12), m_prev)
                    
        # --- VJP Row Norm ---
        for r in range(N):
            r_mask = (row_idx == r) & valid_mask
            row_sum = tl.sum(tl.where(r_mask[None, :], m_prev, 0.0), axis=1)
            inv_r = 1.0 / (row_sum[:, None] + 1e-12)
            dot_r = tl.sum(tl.where(r_mask[None, :], g * m_row_normed, 0.0), axis=1)
            g = tl.where(r_mask[None, :], (g - dot_r[:, None]) * inv_r, g)

    return g * tl.exp(logits)

@triton.jit
def mhc_coeffs_fwd_kernel(
    x_ptr, phi_ptr, b_ptr, alpha_pre_ptr, alpha_post_ptr, alpha_res_ptr,
    Hpre_ptr, Hpost_ptr, Hres_ptr, B, NC, stride_xm, stride_xk, stride_phik, stride_phin,
    iters: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, N: tl.constexpr, PAD: tl.constexpr, EPS: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B
    offs_m_safe = tl.where(mask_m, offs_m, 0)
    
    y_res = tl.zeros((BLOCK_M, PAD), dtype=tl.float32)
    y_pre = tl.zeros((BLOCK_M, PAD), dtype=tl.float32)
    y_post = tl.zeros((BLOCK_M, PAD), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    phi_res_ptr = phi_ptr
    phi_pre_ptr = phi_ptr + PAD * stride_phin
    phi_post_ptr = phi_ptr + 2 * PAD * stride_phin
    
    # FIX: Use Python range for loop
    for k in range(0, NC, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < NC
        
        # Safe Pointer Clamping
        offs_k_safe = tl.where(mask_k, offs_k, 0)
        ptr_x = x_ptr + offs_m_safe[:, None] * stride_xm + offs_k_safe[None, :] * stride_xk
        x_chunk = tl.load(ptr_x, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        
        sumsq += tl.sum(x_chunk * x_chunk, axis=1)
        
        idx_pad = tl.arange(0, PAD)
        y_res += tl.dot(x_chunk, tl.load(phi_res_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))
        y_pre += tl.dot(x_chunk, tl.load(phi_pre_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))
        y_post += tl.dot(x_chunk, tl.load(phi_post_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))

    r = tl.sqrt(sumsq) / tl.sqrt(NC.to(tl.float32))
    r = tl.maximum(r, EPS)
    y_res = y_res / r[:, None]; y_pre = y_pre / r[:, None]; y_post = y_post / r[:, None]
    
    a_pre = tl.load(alpha_pre_ptr); a_post = tl.load(alpha_post_ptr); a_res = tl.load(alpha_res_ptr)
    idx_pad = tl.arange(0, PAD)
    b_res = tl.load(b_ptr + idx_pad, mask=idx_pad < N*N, other=0.0).to(tl.float32)
    b_pre = tl.load(b_ptr + PAD + idx_pad, mask=idx_pad < N, other=0.0).to(tl.float32)
    b_post = tl.load(b_ptr + 2*PAD + idx_pad, mask=idx_pad < N, other=0.0).to(tl.float32)
    
    h_res = _sinkhorn_fwd_flat(tl.exp(y_res * a_res + b_res[None, :]), iters, N)
    h_pre = _sigmoid(y_pre * a_pre + b_pre[None, :])
    h_post = 2.0 * _sigmoid(y_post * a_post + b_post[None, :])

    offs_pad = tl.arange(0, PAD)
    
    mask_store_pre = mask_m[:, None] & (offs_pad[None, :] < N)
    ptr_pre = Hpre_ptr + offs_m[:, None]*N + offs_pad[None, :]
    tl.store(tl.where(mask_store_pre, ptr_pre, Hpre_ptr), h_pre, mask=mask_store_pre)
    
    mask_store_post = mask_m[:, None] & (offs_pad[None, :] < N)
    ptr_post = Hpost_ptr + offs_m[:, None]*N + offs_pad[None, :]
    tl.store(tl.where(mask_store_post, ptr_post, Hpost_ptr), h_post, mask=mask_store_post)
    
    mask_store_res = mask_m[:, None] & (offs_pad[None, :] < N*N)
    ptr_res = Hres_ptr + offs_m[:, None]*N*N + offs_pad[None, :]
    tl.store(tl.where(mask_store_res, ptr_res, Hres_ptr), h_res, mask=mask_store_res)

@triton.jit
def mhc_coeffs_bwd_kernel(
    x_ptr, phi_ptr, b_ptr, alpha_pre_ptr, alpha_post_ptr, alpha_res_ptr,
    grad_pre_ptr, grad_post_ptr, grad_res_ptr, dy_res_ptr,
    dx_ptr, dphi_ptr, db_ptr, da_pre_ptr, da_post_ptr, da_res_ptr,
    B, NC, stride_xm, stride_xk, stride_phik, stride_phin, stride_dxm, stride_dxk,
    iters: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, N: tl.constexpr, PAD: tl.constexpr, EPS: tl.constexpr,
    SKIP_ATOMIC: tl.constexpr, SKIP_SECOND: tl.constexpr, SKIP_RES: tl.constexpr, SIMPLE_RES: tl.constexpr, USE_DY_RES: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B
    offs_m_safe = tl.where(mask_m, offs_m, 0)
    
    y_res = tl.zeros((BLOCK_M, PAD), dtype=tl.float32); y_pre = tl.zeros((BLOCK_M, PAD), dtype=tl.float32)
    y_post = tl.zeros((BLOCK_M, PAD), dtype=tl.float32); sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    phi_res_ptr = phi_ptr; phi_pre_ptr = phi_ptr + PAD * stride_phin; phi_post_ptr = phi_ptr + 2 * PAD * stride_phin
    
    # FIX: Use Python range for loop
    for k in range(0, NC, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K); mask_k = offs_k < NC
        
        offs_k_safe = tl.where(mask_k, offs_k, 0)
        ptr_x = x_ptr + offs_m_safe[:, None] * stride_xm + offs_k_safe[None, :] * stride_xk
        x_chunk = tl.load(ptr_x, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        
        sumsq += tl.sum(x_chunk * x_chunk, axis=1)
        y_res += tl.dot(x_chunk, tl.load(phi_res_ptr + offs_k[:, None]*stride_phik + tl.arange(0, PAD)[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))
        y_pre += tl.dot(x_chunk, tl.load(phi_pre_ptr + offs_k[:, None]*stride_phik + tl.arange(0, PAD)[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))
        y_post += tl.dot(x_chunk, tl.load(phi_post_ptr + offs_k[:, None]*stride_phik + tl.arange(0, PAD)[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32))

    r_inv = tl.sqrt(NC.to(tl.float32)) / (tl.sqrt(sumsq) + EPS)
    y_norm_res = y_res * r_inv[:, None]; y_norm_pre = y_pre * r_inv[:, None]; y_norm_post = y_post * r_inv[:, None]
    
    a_pre = tl.load(alpha_pre_ptr); a_post = tl.load(alpha_post_ptr); a_res = tl.load(alpha_res_ptr)
    idx_pad = tl.arange(0, PAD); mask_res_b = idx_pad < N*N; mask_pp_b = idx_pad < N
    b_res = tl.load(b_ptr + idx_pad, mask=mask_res_b, other=0.0).to(tl.float32)
    b_pre = tl.load(b_ptr + PAD + idx_pad, mask=mask_pp_b, other=0.0).to(tl.float32)
    b_post = tl.load(b_ptr + 2*PAD + idx_pad, mask=mask_pp_b, other=0.0).to(tl.float32)
    
    idx_safe_pp = tl.where(mask_pp_b, idx_pad, 0)
    idx_safe_res = tl.where(mask_res_b, idx_pad, 0)
    
    ptr_dh_pre = grad_pre_ptr + offs_m_safe[:, None]*N + idx_safe_pp[None, :]
    mask_load_pre = mask_m[:, None] & mask_pp_b[None, :]
    dh_pre = tl.load(ptr_dh_pre, mask=mask_load_pre, other=0.0)
    
    ptr_dh_post = grad_post_ptr + offs_m_safe[:, None]*N + idx_safe_pp[None, :]
    mask_load_post = mask_m[:, None] & mask_pp_b[None, :]
    dh_post = tl.load(ptr_dh_post, mask=mask_load_post, other=0.0)
    
    ptr_dh_res = grad_res_ptr + offs_m_safe[:, None]*N*N + idx_safe_res[None, :]
    mask_load_res = mask_m[:, None] & mask_res_b[None, :]
    dh_res = tl.load(ptr_dh_res, mask=mask_load_res, other=0.0)
    
    lin_pre = y_norm_pre * a_pre + b_pre[None, :]; sig_pre = _sigmoid(lin_pre)
    dy_pre_sub = tl.where(mask_pp_b[None, :], dh_pre * sig_pre * (1.0 - sig_pre), 0.0)
    lin_post = y_norm_post * a_post + b_post[None, :]; sig_post = _sigmoid(lin_post)
    dy_post_sub = tl.where(mask_pp_b[None, :], dh_post * 2.0 * sig_post * (1.0 - sig_post), 0.0)
    if USE_DY_RES == 1:
        ptr_dy_res = dy_res_ptr + offs_m_safe[:, None]*PAD + idx_safe_res[None, :]
        dy_res_sub = tl.load(ptr_dy_res, mask=mask_m[:, None] & mask_res_b[None, :], other=0.0)
    else:
        if SKIP_RES == 0:
            lin_res = y_norm_res * a_res + b_res[None, :]
            if SIMPLE_RES == 1:
                dy_res_sub = tl.where(mask_res_b[None, :], dh_res * tl.exp(lin_res), 0.0)
            else:
                dy_res_sub = tl.where(mask_res_b[None, :], _sinkhorn_bwd_step_flat(lin_res, dh_res, iters, N), 0.0)
        else:
            dy_res_sub = tl.zeros((BLOCK_M, PAD), dtype=tl.float32)
    
    if SKIP_ATOMIC == 0:
        tl.atomic_add(da_pre_ptr, tl.sum(dy_pre_sub * y_norm_pre))
        tl.atomic_add(da_post_ptr, tl.sum(dy_post_sub * y_norm_post))
        tl.atomic_add(da_res_ptr, tl.sum(dy_res_sub * y_norm_res))
        tl.atomic_add(db_ptr + idx_pad, tl.sum(dy_res_sub, axis=0), mask=mask_res_b)
        tl.atomic_add(db_ptr + PAD + idx_pad, tl.sum(dy_pre_sub, axis=0), mask=mask_pp_b)
        tl.atomic_add(db_ptr + 2*PAD + idx_pad, tl.sum(dy_post_sub, axis=0), mask=mask_pp_b)
    
    val_pre = dy_pre_sub * a_pre; val_post = dy_post_sub * a_post; val_res = dy_res_sub * a_res
    term2_scalar = -(tl.sum(val_pre * y_norm_pre, axis=1) + tl.sum(val_post * y_norm_post, axis=1) + tl.sum(val_res * y_norm_res, axis=1)) * r_inv
    
    dphi_res_ptr = dphi_ptr; dphi_pre_ptr = dphi_ptr + PAD * stride_phin; dphi_post_ptr = dphi_ptr + 2 * PAD * stride_phin
    
    if SKIP_SECOND == 0:
        for k in range(0, NC, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K); mask_k = offs_k < NC
            
            offs_k_safe = tl.where(mask_k, offs_k, 0)
            ptr_x = x_ptr + offs_m_safe[:, None] * stride_xm + offs_k_safe[None, :] * stride_xk
            x_chunk = tl.load(ptr_x, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            
            phi_res = tl.load(phi_res_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32)
            phi_pre = tl.load(phi_pre_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32)
            phi_post = tl.load(phi_post_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, mask=mask_k[:, None], other=0.0).to(tl.float32)
            
            dx_dot = tl.dot(val_res * r_inv[:, None], tl.trans(phi_res)) + tl.dot(val_pre * r_inv[:, None], tl.trans(phi_pre)) + tl.dot(val_post * r_inv[:, None], tl.trans(phi_post))
            dx_norm = term2_scalar[:, None] * x_chunk * r_inv[:, None] / NC.to(tl.float32)
            
            ptr_dx = dx_ptr + offs_m_safe[:, None] * stride_dxm + offs_k_safe[None, :] * stride_dxk
            mask_store_dx = mask_m[:, None] & mask_k[None, :]
            tl.store(ptr_dx, (dx_dot + dx_norm).to(tl.bfloat16), mask=mask_store_dx)
            
            dphi_res = tl.dot(tl.trans(x_chunk), val_res * r_inv[:, None])
            dphi_pre = tl.dot(tl.trans(x_chunk), val_pre * r_inv[:, None])
            dphi_post = tl.dot(tl.trans(x_chunk), val_post * r_inv[:, None])
            
            if SKIP_ATOMIC == 0:
                tl.atomic_add(dphi_res_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, dphi_res, mask=mask_k[:, None] & mask_res_b[None, :])
                tl.atomic_add(dphi_pre_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, dphi_pre, mask=mask_k[:, None] & mask_pp_b[None, :])
                tl.atomic_add(dphi_post_ptr + offs_k_safe[:, None]*stride_phik + idx_pad[None, :]*stride_phin, dphi_post, mask=mask_k[:, None] & mask_pp_b[None, :])

# ============================================================================
# APPLY KERNELS & MODULE
# ============================================================================

@triton.jit
def mhc_apply_pre_fwd_kernel(x_ptr, h_pre_ptr, z_ptr, stride_xb, stride_xn, stride_xc, stride_hb, stride_hn, stride_zb, stride_zc, B, C, BLOCK_C: tl.constexpr, N: tl.constexpr):
    pid_b, pid_c = tl.program_id(0), tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C); mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for i in range(N):
        h_val = tl.load(h_pre_ptr + pid_b * stride_hb + i * stride_hn)
        ptr_x = x_ptr + pid_b * stride_xb + i * stride_xn + offs_c * stride_xc
        val = tl.load(tl.where(mask_c, ptr_x, x_ptr), mask=mask_c, other=0.0).to(tl.float32)
        acc += val * h_val
    ptr_z = z_ptr + pid_b * stride_zb + offs_c * stride_zc
    tl.store(tl.where(mask_c, ptr_z, z_ptr), acc, mask=mask_c)

@triton.jit
def mhc_apply_pre_bwd_kernel(x_ptr, h_pre_ptr, dz_ptr, dx_ptr, dh_pre_ptr, stride_xb, stride_xn, stride_xc, stride_hb, stride_hn, stride_zb, stride_zc, stride_dxb, stride_dxn, stride_dxc, stride_dhb, stride_dhn, B, C, BLOCK_C: tl.constexpr, N: tl.constexpr):
    pid_b, pid_c = tl.program_id(0), tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C); mask_c = offs_c < C
    ptr_dz = dz_ptr + pid_b * stride_zb + offs_c * stride_zc
    dz = tl.load(tl.where(mask_c, ptr_dz, dz_ptr), mask=mask_c, other=0.0).to(tl.float32)
    for i in range(N):
        h_val = tl.load(h_pre_ptr + pid_b * stride_hb + i * stride_hn)
        ptr_dx = dx_ptr + pid_b * stride_dxb + i * stride_dxn + offs_c * stride_dxc
        tl.store(tl.where(mask_c, ptr_dx, dx_ptr), (dz * h_val).to(tl.bfloat16), mask=mask_c)
        ptr_x = x_ptr + pid_b * stride_xb + i * stride_xn + offs_c * stride_xc
        x_val = tl.load(tl.where(mask_c, ptr_x, x_ptr), mask=mask_c, other=0.0).to(tl.float32)
        tl.atomic_add(dh_pre_ptr + pid_b * stride_dhb + i * stride_dhn, tl.sum(dz * x_val))

@triton.jit
def mhc_apply_postres_fwd_kernel(x_ptr, y_ptr, h_post_ptr, h_res_ptr, out_ptr, stride_xb, stride_xn, stride_xc, stride_yb, stride_yc, stride_hp_b, stride_hp_n, stride_hr_b, stride_hr_n, stride_ob, stride_on, stride_oc, B, C, BLOCK_C: tl.constexpr, N: tl.constexpr):
    pid_b, pid_n, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C); mask_c = offs_c < C
    h_p = tl.load(h_post_ptr + pid_b * stride_hp_b + pid_n * stride_hp_n)
    ptr_y = y_ptr + pid_b * stride_yb + offs_c * stride_yc
    y_val = tl.load(tl.where(mask_c, ptr_y, y_ptr), mask=mask_c, other=0.0).to(tl.float32)
    out = h_p * y_val
    for j in range(N):
        h_r_val = tl.load(h_res_ptr + pid_b * stride_hr_b + (pid_n * N + j))
        ptr_x = x_ptr + pid_b * stride_xb + j * stride_xn + offs_c * stride_xc
        x_val = tl.load(tl.where(mask_c, ptr_x, x_ptr), mask=mask_c, other=0.0).to(tl.float32)
        out += h_r_val * x_val
    ptr_out = out_ptr + pid_b * stride_ob + pid_n * stride_on + offs_c * stride_oc
    tl.store(tl.where(mask_c, ptr_out, out_ptr), out.to(tl.bfloat16), mask=mask_c)

@triton.jit
def mhc_apply_postres_bwd_kernel(x_ptr, y_ptr, h_post_ptr, h_res_ptr, dout_ptr, dx_ptr, dy_ptr, dh_post_ptr, dh_res_ptr, stride_xb, stride_xn, stride_xc, stride_yb, stride_yc, stride_hp_b, stride_hp_n, stride_hr_b, stride_hr_n, stride_dob, stride_don, stride_doc, stride_dxb, stride_dxn, stride_dxc, stride_dyb, stride_dyc, stride_dhp_b, stride_dhp_n, stride_dhr_b, stride_dhr_n, B, C, BLOCK_C: tl.constexpr, N: tl.constexpr):
    pid_b, pid_n, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C); mask_c = offs_c < C
    ptr_dout = dout_ptr + pid_b * stride_dob + pid_n * stride_don + offs_c * stride_doc
    dout = tl.load(tl.where(mask_c, ptr_dout, dout_ptr), mask=mask_c, other=0.0).to(tl.float32)
    h_p = tl.load(h_post_ptr + pid_b * stride_hp_b + pid_n * stride_hp_n)
    ptr_y = y_ptr + pid_b * stride_yb + offs_c * stride_yc
    y_val = tl.load(tl.where(mask_c, ptr_y, y_ptr), mask=mask_c, other=0.0).to(tl.float32)
    ptr_dy = dy_ptr + pid_b * stride_dyb + offs_c * stride_dyc
    tl.atomic_add(tl.where(mask_c, ptr_dy, dy_ptr), dout * h_p, mask=mask_c)
    tl.atomic_add(dh_post_ptr + pid_b * stride_dhp_b + pid_n * stride_dhp_n, tl.sum(dout * y_val))
    for j in range(N):
        h_r_val = tl.load(h_res_ptr + pid_b * stride_hr_b + (pid_n * N + j))
        ptr_x = x_ptr + pid_b * stride_xb + j * stride_xn + offs_c * stride_xc
        x_val = tl.load(tl.where(mask_c, ptr_x, x_ptr), mask=mask_c, other=0.0).to(tl.float32)
        ptr_dx = dx_ptr + pid_b * stride_dxb + j * stride_dxn + offs_c * stride_dxc
        tl.atomic_add(tl.where(mask_c, ptr_dx, dx_ptr), dout * h_r_val, mask=mask_c)
        tl.atomic_add(dh_res_ptr + pid_b * stride_dhr_b + (pid_n * N + j), tl.sum(dout * x_val))

@triton.jit
def mhc_sinkhorn_bwd_kernel(logits_ptr, grad_ptr, out_ptr, B, stride_lb, stride_lk, stride_gb, stride_gk, stride_ob, stride_ok, iters: tl.constexpr, N: tl.constexpr, PAD: tl.constexpr, SIMPLE_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, PAD)
    mask = (pid < B) & (offs < N*N)
    logits = tl.load(logits_ptr + pid * stride_lb + offs * stride_lk, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + pid * stride_gb + offs * stride_gk, mask=mask, other=0.0).to(tl.float32)
    if SIMPLE_RES == 1:
        dy = grad * tl.exp(logits)
        tl.store(out_ptr + pid * stride_ob + offs * stride_ok, dy, mask=mask)
    else:
        logits = logits[None, :]
        grad = grad[None, :]
        dy = _sinkhorn_bwd_step_flat(logits, grad, iters, N)
        dy_1d = tl.sum(dy, axis=0)
        tl.store(out_ptr + pid * stride_ob + offs * stride_ok, dy_1d, mask=mask)

class MHCCoeffsFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, phi, b, alpha_pre, alpha_post, alpha_res, n):
        B, NC = x_flat.shape
        pad = 16 
        Hpre = torch.empty(B, n, device=x_flat.device, dtype=torch.float32)
        Hpost = torch.empty(B, n, device=x_flat.device, dtype=torch.float32)
        Hres = torch.empty(B, n*n, device=x_flat.device, dtype=torch.float32)
        block_m = _get_block_m()
        block_k = _get_block_k()
        mhc_coeffs_fwd_kernel[(triton.cdiv(B, block_m),)](
            x_flat, phi, b, alpha_pre, alpha_post, alpha_res, Hpre, Hpost, Hres,
            B, NC, x_flat.stride(0), x_flat.stride(1), phi.stride(0), phi.stride(1),
            iters=_get_sinkhorn_iters(), BLOCK_M=block_m, BLOCK_K=block_k, N=n, PAD=pad, EPS=EPS_R
        )
        ctx.save_for_backward(x_flat, phi, b, alpha_pre, alpha_post, alpha_res)
        ctx.n = n; ctx.pad = pad
        return Hpre, Hpost, Hres

    @staticmethod
    def backward(ctx, grad_pre, grad_post, grad_res):
        x_flat, phi, b, alpha_pre, alpha_post, alpha_res = ctx.saved_tensors
        B, NC = x_flat.shape
        n, pad = ctx.n, ctx.pad
        dx = torch.zeros_like(x_flat); dphi = torch.zeros_like(phi, dtype=torch.float32); db = torch.zeros_like(b, dtype=torch.float32)
        da_pre_buf = torch.zeros(1, device=alpha_pre.device, dtype=torch.float32)
        da_post_buf = torch.zeros(1, device=alpha_post.device, dtype=torch.float32)
        da_res_buf = torch.zeros(1, device=alpha_res.device, dtype=torch.float32)
        dy_res_sub = None
        use_torch_res = bool(int(os.environ.get("MHC_TORCH_RES_BWD", "1")))
        if use_torch_res:
            with torch.enable_grad():
                x_flat_f = x_flat.detach().float()
                phi_f = phi.detach().float()
                r = x_flat_f.norm(dim=1, keepdim=True) / math.sqrt(NC)
                r = torch.clamp(r, min=EPS_R)
                y_res = (x_flat_f @ phi_f[:, 0:pad]) / r
                lin_res = y_res[:, :n*n] * alpha_res.detach().float() + b.detach().float()[:n*n]
                logits = lin_res.view(B, n, n).clone().requires_grad_(True)
                M = torch.exp(logits)
                for _ in range(_get_sinkhorn_iters()):
                    M = M / (M.sum(dim=2, keepdim=True) + 1e-12)
                    M = M / (M.sum(dim=1, keepdim=True) + 1e-12)
                grad_in = grad_res.detach().float().view(B, n, n)
                grad_logits = torch.autograd.grad(M, logits, grad_outputs=grad_in, retain_graph=False, create_graph=False)[0]
                dy_res_sub = torch.zeros((B, pad), device=x_flat.device, dtype=torch.float32)
                dy_res_sub[:, :n*n] = grad_logits.reshape(B, n*n)
        else:
            with torch.no_grad():
                x_flat_f = x_flat.detach().float()
                phi_f = phi.detach().float()
                r = x_flat_f.norm(dim=1, keepdim=True) / math.sqrt(NC)
                r = torch.clamp(r, min=EPS_R)
                y_res = (x_flat_f @ phi_f[:, 0:pad]) / r
                lin_res = y_res[:, :n*n] * alpha_res.detach().float() + b.detach().float()[:n*n]
                logits_padded = torch.zeros((B, pad), device=x_flat.device, dtype=torch.float32)
                logits_padded[:, :n*n] = lin_res
                dy_res_sub = torch.empty_like(logits_padded)
                simple_res_bwd = int(os.environ.get("MHC_SIMPLE_RES_BWD", "0"))
                mhc_sinkhorn_bwd_kernel[(B,)](
                    logits_padded, grad_res.float(), dy_res_sub,
                    B, logits_padded.stride(0), logits_padded.stride(1),
                    grad_res.stride(0), grad_res.stride(1),
                    dy_res_sub.stride(0), dy_res_sub.stride(1),
                    iters=_get_sinkhorn_iters(), N=n, PAD=pad, SIMPLE_RES=simple_res_bwd
                )
        block_m = _get_block_m()
        block_k = _get_block_k()
        skip_atomic = int(os.environ.get("MHC_SKIP_ATOMIC", "0"))
        skip_second = int(os.environ.get("MHC_SKIP_SECOND", "0"))
        skip_res = int(os.environ.get("MHC_SKIP_RES", "0"))
        simple_res = int(os.environ.get("MHC_SIMPLE_RES", "0"))
        use_dy_res = int(dy_res_sub is not None)
        dy_res_ptr = dy_res_sub if dy_res_sub is not None else grad_res
        mhc_coeffs_bwd_kernel[(triton.cdiv(B, block_m),)](
            x_flat, phi, b, alpha_pre, alpha_post, alpha_res, grad_pre, grad_post, grad_res, dy_res_ptr,
            dx, dphi, db, da_pre_buf, da_post_buf, da_res_buf, B, NC, x_flat.stride(0), x_flat.stride(1),
            phi.stride(0), phi.stride(1), dx.stride(0), dx.stride(1), iters=_get_sinkhorn_iters(), BLOCK_M=block_m, BLOCK_K=block_k, N=n, PAD=pad, EPS=EPS_R,
            SKIP_ATOMIC=skip_atomic, SKIP_SECOND=skip_second, SKIP_RES=skip_res, SIMPLE_RES=simple_res, USE_DY_RES=use_dy_res
        )
        da_pre = da_pre_buf[0].to(alpha_pre.dtype)
        da_post = da_post_buf[0].to(alpha_post.dtype)
        da_res = da_res_buf[0].to(alpha_res.dtype)
        return dx, dphi.to(phi.dtype), db, da_pre, da_post, da_res, None

class MHCApplyPreFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_pre):
        z = torch.empty(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32)
        mhc_apply_pre_fwd_kernel[(x.shape[0], triton.cdiv(x.shape[2], 256))](
            x, h_pre, z, x.stride(0), x.stride(1), x.stride(2), h_pre.stride(0), h_pre.stride(1),
            z.stride(0), z.stride(1), x.shape[0], x.shape[2], BLOCK_C=256, N=x.shape[1]
        )
        ctx.save_for_backward(x, h_pre)
        return z
    @staticmethod
    def backward(ctx, dz):
        x, h_pre = ctx.saved_tensors
        dx = torch.empty_like(x); dh_pre = torch.zeros_like(h_pre)
        mhc_apply_pre_bwd_kernel[(x.shape[0], triton.cdiv(x.shape[2], 256))](
            x, h_pre, dz, dx, dh_pre, x.stride(0), x.stride(1), x.stride(2), h_pre.stride(0), h_pre.stride(1),
            dz.stride(0), dz.stride(1), dx.stride(0), dx.stride(1), dx.stride(2), dh_pre.stride(0), dh_pre.stride(1),
            x.shape[0], x.shape[2], BLOCK_C=256, N=x.shape[1]
        )
        return dx, dh_pre

class MHCApplyPostResFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, h_post, h_res):
        out = torch.empty_like(x)
        mhc_apply_postres_fwd_kernel[(x.shape[0], x.shape[1], triton.cdiv(x.shape[2], 256))](
            x, y, h_post, h_res, out, x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1),
            h_post.stride(0), h_post.stride(1), h_res.stride(0), h_res.stride(1), out.stride(0), out.stride(1), out.stride(2),
            x.shape[0], x.shape[2], BLOCK_C=256, N=x.shape[1]
        )
        ctx.save_for_backward(x, y, h_post, h_res)
        return out
    @staticmethod
    def backward(ctx, dout):
        x, y, h_post, h_res = ctx.saved_tensors
        dx = torch.zeros_like(x, dtype=torch.float32); dy = torch.zeros_like(y, dtype=torch.float32)
        dh_post = torch.zeros_like(h_post); dh_res = torch.zeros_like(h_res)
        mhc_apply_postres_bwd_kernel[(x.shape[0], x.shape[1], triton.cdiv(x.shape[2], 256))](
            x, y, h_post, h_res, dout, dx, dy, dh_post, dh_res, x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1),
            h_post.stride(0), h_post.stride(1), h_res.stride(0), h_res.stride(1), dout.stride(0), dout.stride(1), dout.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2), dy.stride(0), dy.stride(1), dh_post.stride(0), dh_post.stride(1), dh_res.stride(0), dh_res.stride(1),
            x.shape[0], x.shape[2], BLOCK_C=256, N=x.shape[1]
        )
        return dx.to(x.dtype), dy.to(y.dtype), dh_post, dh_res

class MHC(nn.Module):
    def __init__(self, inner, dim, n=4):
        super().__init__()
        self.n = n; self.dim = dim; self.pad = 16 
        self.k_total_padded = 3 * self.pad
        self.phi = nn.Parameter(torch.randn(n*dim, self.k_total_padded) * 0.02)
        self.b = nn.Parameter(torch.zeros(3*self.pad)) 
        self.alpha_pre = nn.Parameter(torch.tensor(0.01)); self.alpha_post = nn.Parameter(torch.tensor(0.01)); self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.inner = inner

    def forward(self, x):
        B, N, C = x.shape
        x_flat = x.view(B, -1)
        h_pre, h_post, h_res = MHCCoeffsFunc.apply(x_flat, self.phi, self.b, self.alpha_pre, self.alpha_post, self.alpha_res, self.n)
        z = MHCApplyPreFunc.apply(x, h_pre)
        y = self.inner(z.to(x.dtype))
        out = MHCApplyPostResFunc.apply(x, y, h_post, h_res)
        return out

def test_correctness():
    print("\n--- Running Correctness Tests (Safe Pointers) ---")
    torch.manual_seed(42)
    B, C, N = 16, 128, 4
    inner = nn.Linear(C, C).cuda()
    model = MHC(inner, C, n=N).cuda().to(torch.bfloat16)
    
    class NaiveMHC(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x):
            x_flat = x.view(B, -1).float()
            r = x_flat.norm(dim=1, keepdim=True) / math.sqrt(N*C)
            r = torch.clamp(r, min=EPS_R)
            phi_f = model.phi.float()
            y_res = (x_flat @ phi_f[:, 0:16]) / r; y_pre = (x_flat @ phi_f[:, 16:32]) / r; y_post = (x_flat @ phi_f[:, 32:48]) / r
            y_res = y_res[:, :N*N] * model.alpha_res + model.b[:N*N]
            y_pre = y_pre[:, :N] * model.alpha_pre + model.b[16:16+N]
            y_post = y_post[:, :N] * model.alpha_post + model.b[32:32+N]
            h_pre = torch.sigmoid(y_pre); h_post = 2 * torch.sigmoid(y_post)
            M = torch.exp(y_res.view(B, N, N))
            for _ in range(_get_sinkhorn_iters()):
                M = M / (M.sum(dim=2, keepdim=True) + 1e-12); M = M / (M.sum(dim=1, keepdim=True) + 1e-12)
            h_res = M.view(B, N*N)
            z = (h_pre.unsqueeze(2) * x.float()).sum(dim=1); y_inner = inner(z.to(x.dtype)).float()
            return torch.einsum('bij,bjc->bic', h_res.view(B, N, N), x.float()) + h_post.unsqueeze(2) * y_inner.unsqueeze(1)

    naive = NaiveMHC()
    x = torch.randn(B, N, C, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    out_t = model(x); out_n = naive(x)
    print(f"Diff: {(out_t.float()-out_n.float()).abs().max().item():.6f}")
    assert torch.allclose(out_t.float(), out_n.float(), atol=0.05)
    print("✅ Forward Passed")
    if bool(int(os.environ.get("MHC_DEBUG", "0"))):
        print("🔎 Debug: entering debug block (pre-sync)...", flush=True)
        torch.cuda.synchronize()
        print("🔎 Debug: testing MHCCoeffsFunc backward...", flush=True)
        x_dbg = x.detach().clone().requires_grad_(True)
        h_pre_dbg, h_post_dbg, h_res_dbg = MHCCoeffsFunc.apply(
            x_dbg.view(B, -1), model.phi, model.b, model.alpha_pre, model.alpha_post, model.alpha_res, N
        )
        torch.autograd.backward([h_pre_dbg, h_post_dbg, h_res_dbg],
                                [torch.randn_like(h_pre_dbg), torch.randn_like(h_post_dbg), torch.randn_like(h_res_dbg)])
        torch.cuda.synchronize()
        print("✅ Debug: MHCCoeffsFunc backward finished", flush=True)

        print("🔎 Debug: testing MHCApplyPreFunc backward...", flush=True)
        x_dbg = x.detach().clone().requires_grad_(True)
        h_pre_dbg, h_post_dbg, h_res_dbg = MHCCoeffsFunc.apply(
            x_dbg.view(B, -1), model.phi, model.b, model.alpha_pre, model.alpha_post, model.alpha_res, N
        )
        z_dbg = MHCApplyPreFunc.apply(x_dbg, h_pre_dbg)
        torch.autograd.backward([z_dbg], [torch.randn_like(z_dbg)])
        torch.cuda.synchronize()
        print("✅ Debug: MHCApplyPreFunc backward finished", flush=True)

        print("🔎 Debug: testing MHCApplyPostResFunc backward...", flush=True)
        x_dbg = x.detach().clone().requires_grad_(True)
        h_pre_dbg, h_post_dbg, h_res_dbg = MHCCoeffsFunc.apply(
            x_dbg.view(B, -1), model.phi, model.b, model.alpha_pre, model.alpha_post, model.alpha_res, N
        )
        z_dbg = MHCApplyPreFunc.apply(x_dbg, h_pre_dbg)
        y_dbg = model.inner(z_dbg.to(x_dbg.dtype))
        out_dbg = MHCApplyPostResFunc.apply(x_dbg, y_dbg, h_post_dbg, h_res_dbg)
        torch.autograd.backward([out_dbg], [torch.randn_like(out_dbg)])
        torch.cuda.synchronize()
        print("✅ Debug: MHCApplyPostResFunc backward finished", flush=True)

    out_t.backward(torch.randn_like(out_t))
    print("✅ Backward Runs")

if __name__ == "__main__":
    if torch.cuda.is_available(): test_correctness()

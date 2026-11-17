"""Fused CUDA kernels for the BF16 engine."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize(values, Weight, cols, mask, WIDTH: tl.constexpr, EPS: tl.constexpr):
    mean_square = tl.sum(values * values, axis=0) / WIDTH
    normalized = values * tl.rsqrt(mean_square + EPS)
    # Qwen3 rounds the normalized activations to BF16 before scaling.
    normalized = tl.cast(normalized, tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    return normalized * tl.load(Weight + cols, mask, other=0).to(tl.float32)


@triton.jit
def _rms_norm_kernel(X, Weight, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    # One program handles one row.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    index = row * WIDTH + cols
    values = tl.load(X + index, mask, other=0).to(tl.float32)
    tl.store(Out + index, _normalize(values, Weight, cols, mask, WIDTH, EPS), mask)


def fused_rms_norm(x, weight, eps):
    width = x.shape[-1]
    out = torch.empty_like(x)
    _rms_norm_kernel[(x.numel() // width,)](
        x, weight, out, width, triton.next_power_of_2(width), eps,
        num_warps=1 if width <= 128 else 4, enable_fp_fusion=False,
    )
    return out


@triton.jit
def _residual_rms_norm_kernel(
    Hidden, Update, Weight, Sum, Norm, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    residual = tl.load(Hidden + cols, mask, other=0).to(tl.float32)
    update = tl.load(Update + cols, mask, other=0).to(tl.float32)
    values = tl.cast(residual + update, tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Sum + cols, values, mask)
    normalized = _normalize(values.to(tl.float32), Weight, cols, mask, WIDTH, EPS)
    tl.store(Norm + cols, normalized, mask)


def fused_residual_rms_norm(hidden, update, weight, eps):
    """Add and normalize the hidden state for one decode token."""
    width = hidden.shape[-1]
    residual, normalized = torch.empty_like(hidden), torch.empty_like(hidden)
    _residual_rms_norm_kernel[(1,)](
        hidden, update, weight, residual, normalized, width, triton.next_power_of_2(width), eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return residual, normalized


@triton.jit
def _qk_rms_norm_kernel(
    Q, K, QWeight, KWeight, QOut, KOut,
    Q_HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    head = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    if head < Q_HEADS:
        values = tl.load(Q + head * WIDTH + cols, mask, other=0).to(tl.float32)
        result = _normalize(values, QWeight, cols, mask, WIDTH, EPS)
        tl.store(QOut + head * WIDTH + cols, result, mask)
    else:
        index = (head - Q_HEADS) * WIDTH + cols
        values = tl.load(K + index, mask, other=0).to(tl.float32)
        result = _normalize(values, KWeight, cols, mask, WIDTH, EPS)
        tl.store(KOut + index, result, mask)


def fused_qk_rms_norm(q, k, q_weight, k_weight, eps):
    q_heads, width = q.shape[-2:]
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    _qk_rms_norm_kernel[(q_heads + k.shape[-2],)](
        q, k, q_weight, k_weight, q_out, k_out, q_heads, width, triton.next_power_of_2(width), eps,
        num_warps=1 if width <= 128 else 4, enable_fp_fusion=False,
    )
    return q_out, k_out


@triton.jit
def _swiglu_kernel(Gate, Up, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    gate = tl.load(Gate + cols, mask, other=0).to(tl.float32)
    up = tl.load(Up + cols, mask, other=0).to(tl.float32)
    activated = tl.cast(gate / (1 + tl.exp(-gate)), tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + cols, activated.to(tl.float32) * up, mask)


def fused_swiglu(gate, up):
    width = gate.shape[-1]
    out = torch.empty_like(gate)
    block = 256
    _swiglu_kernel[(triton.cdiv(width, block),)](gate, up, out, width, block)
    return out

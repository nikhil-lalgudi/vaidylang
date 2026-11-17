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
    index = tl.program_id(0) * WIDTH + cols
    residual = tl.load(Hidden + index, mask, other=0).to(tl.float32)
    update = tl.load(Update + index, mask, other=0).to(tl.float32)
    values = tl.cast(residual + update, tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Sum + index, values, mask)
    normalized = _normalize(values.to(tl.float32), Weight, cols, mask, WIDTH, EPS)
    tl.store(Norm + index, normalized, mask)


def fused_residual_rms_norm(hidden, update, weight, eps):
    """Add and normalize each row of a contiguous hidden state."""
    width = hidden.shape[-1]
    residual, normalized = torch.empty_like(hidden), torch.empty_like(hidden)
    _residual_rms_norm_kernel[(hidden.numel() // width,)](
        hidden, update, weight, residual, normalized, width, triton.next_power_of_2(width), eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return residual, normalized


@triton.jit
def _qk_rms_norm_kernel(
    Q, K, QWeight, KWeight, QOut, KOut,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, Q_STRIDE: tl.constexpr, K_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    head, row = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    if head < Q_HEADS:
        index = head * WIDTH + cols
        values = tl.load(Q + row * Q_STRIDE + index, mask, other=0).to(tl.float32)
        result = _normalize(values, QWeight, cols, mask, WIDTH, EPS)
        tl.store(QOut + row * Q_HEADS * WIDTH + index, result, mask)
    else:
        index = (head - Q_HEADS) * WIDTH + cols
        values = tl.load(K + row * K_STRIDE + index, mask, other=0).to(tl.float32)
        result = _normalize(values, KWeight, cols, mask, WIDTH, EPS)
        tl.store(KOut + row * K_HEADS * WIDTH + index, result, mask)


def fused_qk_rms_norm(q, k, q_weight, k_weight, eps):
    rows, q_heads, width = q.shape
    k_heads = k.shape[1]
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    _qk_rms_norm_kernel[(q_heads + k_heads, rows)](
        q, k, q_weight, k_weight, q_out, k_out, q_heads, k_heads, q.stride(0), k.stride(0),
        width, triton.next_power_of_2(width), eps,
        num_warps=1 if width <= 128 else 4, enable_fp_fusion=False,
    )
    return q_out, k_out


@triton.jit
def _swiglu_kernel(
    Gate, Up, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr, GATE_STRIDE: tl.constexpr, UP_STRIDE: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    gate = tl.load(Gate + row * GATE_STRIDE + cols, mask, other=0).to(tl.float32)
    up = tl.load(Up + row * UP_STRIDE + cols, mask, other=0).to(tl.float32)
    activated = tl.cast(gate / (1 + tl.exp(-gate)), tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + row * WIDTH + cols, activated.to(tl.float32) * up, mask)


def fused_swiglu(gate, up):
    width = gate.shape[-1]
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)
    block = 256
    _swiglu_kernel[(triton.cdiv(width, block), gate.numel() // width)](
        gate, up, out, width, block, gate.stride(-2) if gate.ndim > 1 else width,
        up.stride(-2) if up.ndim > 1 else width,
    )
    return out

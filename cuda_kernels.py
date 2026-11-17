"""Fused CUDA kernels for the BF16 engine."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(X, Weight, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    # One program handles one row.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    index = row * WIDTH + cols
    values = tl.load(X + index, mask, other=0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / WIDTH
    normalized = values * tl.rsqrt(mean_square + EPS)
    # Qwen3 rounds the normalized activations to BF16 before scaling.
    normalized = tl.cast(normalized, tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    weight = tl.load(Weight + cols, mask, other=0).to(tl.float32)
    tl.store(Out + index, normalized * weight, mask)


def fused_rms_norm(x, weight, eps):
    width = x.shape[-1]
    out = torch.empty_like(x)
    _rms_norm_kernel[(x.numel() // width,)](
        x, weight, out, width, triton.next_power_of_2(width), eps,
        num_warps=1 if width <= 128 else 4, enable_fp_fusion=False,
    )
    return out

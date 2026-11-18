"""Safetensors storage and logical matrix layouts for DeepSeek V4.1."""

import numpy as np

from deepseek_ops import bf16_round
from deepseek_quant import QuantizedWeight, quantized_shapes


def quant_layout(name):
    """Return the published encoding and block geometry for a logical tensor."""
    if ".ffn.experts." in name:
        return "fp4", (1, 32)
    if name.endswith(".engram.embed.weight"):
        return "fp8", (1, 32)
    if name.endswith((
        ".attn.wq_a.weight", ".attn.wq_b.weight", ".attn.wkv.weight", ".attn.wo_a.weight",
        ".attn.wo_b.weight", ".attn.indexer.wq_b.weight", ".engram.wkv.weight", ".main_proj.weight",
    )) or ".ffn.shared_experts." in name:
        return "fp8", (32, 32)
    return None


def scale_name(name):
    return name.removesuffix(".weight") + ".scale"


def read_deepseek_weights(checkpoint, config, shapes):
    layouts = {name: quant_layout(name) for name in shapes} if config.quantization_config else {}
    names = [*shapes, *(scale_name(name) for name, layout in layouts.items() if layout)]
    stored = {name: (code, data) for name, code, data in checkpoint.read_storage(names)}
    weights = {}
    for name, shape in shapes.items():
        code, data = stored[name]
        layout = layouts.get(name)
        if not layout:
            if data.shape != shape or code not in ("F16", "BF16", "F32"):
                raise ValueError(f"Tensor {name} has {code} {data.shape}; expected float {shape}")
            weights[name] = (
                (data.astype(np.uint32) << 16).view(np.float32)
                if code == "BF16" else data.astype(np.float32, copy=False)
            )
            continue
        encoding, block = layout
        w_shape, s_shape = quantized_shapes(shape, block, encoding)
        expected_code = "I8" if encoding == "fp4" else "F8_E4M3"
        if code != expected_code or data.shape != w_shape:
            raise ValueError(f"Tensor {name} has {code} {data.shape}; expected {expected_code} {w_shape} for logical {shape}")
        scale = scale_name(name)
        scales_code, scales = stored[scale]
        if scales_code != "F8_E8M0" or scales.shape != s_shape:
            raise ValueError(f"Tensor {scale} has {scales_code} {scales.shape}; expected F8_E8M0 {s_shape} for blocks {block}")
        converted = name.endswith((".attn.wo_a.weight", ".engram.embed.weight"))
        try:
            quant = QuantizedWeight(
                data, scales, shape, block_shape=block, encoding=encoding,
                activation_group_size=None if converted else 32,
            )
        except ValueError as error:
            raise ValueError(f"Invalid quantized weight {name} or scale {scale}: {error}") from error
        weights[name] = bf16_round(quant.weight) if converted else quant
    return weights

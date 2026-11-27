"""NumPy reference for DeepSeek V4.1 microscaled storage and arithmetic.

The encoding and scale rules are derived from DeepSeek's MIT-licensed
inference/convert.py and inference/kernel.py at 314113f8a7dadb388215396ba5fbee2c610a04d6.
"""

from functools import cache
from operator import index

import numpy as np


FP4_MAGNITUDES = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float32)


def _bytes(raw):
    result = np.asarray(raw)
    if result.dtype not in (np.dtype("i1"), np.dtype("u1")):
        raise ValueError("Expected a signed or unsigned byte array")
    return result.view(np.uint8)


def decode_fp4(raw):
    """Unpack E2M1 bytes along the last axis, low nibble first."""
    packed = np.atleast_1d(_bytes(raw))
    nibbles = np.stack([packed & 15, packed >> 4], axis=-1)
    values = np.copysign(FP4_MAGNITUDES[nibbles & 7], 8 - 2 * (nibbles & 8).astype(np.int16))
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def decode_fp8(raw):
    """Decode E4M3FN, including the finite range beyond IEEE's reserved exponent."""
    bits = _bytes(raw)
    exponent = (bits >> 3 & 15).astype(np.int32)
    mantissa = (bits & 7).astype(np.float32)
    significand = np.where(exponent == 0, mantissa / 8, 1 + mantissa / 8)
    magnitude = np.ldexp(significand, np.maximum(exponent, 1) - 7)
    magnitude = np.where((bits & 127) == 127, np.nan, magnitude)
    return np.copysign(magnitude, 64 - (bits & 128).astype(np.int16))


def decode_e8m0(raw):
    """Decode unsigned E8M0, where byte 127 means one and 255 encodes NaN."""
    bits = _bytes(raw)
    exponents = np.minimum(bits, 254).astype(np.int32) - 127
    return np.where(bits == 255, np.nan, np.ldexp(np.float32(1), exponents))


def quantized_shapes(logical_shape, block_shape, encoding):
    """Return stored weight and scale shapes for a logical [out, in] matrix."""
    rows, columns = logical_shape
    block_rows, block_columns = block_shape
    if min(rows, columns, block_rows, block_columns) < 1:
        raise ValueError("Matrix dimensions and scale blocks must be positive")
    if encoding not in ("fp4", "fp8"):
        raise ValueError(f"Unsupported weight encoding {encoding!r}")
    storage = (rows, (columns + 1) // 2 if encoding == "fp4" else columns)
    scales = ((rows + block_rows - 1) // block_rows, (columns + block_columns - 1) // block_columns)
    return storage, scales


def dequantize(raw, scales, logical_shape, *, block_shape, encoding):
    """Expand E8M0 scale tiles and return a small FP32 logical weight matrix."""
    raw = _bytes(raw)
    scales = _bytes(scales)
    storage_shape, scale_shape = quantized_shapes(logical_shape, block_shape, encoding)
    if raw.shape != storage_shape:
        raise ValueError(f"{encoding} storage shape {raw.shape}, expected {storage_shape} for logical {logical_shape}")
    if scales.shape != scale_shape:
        raise ValueError(f"{encoding} scale shape {scales.shape}, expected {scale_shape} for blocks {block_shape}")

    rows, columns = logical_shape
    block_rows, block_columns = block_shape
    values = decode_fp4(raw)[:, :columns] if encoding == "fp4" else decode_fp8(raw)
    scales = decode_e8m0(scales)
    if not np.isfinite(scales).all():
        raise ValueError("Found a nonfinite E8M0 scale")
    return values * scales[np.arange(rows)[:, None] // block_rows, np.arange(columns) // block_columns]


_FP8_MAGNITUDES = decode_fp8(np.arange(127, dtype=np.uint8))


def _round_values(x, table):
    """Nearest available finite value, ties to even encoding, with saturation."""
    magnitude = np.abs(x)
    high = np.minimum(np.searchsorted(table, magnitude, side="left"), table.size - 1)
    low = np.maximum(high - 1, 0)
    towards_low = magnitude - table[low]
    towards_high = table[high] - magnitude
    use_low = (towards_low < towards_high) | ((towards_low == towards_high) & (low % 2 == 0))
    return np.copysign(table[np.where(use_low, low, high)], x)


@cache
def _native_quantizer():
    from ctypes import c_bool, c_size_t, c_void_p

    from native_cpu import check, load_library

    library = load_library()
    function = getattr(library, "vaidy_quantize_activation", None)
    if function is None:
        return None
    function.argtypes = [c_void_p, c_void_p, c_size_t, c_size_t, c_size_t, c_bool, c_bool]
    function.restype = c_bool

    def quantize(x, group_size, fp4, fp8_scale):
        values = np.ascontiguousarray(x)
        result = np.empty(x.shape, dtype=np.float32)
        width = x.shape[-1]
        check(library, function(
            values.ctypes.data, result.ctypes.data, x.size // width, width,
            min(group_size, width), fp4, fp8_scale,
        ))
        return result

    return quantize


def quantize_activation(x, *, encoding, group_size, scale_encoding="e8m0"):
    """Fake-quantize per row and input block, returning dequantized FP32.

    Full blocks use the reference scale selection and round-to-nearest-even.
    A final partial group follows the same rule over its existing elements.
    """
    x = np.asarray(x, dtype=np.float32)
    group_size = index(group_size)
    if x.ndim == 0 or not x.shape[-1] or group_size < 1:
        raise ValueError("Activation must have a nonempty last axis and positive group size")
    if encoding not in ("fp4", "fp8"):
        raise ValueError(f"Unsupported activation encoding {encoding!r}")
    if scale_encoding not in ("e8m0", "fp8") or (scale_encoding == "fp8" and encoding != "fp4"):
        raise ValueError(f"Unsupported scale encoding {scale_encoding!r} for {encoding}")

    native = _native_quantizer()
    if native is not None:
        return native(x, group_size, encoding == "fp4", scale_encoding == "fp8")
    if not np.isfinite(x).all():
        raise ValueError("Found a nonfinite activation")

    table = FP4_MAGNITUDES if encoding == "fp4" else _FP8_MAGNITUDES
    result = np.empty_like(x)
    for start in range(0, x.shape[-1], group_size):
        values = x[..., start:start + group_size]
        maximum = np.max(np.abs(values), axis=-1, keepdims=True)
        if scale_encoding == "fp8":
            maximum = np.maximum(maximum, np.float32(6 * 2**-9))
            scale = _round_values(maximum / np.float32(6), _FP8_MAGNITUDES)
        else:
            minimum = 6 * 2**-126 if encoding == "fp4" else 1e-4
            relative = np.maximum(maximum, np.float32(minimum)) * np.float32(1 / float(table[-1]))
            fraction, exponent = np.frexp(relative)
            scale = np.ldexp(np.float32(1), exponent - (fraction == 0.5))
        with np.errstate(over="ignore"):
            result[..., start:start + group_size] = _round_values(values / scale, table) * scale
    return result


class QuantizedWeight:
    """Small CPU reference linear with a dequantized weight and optional FP8 input groups."""

    def __init__(self, raw, scales, logical_shape, *, block_shape, encoding, activation_group_size=None):
        if activation_group_size is not None and activation_group_size < 1:
            raise ValueError("Activation group size must be positive")
        self.shape = tuple(logical_shape)
        self.block_shape = tuple(block_shape)
        self.encoding = encoding
        self.activation_group_size = activation_group_size
        self.weight = dequantize(raw, scales, self.shape, block_shape=self.block_shape, encoding=encoding)

    def linear(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0 or x.shape[-1] != self.shape[1]:
            raise ValueError(f"Expected input width {self.shape[1]}, received shape {x.shape}")
        if self.activation_group_size is not None:
            x = quantize_activation(x, encoding="fp8", group_size=self.activation_group_size)
        return x @ self.weight.T

    __call__ = linear

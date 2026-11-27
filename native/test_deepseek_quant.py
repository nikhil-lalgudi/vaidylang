"""Check native DeepSeek activation quantization against the NumPy reference."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from deepseek import DeepSeekEngine
from deepseek_fixture import write_fixture
from deepseek_quant import _native_quantizer, quantize_activation


def main():
    assert _native_quantizer() is not None
    rng = np.random.default_rng(43)
    midpoints = np.array([-.0, 0., -.25, .75, 1.0625, -1.1875, 5., 6., 448., 449.], np.float32)
    inputs = [
        midpoints, midpoints.reshape(2, -1).T,
        (rng.standard_normal((2, 3, 63)) * np.logspace(-32, 32, 63)).astype(np.float32),
        np.zeros((0, 3), dtype=np.float32),
    ]
    for encoding, scale_encoding in (("fp8", "e8m0"), ("fp4", "e8m0"), ("fp4", "fp8")):
        for group_size in (1, 3, 16, 32, 128):
            for values in inputs:
                options = dict(encoding=encoding, scale_encoding=scale_encoding, group_size=group_size)
                actual = quantize_activation(values, **options)
                with patch("deepseek_quant._native_quantizer", return_value=None):
                    expected = quantize_activation(values, **options)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    for bad in (np.nan, np.inf, -np.inf):
        with np.testing.assert_raises_regex(ValueError, "nonfinite activation"):
            quantize_activation(np.array([[0, 1, bad]]), encoding="fp8", group_size=2)

    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        write_fixture(path, quantized=True)
        engine = DeepSeekEngine.from_checkpoint(path)
        prompt = [2, 11, 3, 7, 9] * 3
        tail = [8, 13, 5, 2, 11, 11, 7, 3, 9, 9]
        capacity = len(prompt) + len(tail)
        actual, request = engine.prefill(prompt, capacity)
        with patch("deepseek_quant._native_quantizer", return_value=None):
            expected, oracle = engine.prefill(prompt, capacity)
        np.testing.assert_array_equal(actual, expected)
        for token in tail:
            actual = engine.decode(token, request)
            with patch("deepseek_quant._native_quantizer", return_value=None):
                expected = engine.decode(token, oracle)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    main()

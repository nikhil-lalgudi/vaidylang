import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


class DeepSeekQuantizationTests(unittest.TestCase):
    def test_signed_storage_low_nibble_first(self):
        from deepseek_quant import decode_fp4

        packed = np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8)
        expected = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
        np.testing.assert_array_equal(decode_fp4(packed.view(np.int8)), expected)

    def test_e4m3_and_e8m0_byte_values(self):
        from deepseek_quant import decode_e8m0, decode_fp8

        data = np.array([0, 1, 7, 8, 0x38, 0x3D, 0x7E, 0xFF, 0xB8, 0xFE], dtype=np.uint8)
        expected = [0., 2**-9, 7 * 2**-9, 2**-6, 1., 1.625, 448., np.nan, -1., -448.]
        np.testing.assert_array_equal(decode_fp8(data), expected)
        scales = np.array([0, 1, 126, 127, 128, 254, 255], dtype=np.uint8)
        np.testing.assert_array_equal(
            decode_e8m0(scales), [2**-127, 2**-126, .5, 1., 2., float(2**127), np.nan],
        )

    def test_fp4_partial_byte_and_multidimensional_scale_blocks(self):
        from deepseek_quant import dequantize

        packed = np.array([[0x32, 0x54, 0xF6], [0xA9, 0xBC, 0x0E]], dtype=np.uint8)
        scales = np.array([[127, 129], [126, 128]], dtype=np.uint8)
        result = dequantize(packed.view(np.int8), scales, (2, 5), block_shape=(1, 3), encoding="fp4")
        np.testing.assert_array_equal(result, [[1, 1.5, 2, 12, 16], [-.25, -.5, -1, -3, -8]])
        weight = np.full((3, 5), 0x38, dtype=np.uint8)
        result = dequantize(weight, scales, (3, 5), block_shape=(2, 3), encoding="fp8")
        np.testing.assert_array_equal(result, [[1, 1, 1, 4, 4], [1, 1, 1, 4, 4], [.5, .5, .5, 2, 2]])

    def test_malformed_storage_has_actionable_error(self):
        from deepseek_quant import dequantize

        weight = np.full((3, 2), 0x11, dtype=np.uint8)
        scales = np.full((3, 1), 127, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "scale shape.*3, 2"):
            dequantize(weight, scales, (3, 4), block_shape=(1, 2), encoding="fp4")
        with self.assertRaisesRegex(ValueError, "storage shape.*3, 3"):
            dequantize(weight, scales, (3, 5), block_shape=(1, 5), encoding="fp4")
        with self.assertRaisesRegex(ValueError, "signed or unsigned byte"):
            dequantize(weight, scales.astype(np.float32), (3, 4), block_shape=(1, 4), encoding="fp4")
        scales[-1, 0] = 255
        with self.assertRaisesRegex(ValueError, "nonfinite.*scale"):
            dequantize(weight, scales, (3, 4), block_shape=(1, 4), encoding="fp4")

    def test_dynamic_fp8_uses_power_of_two_and_ties_to_even(self):
        from deepseek_quant import quantize_activation

        # 448, 1.0625 and 1.1875 force scale=1 and the two midpoint directions.
        # 449 selects scale=2 for the next row. A singleton tail has its own scale.
        x = np.array([[448., 1.0625, -1.1875, .125], [449., 1.125, -1.125, 0.]], dtype=np.float32)
        expected = [[448., 1., -1.25, .125], [448., 1.125, -1.125, 0.]]
        np.testing.assert_array_equal(quantize_activation(x, encoding="fp8", group_size=3), expected)

    def test_fp4_activation_e8m0_and_e4m3_selection_and_saturation(self):
        from deepseek_quant import quantize_activation

        # E8M0 scale=1. A .75 midpoint goes to 1 (even mantissa);
        # 5 goes to 4 (even mantissa), and a separate zero tail remains zero.
        e8m0_input = np.array([6., -.75, 5., 0.], dtype=np.float32)
        np.testing.assert_array_equal(
            quantize_activation(e8m0_input, encoding="fp4", group_size=3),
            [6., -1., 4., 0.],
        )
        # With E4M3 scale the amax/6 ~= .9667 rounds down to .9375, so
        # 5.8/.9375 exceeds 6 and saturates. E8M0 would select scale=1.
        x = np.array([5.8, -1.875, 0.], dtype=np.float32)
        np.testing.assert_array_equal(
            quantize_activation(x, encoding="fp4", group_size=3, scale_encoding="fp8"),
            [5.625, -1.875, 0.],
        )
        np.testing.assert_array_equal(
            quantize_activation(x, encoding="fp4", group_size=3),
            [6., -2., 0.],
        )
        with self.assertRaisesRegex(ValueError, "nonfinite activation"):
            quantize_activation(np.array([np.inf]), encoding="fp4", group_size=3)

    def test_quantized_linears_and_weighted_expert(self):
        from deepseek_quant import QuantizedWeight

        fp8 = QuantizedWeight(
            np.array([[0x38, 0xC0], [0x48, 0x38]], dtype=np.uint8),
            np.array([[127]], dtype=np.uint8), (2, 2),
            block_shape=(2, 2), encoding="fp8", activation_group_size=2,
        )
        fp4 = QuantizedWeight(
            np.array([0x31, 0xD6], dtype=np.uint8).reshape(2, 1).view(np.int8),
            np.array([[128], [126]], dtype=np.uint8), (2, 2),
            block_shape=(1, 2), encoding="fp4", activation_group_size=2,
        )
        x = np.array([1.25, -.5], dtype=np.float32)
        np.testing.assert_array_equal(fp8(x), [2.25, 4.5])
        np.testing.assert_array_equal(fp4(x), [-.25, 3.25])
        gate, up = fp4(x), fp8(x)
        h = (gate / (1. + np.exp(-gate))) * up * .5
        expected_hidden = [
            -.25 / (1 + math.exp(.25)) * 2.25 / 2,
            3.25 / (1 + math.exp(-3.25)) * 4.5 / 2,
        ]
        np.testing.assert_allclose(h, expected_hidden, rtol=2e-6, atol=1e-6)
        # FP8 activation group reaches 7.03955, using scale=1/32.
        # With RNE the hidden vector is [-.125, 7] before w2.
        np.testing.assert_array_equal(fp4(h), [20.875, -10.75])

    def test_moe_quantizes_each_shared_expert_input_once(self):
        from deepseek import DeepSeekEngine
        from deepseek_fixture import write_fixture
        from deepseek_quant import quantize_activation

        with TemporaryDirectory() as directory:
            checkpoint = write_fixture(Path(directory), quantized=True)
            engine = DeepSeekEngine.from_checkpoint(checkpoint)
        x = np.linspace(-1, 1, engine.config.hidden_size, dtype=np.float32)
        with patch("deepseek_quant.quantize_activation", wraps=quantize_activation) as quantizer:
            output = engine.moe("layers.0", x, 0)
        shared_calls = [call for call in quantizer.call_args_list if call.args[0] is x]
        self.assertEqual(len(shared_calls), 1)
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

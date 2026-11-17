from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from engine import Qwen3Engine, read_weights, tiny_weights
from harness import compare, timed_request


def write_checkpoint_weights(path, tensors):
    header, data = {}, bytearray()
    for name, (dtype, tensor) in tensors.items():
        start = len(data)
        data.extend(tensor.tobytes())
        header[name] = {"dtype": dtype, "shape": tensor.shape, "data_offsets": [start, len(data)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = tiny_weights()
        self.engine = Qwen3Engine(self.config, self.weights)

    def test_prefill_and_decode_match_float64_golden_logits(self):
        # Values from a head-by-head float64 reference that recomputed each prefix.
        selected = [0, 1, 3, 5, 9, 17, 31, 63]
        logits, state = self.engine.prefill([3, 1, 4, 1, 5], capacity=6)
        np.testing.assert_allclose(logits[selected], [
            0.129456641, -0.069503769, 0.623245402, 2.743572766,
            0.319725926, 0.984554149, -0.344220431, -0.271592374,
        ], rtol=5e-6, atol=5e-6)
        continuation = self.engine.decode(9, state)
        np.testing.assert_allclose(continuation[selected], [
            0.939309490, 0.444885470, -0.414858688, 2.872560612,
            1.403076391, 0.825960915, -0.903862404, 1.251496508,
        ], rtol=5e-6, atol=5e-6)
        self.assertEqual(int(continuation.argmax()), 5)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            self.engine.decode(2, state)
        self.assertEqual(state.length, 6)

    def test_checkpoint_loader_reproduces_in_memory_engine(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            config = {**asdict(self.config), "model_type": "qwen3", "hidden_act": "silu"}
            (path / "config.json").write_text(json.dumps(config))
            tensors = {name: ("F32", weight.astype("<f4")) for name, weight in self.weights.items()}
            write_checkpoint_weights(path / "model.safetensors", tensors)
            loaded = Qwen3Engine.from_checkpoint(path)
            prompt = [7, 11, 13, 17]
            np.testing.assert_array_equal(loaded.embeddings, self.engine.embeddings)
            np.testing.assert_allclose(
                loaded.prefill(prompt, 4)[0], self.engine.prefill(prompt, 4)[0], atol=3e-6, rtol=3e-6,
            )

    def test_bfloat16_and_float16_load_as_float32(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            write_checkpoint_weights(path, {
                "bf16": ("BF16", np.array([0x3FC0, 0xC010], dtype="<u2")),
                "fp16": ("F16", np.array([0.5, -2.25], dtype="<f2")),
            })
            tensors = read_weights(path, {"bf16": (2,), "fp16": (2,)})
            np.testing.assert_array_equal(tensors["bf16"], np.array([1.5, -2.25], dtype=np.float32))
            np.testing.assert_array_equal(tensors["fp16"], np.array([0.5, -2.25], dtype=np.float32))
            self.assertEqual(tensors["bf16"].dtype, np.float32)

    def test_timing_excludes_prefill_from_tpot(self):
        with patch("harness.perf_counter_ns", side_effect=[0, 10_000_000, 12_000_000, 18_000_000]):
            sample = timed_request(self.engine, [3, 1, 4], new_tokens=3)
        self.assertEqual(sample["ttft_ms"], 10)
        self.assertEqual(sample["tpot_ms"], 4)
        self.assertEqual(sample["itl_ms"], [2, 6])
        self.assertEqual(sample["e2e_ms"], 18)

    def test_compare_rejects_changed_tokens_even_if_faster(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(
                baseline=path / "base.json", candidate=path / "candidate.json",
                min_tpot_improvement_pct=5.0, max_ttft_regression_pct=5.0,
            )

            def write_result(path, decode_ms, output_hash):
                path.write_text(json.dumps({"schema_version": 1, "run": {"dtype": "float32"}, "cases": [{
                    "prompt_tokens": 8, "input_sha256": "a", "ttft_ms": {"p95": 9},
                    "tpot_ms": {"p50": decode_ms}, "samples": [{"output_sha256": output_hash}],
                }]}))

            write_result(args.baseline, 5, "b")
            write_result(args.candidate, 4, "b")
            self.assertTrue(compare(args)["passed"])
            write_result(args.candidate, 4, "c")
            self.assertFalse(compare(args)["passed"])

    def test_compare_accepts_both_exact_latency_limits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(
                baseline=path / "base.json", candidate=path / "candidate.json",
                min_tpot_improvement_pct=10.0, max_ttft_regression_pct=5.0,
            )
            for target, ttft, tpot in ((args.baseline, 100, 100), (args.candidate, 105, 90)):
                target.write_text(json.dumps({
                    "schema_version": 1, "run": {"dtype": "bfloat16"},
                    "cases": [{
                        "prompt_tokens": 128, "input_sha256": "a", "ttft_ms": {"p95": ttft},
                        "tpot_ms": {"p50": tpot}, "samples": [{"output_sha256": "b"}],
                    }],
                }))
            self.assertTrue(compare(args)["passed"])


if __name__ == "__main__":
    unittest.main()

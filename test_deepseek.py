from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from harness import main


class DeepSeekPrimitiveTests(unittest.TestCase):
    def test_rotary_rotates_adjacent_pairs_and_preserves_prefix(self):
        from deepseek_ops import rotary

        x = np.array([9, 8, 1, 2, 3, 4], dtype=np.float32)
        actual = rotary(x, 4, np.array([np.pi / 2, np.pi]))
        np.testing.assert_allclose(actual, [9, 8, -2, 1, -3, -4], atol=1e-6)
        np.testing.assert_allclose(rotary(actual, 4, [np.pi / 2, np.pi], inverse=True), x, atol=1e-6)

    def test_attention_has_a_zero_value_sink_and_ignores_padding(self):
        from deepseek_ops import sparse_attention

        q = np.array([[0, 0], [0, 0]], np.float32)
        kv = np.array([[2, 4], [4, 2], [999, 999]], np.float32)
        # The two live values and sink have multipliers (1, 1, 1) and (1, 1, 2).
        actual = sparse_attention(q, kv, np.log([1, 2]), selected=[0, 1, -1])
        np.testing.assert_allclose(actual, [[2, 2], [1.5, 1.5]], atol=1e-7)

    def test_compression_pools_each_feature_with_its_own_gate(self):
        from deepseek_ops import pool_latents

        values = np.array([[1, 4], [7, 2]], dtype=np.float32)
        scores = np.log(np.array([[1, 3], [2, 1]], dtype=np.float32))
        np.testing.assert_allclose(pool_latents(values, scores), [5, 3.5], atol=1e-6)

    def test_latest_candidate_block_is_pinned_and_padding_is_dropped(self):
        from deepseek_ops import candidate_mask, top_indices

        scores = np.array([10, 20, 17, -2, -3], dtype=np.float32)
        mask = candidate_mask(scores, 2, 2)
        np.testing.assert_array_equal(mask, [True, True, False, False, True])
        np.testing.assert_array_equal(top_indices(np.where(mask, scores, -np.inf), 5), [1, 0, 4])

    def test_hyper_connection_uses_input_axis_for_residual(self):
        from deepseek_ops import hc_post, hc_pre, split_hc

        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_allclose(hc_pre(x, [0.25, 0.75]), [2.5, 3.5])
        actual = hc_post(np.array([2, 1]), x, np.array([2, 3]), np.array([[1, 2], [0, 1]]))
        np.testing.assert_allclose(actual, [[5, 4], [11, 11]])
        pre, post, comb = split_hc(np.zeros(8), [1, 1, 1], np.zeros(8), 2, 5, 1e-6)
        np.testing.assert_allclose(pre, [0.500001] * 2)
        np.testing.assert_allclose(post, [1.0] * 2)
        np.testing.assert_allclose(comb, np.full((2, 2), 0.5), atol=1e-6)

    def test_gate_bias_selects_but_route_uses_unbiased_scores(self):
        from deepseek_ops import route

        ids, weights = route(np.array([0, 1, 2]), np.array([8, 0, 0]), 2, "sqrtsoftplus", True, 1.5)
        scores = np.sqrt(np.log1p(np.exp([0, 2])))
        np.testing.assert_array_equal(ids, [0, 2])
        np.testing.assert_allclose(weights, 1.5 * scores / sum(scores), rtol=1e-6)


class DeepSeekIntegrationTests(unittest.TestCase):
    def test_loader_generates_through_the_same_cli_for_both_architectures(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine, tiny_weights
        from test_engine import write_checkpoint_weights

        with TemporaryDirectory() as tmp:
            deepseek, qwen = Path(tmp) / "deepseek", Path(tmp) / "qwen"
            write_fixture(deepseek, quantized=False)
            qwen.mkdir()
            cfg, weights = tiny_weights()
            (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_act": "silu", **vars(cfg)}))
            write_checkpoint_weights(qwen / "model.safetensors", {n: ("F32", v) for n, v in weights.items()})
            for directory, architecture in ((qwen, "qwen3"), (deepseek, "deepseek_v41")):
                with self.subTest(architecture=architecture):
                    stdout = StringIO()
                    with redirect_stdout(stdout):
                        status = main(["generate", "--checkpoint", str(directory), "--ids", "4", "7", "--new-tokens", "16"])
                    self.assertEqual(status, 0)
                    report = json.loads(stdout.getvalue())
                    self.assertEqual(report["model_type"], architecture)
                    self.assertEqual(len(report["output_ids"]), 16)
                    engine = load_engine(directory)
                    logits, _ = engine.prefill([4, 7], 20)
                    self.assertTrue(np.isfinite(logits).all())
                    self.assertEqual(report["output_ids"][0], logits.argmax())

    def test_request_state_and_full_prefix_agree_across_boundaries(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine

        with TemporaryDirectory() as tmp:
            write_fixture(Path(tmp), quantized=False)
            engine = load_engine(tmp)
            prompt = [4, 7, 8, 9, 4, 11, 12, 8, 6]
            first, state = engine.prefill(prompt[:1], len(prompt))
            _, other = engine.prefill([10], len(prompt))
            for length, token in enumerate(prompt[1:], 2):
                actual = engine.decode(token, state)
                engine.decode(10, other)
                expected, fresh = engine.prefill(prompt[:length], length)
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
                self.assertEqual(state.length, fresh.length)
            self.assertEqual(state.capacity, len(prompt))
            self.assertEqual(set(state.compressed), {1, 3})
            self.assertEqual([len(state.compressed[i]) for i in (1, 3)], [4, 9])
            self.assertLessEqual(max(map(len, state.raw)), engine.config.sliding_window)
            self.assertNotEqual(state.mapped_tokens, other.mapped_tokens)
            np.testing.assert_array_equal(engine.prefill(prompt[:1], 1)[0], first)
            with self.assertRaisesRegex(ValueError, "capacity"):
                engine.decode(5, state)

    def test_quantized_shards_generate_and_identify_all_weight_bytes(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine
        from harness import checkpoint_identity

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_fixture(path)
            identity = checkpoint_identity(path)
            self.assertIn("/sharded-v1-", identity)
            engine = load_engine(path)
            prompt = [4, 7, 4, 11, 16, 9]
            logits, state = engine.prefill(prompt, 23)
            output_ids = []
            for _ in range(16):
                self.assertTrue(np.isfinite(logits).all())
                token = int(np.argmax(logits))
                output_ids.append(token)
                logits = engine.decode(token, state)
                full = engine.prefill(prompt + output_ids, 22)[0]
                np.testing.assert_allclose(logits, full, atol=2e-6, rtol=2e-6)
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(["generate", "--checkpoint", tmp, "--ids", *map(str, prompt)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue())["output_ids"], output_ids)
            with (path / "engram_token_map.json").open("a") as sidecar:
                sidecar.write("\n")
            self.assertNotEqual(checkpoint_identity(path), identity)

    def test_quantized_checkpoint_reports_missing_scale_and_bad_geometry(self):
        from deepseek_fixture import write_fixture, write_safetensors
        from engine import load_engine
        from checkpoint import Checkpoint

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_fixture(path)
            index_file = path / "model.safetensors.index.json"
            index = json.loads(index_file.read_text())
            scale = "layers.0.attn.wq_a.scale"
            del index["weight_map"][scale]
            index_file.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, r"layers\.0\.attn\.wq_a\.scale"):
                load_engine(path)

            write_fixture(path)
            checkpoint = Checkpoint.open(path)
            shard = checkpoint.shard_for(scale)
            names = [n for n in checkpoint.weight_map if checkpoint.shard_for(n) == shard]
            tensors = {n: (code, data.copy()) for n, code, data in checkpoint.read_storage(names)}
            tensors[scale] = ("F8_E8M0", np.zeros((2, 2), np.uint8))
            write_safetensors(shard, tensors)
            with self.assertRaisesRegex(ValueError, r"layers\.0\.attn\.wq_a\.scale"):
                load_engine(path)

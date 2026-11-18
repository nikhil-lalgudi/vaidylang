"""Independent scalar calculations for the published DSpark draft paths."""

from dataclasses import replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from deepseek_draft import DSparkDraft


def fp64_draft(engine, target_features, seed_token):
    """Independently translate forward_spec with whole-window attention and FP64 arithmetic."""
    from tools.deepseek_oracle import _connection, _moe, _norm, _normalized_exp, _rotate

    cfg = engine.config
    eps = cfg.rms_norm_eps
    width = cfg.dspark_block_size
    count = len(target_features)
    start = max(0, count - cfg.sliding_window)
    weights = {name: array.astype(np.float64) for name, array in engine.weights.items()}
    main = _norm(np.asarray(target_features[start:]) @ weights["mtp.0.main_proj.weight"].T,
                 weights["mtp.0.main_norm.weight"], eps)
    embed = weights["embed.weight"]
    noisy_ids = [seed_token, *([cfg.dspark_noise_token_id] * (width - 1))]
    stream = np.tile(embed[noisy_ids, None, :], (1, cfg.hc_mult, 1))
    pre = np.tile(np.eye(1, cfg.hc_mult), (width, 1))
    positions = np.arange(count, count + width)
    n_experts, topk = cfg.expert_counts(cfg.num_hidden_layers)
    routing_config = replace(cfg, n_routed_experts=n_experts, num_experts_per_tok=topk)
    for stage in range(cfg.num_nextn_predict_layers):
        prefix = f"mtp.{stage}."
        w = {name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)}
        attn_pre, post, comb = _connection(stream, w, "attn", cfg)
        qinput = _norm(np.einsum("sh,shd->sd", pre, stream), w["attn_norm.weight"], eps)
        qlatent = _norm(qinput @ w["attn.wq_a.weight"].T, w["attn.q_norm.weight"], eps)
        query = (qlatent @ w["attn.wq_b.weight"].T).reshape(width, cfg.num_attention_heads, cfg.head_dim)
        query = _rotate(query, positions, cfg, False)
        keyvalue = np.concatenate((main, qinput)) @ w["attn.wkv.weight"].T
        keyvalue = _norm(keyvalue, w["attn.kv_norm.weight"], eps)
        keyvalue = _rotate(keyvalue, np.arange(start, count + width), cfg, False)
        scores = query @ keyvalue.T / math.sqrt(cfg.head_dim)
        sink = np.broadcast_to(w["attn.attn_sink"][None, :, None], (*scores.shape[:2], 1))
        probs = _normalized_exp(np.concatenate((scores, sink), axis=-1))
        result = _rotate(probs[..., :-1] @ keyvalue, positions, cfg, False, inverse=True)
        groups = result.reshape(width, cfg.o_groups, -1)
        projectors = w["attn.wo_a.weight"].reshape(cfg.o_groups, cfg.o_lora_rank, groups.shape[-1])
        joined = np.concatenate([groups[:, g] @ projectors[g].T for g in range(cfg.o_groups)], axis=1)
        attention = joined @ w["attn.wo_b.weight"].T
        stream = np.einsum("sij,sid->sjd", comb, stream) + np.einsum("sh,sd->shd", post, attention)
        pre, post, comb = _connection(stream, w, "ffn", cfg)
        expert_in = _norm(np.einsum("sh,shd->sd", attn_pre, stream), w["ffn_norm.weight"], eps)
        expert_out = _moe(expert_in, w, routing_config, lambda *args, **kwargs: None)
        stream = np.einsum("sij,sid->sjd", comb, stream) + np.einsum("sh,sd->shd", post, expert_out)

    final = np.einsum("sh,shd->sd", pre, stream)
    logits = _norm(final, w["norm.weight"], eps) @ weights["head.weight"].T
    selected, confidence = [], []
    for row, hidden in zip(logits, final, strict=True):
        markov = w["markov_head.embed.weight"][seed_token]
        row += w["markov_head.head.weight"] @ markov
        seed_token = int(row.argmax())
        selected.append(seed_token)
        confidence.append((w["confidence_head.proj.weight"] @ np.r_[hidden, markov]).item())
    return selected, logits, confidence


class ArithmeticEngine:
    """Two-channel weights chosen so individual attention/head terms are inspectable."""

    config = SimpleNamespace(
        num_nextn_predict_layers=1, dspark_block_size=2, dspark_noise_token_id=3,
        dspark_target_layer_ids=(0,), dspark_markov_rank=1, hidden_size=2,
        vocab_size=4, max_position_embeddings=32, sliding_window=2, hc_mult=2,
        num_attention_heads=1, head_dim=2, qk_rope_head_dim=2, o_groups=1,
        o_lora_rank=2, rms_norm_eps=1e-6, rope_theta=10000.0,
    )

    def __init__(self):
        self.weights = {
            f"mtp.0.attn.{name}.weight": np.eye(2, dtype=np.float32)
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
        }
        self.weights.update({
            "mtp.0.attn.q_norm.weight": np.ones(2, dtype=np.float32),
            "mtp.0.attn.kv_norm.weight": np.ones(2, dtype=np.float32),
            "mtp.0.attn.attn_sink": np.array([0.4], dtype=np.float32),
            "mtp.0.norm.weight": np.ones(2, dtype=np.float32),
            "head.weight": np.array([[0, 1], [1, 0], [-1, -1], [1, 1]], dtype=np.float32),
            "mtp.0.markov_head.embed.weight": np.array([[1], [-1], [0.5], [2]], dtype=np.float32),
            "mtp.0.markov_head.head.weight": np.array([[0], [2], [1], [-1]], dtype=np.float32),
            "mtp.0.confidence_head.proj.weight": np.array([[2, 3, -4]], dtype=np.float32),
        })

    def linear(self, name, x):
        return x @ self.weights[name].T

    def dense_weight(self, name):
        return self.weights[name]

    @staticmethod
    def activation(x, **kwargs):
        return x


class DSparkArithmeticTests(unittest.TestCase):
    def test_attention_uses_target_window_and_all_noise_slots(self):
        engine = ArithmeticEngine()
        draft = DSparkDraft(engine)
        hidden = np.array([[0.3, -0.7], [-0.1, 0.5]], dtype=np.float32)
        main = np.array([[-0.4, 0.5], [0.8, 0.1]], dtype=np.float32)
        main_positions, draft_positions = (7, 8), (9, 10)

        def normalize(vec):
            scale = math.sqrt(sum(float(x)**2 for x in vec) / 2 + 1e-6)
            return [float(x) / scale for x in vec]

        def rotate(vec, pos):
            a, b = vec
            return [math.cos(pos) * a - math.sin(pos) * b, math.sin(pos) * a + math.cos(pos) * b]

        keys = [
            rotate(normalize(v), p)
            for seq, positions in ((main, main_positions), (hidden, draft_positions))
            for v, p in zip(seq, positions, strict=True)
        ]
        expected = []
        for current, pos in zip(hidden, draft_positions, strict=True):
            query = rotate(normalize(current), pos)
            weights = [math.exp(sum(q * k for q, k in zip(query, key)) / math.sqrt(2)) for key in keys]
            total = math.exp(0.4) + sum(weights)
            attended = [sum(w * key[i] for w, key in zip(weights, keys)) / total for i in range(2)]
            expected.append(rotate(attended, -pos))
        actual = draft._attention("mtp.0", hidden, main, np.array(main_positions), np.array(draft_positions))
        np.testing.assert_allclose(actual, expected, atol=6e-7, rtol=6e-7)
        altered = hidden.copy()
        altered[1] = -altered[1]
        changed = draft._attention("mtp.0", altered, main, np.array(main_positions), np.array(draft_positions))
        self.assertGreater(float(np.max(np.abs(actual[0] - changed[0]))), 0.02)

    def test_markov_and_confidence_condition_on_previous_tokens(self):
        engine = ArithmeticEngine()
        draft = DSparkDraft(engine)
        hidden = np.array([[1.0, 3.0], [3.0, 2.0]], dtype=np.float32)
        output = draft._head(hidden, previous_token=1)
        # RMS denominators are sqrt(5 + eps) and sqrt(6.5 + eps). The first
        # Markov input is token 1 (embedding -1); its winning row is token 3
        # (embedding 2). Confidence uses the pre-norm hidden plus that embedding.
        a = math.sqrt(5 + 1e-6)
        b = math.sqrt(6.5 + 1e-6)
        expected = [[3/a, 1/a-2, -4/a-1, 4/a+1], [2/b, 3/b+4, -5/b+2, 5/b-2]]
        np.testing.assert_allclose(output.logits, expected, atol=6e-7, rtol=6e-7)
        np.testing.assert_array_equal(output.ids, [3, 1])
        np.testing.assert_allclose(output.confidence, [15.0, 4.0], atol=1e-7)


class DSparkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from deepseek_fixture import write_fixture
        from engine import load_engine

        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        quantized = write_fixture(Path(temporary.name) / "quantized", quantized=True)
        dense = write_fixture(Path(temporary.name) / "dense", quantized=False)
        cls.engine = load_engine(quantized)
        cls.dense_engine = load_engine(dense)

    def test_all_draft_stages_match_independent_fp64_reference(self):
        engine = self.dense_engine
        _, state = engine.prefill([3, 5, 7, 11, 13, 17], 24)
        expected_ids, expected_logits, expected_confidence = fp64_draft(engine, state.target_features, 3)
        actual = DSparkDraft(engine).forward(state, 3)
        np.testing.assert_array_equal(actual.ids, expected_ids)
        np.testing.assert_allclose(actual.logits, expected_logits, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(actual.confidence, expected_confidence, atol=1e-5, rtol=1e-5)

    def test_real_draft_reads_target_state_and_has_fixed_width(self):
        engine = self.engine
        draft = DSparkDraft(engine)
        ids = [3, 5, 7, 11, 13, 17]
        _, state = engine.prefill(ids, 24)
        first = draft.forward(state, 3)
        repeated = draft.forward(engine.prefill(ids, 24)[1], 3)
        width = engine.config.dspark_block_size
        np.testing.assert_array_equal(first.ids, repeated.ids)
        np.testing.assert_array_equal(first.logits, repeated.logits)
        np.testing.assert_array_equal(first.confidence, repeated.confidence)
        self.assertEqual(first.logits.shape, (width, engine.config.vocab_size))
        self.assertTrue(np.isfinite(first.logits).all())
        self.assertTrue(np.isfinite(first.confidence).all())
        _, changed = engine.prefill([11, 2, 1, 4, 3, 17], 24)
        self.assertGreater(float(np.max(np.abs(first.logits - draft.forward(changed, 3).logits))), 1e-3)
        for size in (2, 4, 8):
            expected_ids = first.ids[:size - 1]
            self.assertEqual(draft.propose([*ids, 3], size - 1, state), expected_ids.tolist())

    def test_real_dense_and_quantized_speculation_preserve_scalar_output(self):
        from speculative_cpu import speculative_tokens, target_tokens

        ids = [3, 5, 7, 11, 13, 17]
        for path, engine in (("dense", self.dense_engine), ("quantized", self.engine)):
            draft = DSparkDraft(engine)
            reference = list(target_tokens(engine, ids, 16))
            for size in (2, 4, 8):
                with self.subTest(path=path, size=size):
                    cycles = []
                    tokens = list(speculative_tokens(engine, ids, 16, size, draft.propose, cycles=cycles))
                    self.assertEqual(tokens, reference)
                    self.assertGreater(sum(row["proposed"] for row in cycles), 0)

    def test_failed_and_partial_stages_restore_actual_draft_and_target_state(self):
        from speculative_cpu import VerifiedState

        engine = self.engine
        draft = DSparkDraft(engine)
        ids = [3, 5, 7, 11, 13, 17]
        _, original = engine.prefill(ids, 24)
        transaction = VerifiedState(engine, original)
        expected_before = draft.forward(original, 4)

        def fail_after_writes(pos, layer, op, _):
            if pos == len(ids) + 1 and layer == -1 and op == "logits":
                raise RuntimeError("injected trace failure")

        with self.assertRaisesRegex(RuntimeError, "injected trace"):
            transaction.stage([4, 9, 2, 6], trace=fail_after_writes)
        np.testing.assert_array_equal(draft.forward(transaction.state, 4).logits, expected_before.logits)
        transaction.stage([4, 9, 2, 6])
        transaction.commit(2)
        _, expected = engine.prefill([*ids, 4, 9], 24)
        for name in vars(expected):
            actual_value, expected_value = getattr(transaction.state, name), getattr(expected, name)
            if name == "raw":
                actual_value, expected_value = [list(rows) for rows in actual_value], [list(rows) for rows in expected_value]
            np.testing.assert_equal(actual_value, expected_value, err_msg=f"request state field {name}")
        np.testing.assert_array_equal(draft.forward(transaction.state, 2).logits, draft.forward(expected, 2).logits)

    def test_cpu_benchmark_reports_real_held_out_acceptance(self):
        from tools.cpu_speculative_bench import evaluate, prompt_split

        ids = prompt_split(self.engine.config, seed=19)
        self.assertEqual(len(set(tuple(x) for group in ids.values() for x in group)), sum(map(len, ids.values())))
        result = evaluate(self.engine, ids["test"][:1], 8, dspark=True)
        self.assertTrue(result["parity"]["passed"])
        self.assertEqual(result["parity"]["compared"], 6)
        self.assertTrue(all(row["first_mismatch"] is None for row in result["cases"]))
        self.assertTrue(any(row["draft"] == "dspark" and row["proposed"] for row in result["cases"]))

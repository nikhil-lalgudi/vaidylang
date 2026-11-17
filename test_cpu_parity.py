"""CPU replay, numerical policy, and diagnostic localization regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from harness import digest
from tools.cpu_parity import (
    DEEPSEEK_ATOL,
    DEEPSEEK_RTOL,
    SnapshotTrace,
    archived_qwen3,
    compare_logits,
    deepseek_parity,
    frozen_fixture,
    verify_prefix,
)


class LogitParityTests(unittest.TestCase):
    def test_exact_tie_is_numerical_parity_but_a_distinct_choice(self):
        result = compare_logits([10.4375, 10.5, 0.0], [10.5, 10.5, 0.0], 0.5, 0.025)
        self.assertTrue(result["passed"])
        self.assertFalse(result["exact_choice_match"])
        self.assertTrue(result["actual_id_attains_reference_maximum"])
        self.assertEqual(result["actual_id_in_reference"]["rank"], 1)
        self.assertEqual(result["max_abs_error"], 0.0625)

    def test_near_tie_fails_even_with_all_values_in_tolerance(self):
        result = compare_logits([10.4375, 10.5], [10.5, 10.4375], 0.5, 0.025)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violation_count"], 0)
        self.assertEqual(result["actual_id_in_reference"]["rank"], 2)

    def test_exact_tie_requires_every_logit_to_pass(self):
        result = compare_logits([10.4375, 10.5, -1.0], [10.5, 10.5, 0.0], 0.5, 0.025)
        self.assertFalse(result["passed"])
        self.assertTrue(result["actual_id_attains_reference_maximum"])
        self.assertEqual(result["violation_count"], 1)

    def test_nonfinite_vectors_fail_before_argmax(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = compare_logits([1, value], [1, value], 1e-5, 1e-5)
                self.assertFalse(result["passed"])
                self.assertEqual(result["actual_nonfinite_indices"], [1])
                self.assertEqual(result["reference_nonfinite_indices"], [1])
                self.assertIsNone(result["actual_id"])
        one = compare_logits([1.0], [1.0], 0.0, 0.0)
        self.assertTrue(one["passed"])
        self.assertIsNone(one["reference_top_margin"])

    def test_trace_reports_first_divergence_snapshot_and_ignores_matched_masks(self):
        a, b = SnapshotTrace(), SnapshotTrace()
        for trace in (a, b):
            trace(0, 0, "masked", np.array([-np.inf, 0.1], dtype=np.float32))
            trace(0, -1, "logits", np.array([10.0, 0.0], dtype=np.float32))
        actual = np.array([0.3, 30000])
        a(1, 2, "raw_kv", actual)
        b(1, 2, "raw_kv", [0.2, 30000])
        actual[0] = 0.2
        a(1, 3, "raw_kv", [100])
        b(1, 3, "raw_kv", [99])
        failure = a.first_difference(b, atol=1e-5, rtol=1e-5)
        self.assertEqual((failure["position"], failure["layer"], failure["operation"]), (1, 2, "raw_kv"))
        self.assertEqual(failure["actual"]["values"], [0.3, 30000.0])
        self.assertEqual(failure["reference"]["values"], [0.2, 30000])
        self.assertEqual(failure["violation_count"], 1)

    def test_discrete_indices_fail_on_any_change(self):
        a, b = SnapshotTrace(), SnapshotTrace()
        a(0, 1, "index_selected", np.array([30000]))
        b(0, 1, "index_selected", np.array([30001]))
        self.assertEqual(a.first_difference(b, atol=1e-5, rtol=1e-2)["violation_count"], 1)


class ArchivedQwenParityTests(unittest.TestCase):
    def test_regenerates_archived_prompts_and_preserves_all_gpu_replay_rows(self):
        report = archived_qwen3()
        self.assertTrue(report["passed"])
        self.assertEqual((report["atol"], report["rtol"]), (0.5, 0.025))
        self.assertEqual(report["archived_block_rows"]["tiled"]["exact_logit_vectors"], 84)
        self.assertEqual(report["archived_block_rows"]["flash"]["exact_token_choices"], 80)
        self.assertEqual([c["prompt_tokens"] for c in report["cases"]], [128, 1024, 4096])
        self.assertTrue(all(c["prompt_sha256_reproduced"] for c in report["cases"]))
        self.assertEqual(
            [[s["step"] for s in c["steps"]] for c in report["cases"]],
            [[0, 16, 37], [91, 100], [15, 21]],
        )
        steps = [step for case in report["cases"] for step in case["steps"]]
        self.assertTrue(steps[0]["prefix_sha256_reproduced"])
        self.assertTrue(all(step["prefix_sha256_reproduced"] is None for step in steps[1:]))
        self.assertTrue(all(step["gpu_replay_pending"] for step in steps))
        self.assertEqual(steps[1]["archived_cuda_vs_hf"]["hf_bf16"]["actual_id_in_reference"]["rank"], 2)
        self.assertEqual(steps[3]["archived_cuda_vs_hf"]["hf_fp32"]["violation_count"], 495)

    def test_validates_token_ids_and_recomputes_their_prefix_hash(self):
        prompt, generated = [3, 4], [1, 5, 9]
        original = digest(prompt + generated[:2])
        exact = verify_prefix(prompt, generated, 2, original)
        self.assertTrue(exact["prefix_sha256_reproduced"])
        self.assertEqual(exact["prefix_ids"], [3, 4, 1, 5])
        self.assertFalse(verify_prefix(prompt, [1, 6], 2, original)["prefix_sha256_reproduced"])
        self.assertIsNone(verify_prefix(prompt, [1], 2, original)["prefix_sha256_reproduced"])
        with self.assertRaisesRegex(ValueError, "token IDs"):
            verify_prefix(prompt, [1, 5.5], 2, original)


class DeepSeekParityTests(unittest.TestCase):
    def test_full_scalar_blocks_and_independent_oracle_cross_boundaries(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine

        with TemporaryDirectory() as temp:
            path = Path(temp)
            write_fixture(path, quantized=False)
            engine = load_engine(path)
            prompt = [2, 11, 3, 7, 9]
            report = deepseek_parity(engine, prompt, 16, block_sizes=(2, 4, 8), frozen_ids=frozen_fixture(False))
        self.assertTrue(report["passed"], report["first_differences"])
        self.assertEqual((report["atol"], report["rtol"]), (DEEPSEEK_ATOL, DEEPSEEK_RTOL))
        self.assertEqual(len(report["output_ids"]), 16)
        self.assertEqual(report["output_sha256"], digest(report["output_ids"]))
        self.assertIsNotNone(report["paths"]["independent_oracle"]["rows"][0]["actual_id_in_reference"])
        self.assertEqual(report["paths"]["independent_oracle"]["rows"][0]["prefix_ids"], prompt)
        self.assertTrue(all(block["first_logit_mismatch"] is None for block in report["paths"]["blocks"]))
        self.assertTrue(all(block["rollback_verified"] for block in report["paths"]["blocks"]))
        self.assertTrue(any(block["masked_padding_slots"] for block in report["paths"]["blocks"]))
        self.assertTrue(report["strict_frozen_sequence"]["passed"])

    def test_quantized_checkpoint_matches_independent_codebook_oracle(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine
        from tools.deepseek_oracle import QuantizedOracle

        with TemporaryDirectory() as temp:
            path = Path(temp)
            write_fixture(path, quantized=True)
            engine = load_engine(path)
            oracle = QuantizedOracle.from_checkpoint(path, engine.config, engine.token_map, engine.weights)
            report = deepseek_parity(engine, [2, 11, 3, 7, 9], 16, oracle=oracle, frozen_ids=frozen_fixture(True))
        self.assertTrue(report["passed"], report["first_differences"])
        self.assertIn("codebook", report["paths"]["independent_oracle"]["arithmetic_path"])
        self.assertTrue(report["strict_frozen_sequence"]["passed"])

    def test_injected_rotary_snapshot_error_is_localized_before_logits(self):
        from deepseek_fixture import write_fixture
        from engine import load_engine

        with TemporaryDirectory() as temp:
            engine = load_engine(write_fixture(Path(temp), quantized=False))
            original_forward = engine.forward

            def fault_in_scalar_mode(ids, state, trace=None):
                if trace is not None and state.length == 5 and len(ids) == 1:
                    def corrupt_first_rotary(position, layer, op, value):
                        if layer == 0 and op == "raw_kv":
                            value = value.copy()
                            value[-1] += 1
                        trace(position, layer, op, value)
                    return original_forward(ids, state, trace=corrupt_first_rotary)
                return original_forward(ids, state, trace=trace)

            with patch.object(engine, "forward", side_effect=fault_in_scalar_mode):
                report = deepseek_parity(engine, [2, 11, 3, 7, 9], 4, block_sizes=(2,))
        self.assertFalse(report["passed"])
        mismatch = report["first_differences"]["full_prefix_vs_scalar"]
        self.assertEqual((mismatch["position"], mismatch["layer"], mismatch["operation"]), (5, 0, "raw_kv"))
        self.assertEqual(mismatch["worst_flat_index"], 15)


if __name__ == "__main__":
    unittest.main()

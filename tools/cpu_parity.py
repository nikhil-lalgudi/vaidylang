"""Compare reduced DeepSeek execution paths and audit archived Qwen3 inputs on CPU."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gpu_bench import FIXTURES
from harness import digest, workload
from tools import speculative_parity as gpu_replay
from tools.deepseek_oracle import DenseOracle, QuantizedOracle


# Reduced FP32 versus FP64. The longest dot in the fixture (128 elements) has a
# unit-roundoff accumulation bound below 8e-6 before normalization/mixing.
# Combined absolute and relative margins allow propagation across six layers.
DEEPSEEK_ATOL = 3e-5
DEEPSEEK_RTOL = 2e-5
QWEN3_VOCAB_SIZE = 151936
DEFAULT_PROMPT = [2, 11, 3, 7, 9]
FROZEN_DEEPSEEK = gpu_replay.ROOT / "fixtures/deepseek-cpu-parity.json"


def frozen_fixture(quantized):
    fixture = json.loads(FROZEN_DEEPSEEK.read_text())
    expected = fixture["quantized" if quantized else "dense"]
    if digest(expected["output_ids"]) != expected["output_sha256"]:
        raise ValueError("Frozen reduced model output IDs do not match their stored digest")
    return expected["output_ids"]


def compare_logits(actual, reference, atol, rtol):
    """Apply strict oracle maximization and all-logit tolerance independently."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.ndim != 1 or actual.shape != reference.shape or not actual.size:
        raise ValueError("Logit vectors must be nonempty, one-dimensional, and equal in shape")
    a_bad = np.flatnonzero(~np.isfinite(actual)).tolist()
    b_bad = np.flatnonzero(~np.isfinite(reference)).tolist()
    if a_bad or b_bad:
        return {
            "passed": False,
            "actual_id": None,
            "reference_id": None,
            "actual_id_in_reference": None,
            "max_abs_error": None,
            "violation_count": None,
            "actual_nonfinite_indices": a_bad,
            "reference_nonfinite_indices": b_bad,
        }
    a64, b64 = actual.astype(np.float64), reference.astype(np.float64)
    if actual.size == 1:
        error = abs(float(a64[0]) - float(b64[0]))
        excess = max(0, error - atol - rtol * abs(float(b64[0])))
        result = {
            "actual_id": 0,
            "reference_id": 0,
            "exact_choice_match": True,
            "exact_logits_match": error == 0,
            "actual_id_in_reference": gpu_replay.token_score(reference, 0),
            "reference_id_in_actual": gpu_replay.token_score(actual, 0),
            "reference_top_margin": None,
            "max_abs_error": error,
            "max_abs_error_token_id": 0,
            "violation_count": int(excess > 0),
            "max_excess_over_tolerance": excess,
            "max_excess_token_id": 0 if excess else None,
        }
    else:
        result = gpu_replay.compare_logits(a64, b64, atol, rtol)
    at_maximum = bool(reference[result["actual_id"]] == reference[result["reference_id"]])
    result["actual_id_attains_reference_maximum"] = at_maximum
    result["passed"] = at_maximum and result["violation_count"] == 0
    return result


def _snapshot(value, index):
    if value is None:
        return None
    items = value.reshape(-1)

    def safe(value):
        return value.item() if np.isfinite(value) else str(value)

    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "values": [safe(x) for x in items[:16]],
        "value_at_worst_index": safe(items[index]) if index is not None else None,
    }


class SnapshotTrace:
    """Capture request-local operation results when diagnostic tracing is enabled."""

    def __init__(self):
        self.positions = defaultdict(dict)

    def __call__(self, position, layer, operation, value):
        self.positions[int(position)][int(layer), operation] = np.array(value, copy=True)

    def first_difference(self, other, *, atol, rtol):
        for position in sorted(self.positions.keys() | other.positions.keys()):
            actual_ops = self.positions.get(position, {})
            reference_ops = other.positions.get(position, {})
            for layer, operation in dict.fromkeys([*actual_ops, *reference_ops]):
                actual, reference = actual_ops.get((layer, operation)), reference_ops.get((layer, operation))
                if actual is None or reference is None:
                    reason = "missing_snapshot"
                    count, worst, index = None, None, None
                elif actual.shape != reference.shape:
                    reason = "shape_mismatch"
                    count, worst, index = None, None, None
                else:
                    both_finite = np.isfinite(actual) & np.isfinite(reference)
                    with np.errstate(invalid="ignore", over="ignore"):
                        errors = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
                        exceeded = errors > (atol + rtol * np.abs(reference.astype(np.float64)))
                    if actual.dtype.kind in "iub" or reference.dtype.kind in "iub":
                        exceeded = actual != reference
                    mismatches = np.where(both_finite, exceeded, actual != reference)
                    count = int(np.count_nonzero(mismatches))
                    if not count:
                        continue
                    reason = "value_mismatch"
                    candidates = np.where(mismatches, np.where(both_finite, errors, np.inf), -1)
                    index = int(candidates.argmax())
                    error = candidates.reshape(-1)[index]
                    worst = float(error) if np.isfinite(error) else None
                return {
                    "position": position, "layer": layer, "operation": operation,
                    "reason": reason, "violation_count": count,
                    "max_abs_error": worst, "worst_flat_index": index,
                    "actual": _snapshot(actual, index),
                    "reference": _snapshot(reference, index),
                }
        return None


def verify_prefix(prompt, generated, step, expected_sha):
    if any(type(t) is not int or not 0 <= t < QWEN3_VOCAB_SIZE for t in [*prompt, *generated]):
        raise ValueError("Expected in-vocabulary Qwen3 token IDs")
    prefix = prompt + generated[:step] if len(generated) >= step else None
    calculated = digest(prefix) if prefix is not None else None
    return {
        "prefix_ids": prefix,
        "prefix_sha256": calculated,
        "archived_prefix_sha256": expected_sha,
        "prefix_sha256_reproduced": calculated == expected_sha if calculated is not None else None,
    }


def _block_totals(archive):
    rows = [row["cuda_scalar"] for c in archive["cases"] for step in c["steps"] for row in step["block_rows"]]
    return {
        "rows": len(rows),
        "exact_token_choices": sum(row["exact_choice_match"] for row in rows),
        "exact_logit_vectors": sum(row["exact_logits_match"] for row in rows),
    }


def archived_qwen3(replay_input=None):
    """Recover archived prompts and check supplied CUDA output IDs by digest."""
    directory = gpu_replay.ROOT / "results/speculative"
    tiled = json.loads((directory / "parity-tiled.json").read_text())
    flash = json.loads((directory / "parity-flash.json").read_text())
    sglang = json.loads((gpu_replay.ROOT / "results/sglang-output-diagnostic.json").read_text())
    saved = {
        (case["workload"], case["prompt_tokens"]): case.get("scalar_output_ids", [])
        for case in (replay_input or {}).get("cases", [])
    }
    all_frozen_prompts = json.loads(FIXTURES.read_text())
    frozen_bench = json.loads((directory / "frozen-scalar.json").read_text())
    frozen_prompts = [{
        "prompt_tokens": len(ids), "prompt_sha256": digest(ids),
        "archived_prompt_sha256": case["input_sha256"],
        "prompt_sha256_reproduced": digest(ids) == case["input_sha256"],
    } for ids, case in zip(all_frozen_prompts, frozen_bench["cases"], strict=True)]
    held_out = json.loads((gpu_replay.FINAL_TILED / "held-out-bf16-128-greedy.json").read_text())
    held_out_prompts = []
    for case in held_out["cases"]:
        length = case["prompt_tokens"]
        prompt_ids, _ = workload(held_out["seed"], length, held_out["new_tokens"], QWEN3_VOCAB_SIZE)
        actual = digest(prompt_ids)
        held_out_prompts.append({
            "prompt_tokens": length, "seed": held_out["seed"],
            "prompt_sha256": actual, "archived_prompt_sha256": case["prompt_sha256"],
            "prompt_sha256_reproduced": actual == case["prompt_sha256"],
        })
    cases = []
    for (prompt, source, steps), recorded in zip(gpu_replay.replay_cases(QWEN3_VOCAB_SIZE), tiled["cases"], strict=True):
        output = saved.get((source["workload"], len(prompt)), [])
        if len(output) > tiled["scalar_greedy_output_tokens"]:
            raise ValueError("Saved CUDA continuation exceeds the archived output length")
        validation = verify_prefix(prompt, output, 0, source["recorded_prompt_sha256"])
        output_sha = digest(output) if len(output) == tiled["scalar_greedy_output_tokens"] else None
        case_rows = []
        for archive_row in recorded["steps"]:
            step = archive_row["step"]
            row = {
                "step": step,
                "prefix_tokens": archive_row["prefix_tokens"],
                **verify_prefix(prompt, output, step, archive_row["prefix_sha256"]),
                **steps[step],
                "archived_cuda_vs_hf": archive_row["scalar"],
                "gpu_replay_pending": True,
            }
            if source["workload"] == "frozen-benchmark" and step == 0:
                row["archived_sglang_comparison"] = sglang["first_divergence"]["cuda_graph"]
            case_rows.append(row)
        cases.append({
            **source,
            "prompt_tokens": len(prompt),
            "prompt_ids": prompt,
            "prompt_sha256": validation["prefix_sha256"],
            "prompt_sha256_reproduced": validation["prefix_sha256_reproduced"],
            "supplied_scalar_output_ids": output,
            "scalar_output_sha256": output_sha,
            "output_sha256_reproduced": output_sha == source["recorded_cuda_scalar_output_sha256"] if output_sha else None,
            "steps": case_rows,
        })
    passed = all(c["prompt_sha256_reproduced"] and c["output_sha256_reproduced"] is not False
                 and all(row["prefix_sha256_reproduced"] is not False for row in c["steps"]) for c in cases)
    return {
        "device": "cpu",
        "checkpoint": tiled["checkpoint"],
        "atol": tiled["atol"], "rtol": tiled["rtol"],
        "arithmetic_path": "Archived CUDA BF16 tiled scalar/verification versus HF BF16 and FP32",
        "archived_block_rows": {"tiled": _block_totals(tiled), "flash": _block_totals(flash)},
        "frozen_benchmark_prompts": frozen_prompts,
        "held_out_benchmark_prompts": held_out_prompts,
        "cases": cases,
        "passed": passed and all(c["prompt_sha256_reproduced"] for c in [*frozen_prompts, *held_out_prompts]),
        "prefix_source": "Frozen token fixtures and archived held-out RNG seed; later prefixes require original scalar output IDs.",
    }


def _rows(inputs, actual, expected, prompt_length, atol, rtol):
    return [{
        "step": position - prompt_length + 1,
        "prefix_ids": inputs[:position + 1],
        "prefix_sha256": digest(inputs[:position + 1]),
        "actual_dtype": str(a.dtype),
        "reference_dtype": str(b.dtype),
        **compare_logits(a, b, atol, rtol),
    } for position, (a, b) in enumerate(zip(actual, expected, strict=True)) if position + 1 >= prompt_length]


def _scalar(engine, prompt, new_tokens, trace):
    state = engine.new_state(len(prompt) + new_tokens - 1)
    rows = engine.forward(prompt, state, trace=trace)
    generated = [int(rows[-1].argmax())]
    continuation = []
    for _ in range(new_tokens - 1):
        result = engine.forward([generated[-1]], state, trace=trace)
        generated.append(int(result[-1].argmax()))
        continuation.append(result)
    return np.concatenate([rows, *continuation]), generated


def _block_probe(engine, prompt, inputs, scalar, size, atol, rtol):
    from speculative_cpu import VerifiedState

    trace = SnapshotTrace()
    state = engine.new_state(len(inputs))
    prefill = engine.forward(prompt, state, trace=trace)
    verifier = VerifiedState(engine, state)
    position = len(prompt)
    rollback_verified = True
    masked = 0
    parts = [prefill]
    while position < len(inputs):
        chunk = inputs[position:position + size]
        padding = size - len(chunk)
        masked += padding
        # Deliberately stage a wrong branch, causing real KV and hash work,
        # then consume the correct tokens from the restored prefix.
        alternative = [(chunk[0] + 1) % engine.config.vocab_size, *chunk[1:]]
        verifier.stage(alternative)
        verifier.commit(0)
        rollback_verified &= verifier.state.length == position
        result = verifier.stage([*chunk, *([-1] * padding)], valid_count=len(chunk), trace=trace)
        verifier.commit(len(chunk))
        parts.append(result)
        position += len(chunk)
    rows = _rows(inputs, np.concatenate(parts), scalar, len(prompt), atol, rtol)
    same_tokens = all(row.get("exact_choice_match", False) for row in rows)
    return trace, {
        "block_size": size,
        "arithmetic_path": "FP32 staged verifier versus scalar target on identical prefixes",
        "masked_padding_slots": masked,
        "rollback_verified": rollback_verified and same_tokens and all(row["passed"] for row in rows),
        "rows": rows,
        "first_logit_mismatch": next((row["step"] for row in rows if not row.get("exact_logits_match", False)), None),
    }


def deepseek_parity(engine, prompt, new_tokens, *, block_sizes=(2, 4, 8), frozen_ids=None,
                    independent=True, oracle=None, atol=DEEPSEEK_ATOL, rtol=DEEPSEEK_RTOL):
    if not prompt or new_tokens < 1 or any(size < 1 for size in block_sizes):
        raise ValueError("Use a nonempty prompt, positive generation length, and positive block sizes")
    if min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("Use finite, nonnegative parity tolerances")
    prompt = [int(token) for token in prompt]
    scalar_trace = SnapshotTrace()
    scalar, output = _scalar(engine, prompt, new_tokens, scalar_trace)
    inputs = prompt + output[:-1]
    full_trace = SnapshotTrace()
    full = engine.forward(inputs, engine.new_state(len(inputs)), trace=full_trace)
    full_rows = _rows(inputs, full, scalar, len(prompt), atol, rtol)
    first_differences = {"full_prefix_vs_scalar": full_trace.first_difference(scalar_trace, atol=atol, rtol=rtol)}
    passed = all(row["passed"] and row.get("exact_choice_match", False) for row in full_rows)
    paths = {"full_prefix": {"arithmetic_path": "FP32 full-prefix versus scalar target", "rows": full_rows}, "blocks": []}
    for size in block_sizes:
        trace, probe = _block_probe(engine, prompt, inputs, scalar, size, atol, rtol)
        first_differences[f"verifier_block_{size}_vs_scalar"] = trace.first_difference(scalar_trace, atol=atol, rtol=rtol)
        paths["blocks"].append(probe)
        passed &= probe["rollback_verified"]
    if independent:
        ref_trace = SnapshotTrace()
        oracle = oracle if oracle is not None else DenseOracle(engine.config, engine.weights, engine.token_map)
        expected = oracle.forward(inputs, trace=ref_trace)
        oracle_rows = _rows(inputs, scalar, expected, len(prompt), atol, rtol)
        first_differences["scalar_vs_independent_oracle"] = scalar_trace.first_difference(ref_trace, atol=atol, rtol=rtol)
        passed &= all(row["passed"] for row in oracle_rows)
        paths["independent_oracle"] = {
            "arithmetic_path": oracle.arithmetic_path,
            "rows": oracle_rows,
        }
    strict = None
    if frozen_ids is not None:
        first = next((i for i, (a, b) in enumerate(zip(output, frozen_ids)) if a != b), None)
        if first is None and len(output) != len(frozen_ids):
            first = min(len(output), len(frozen_ids))
        strict = {
            "frozen_output_ids": frozen_ids,
            "frozen_output_sha256": digest(frozen_ids),
            "first_mismatch": first,
            "passed": first is None and digest(output) == digest(frozen_ids),
        }
        passed &= strict["passed"]
    return {
        "device": "cpu", "numpy": np.__version__, "atol": atol, "rtol": rtol,
        "prompt_ids": prompt, "prompt_sha256": digest(prompt),
        "output_ids": output, "output_sha256": digest(output),
        "paths": paths, "strict_frozen_sequence": strict,
        "first_differences": first_differences,
        "passed": passed and not any(first_differences.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    history = subparsers.add_parser("qwen3-archive", help="Validate historical inputs and digest optional CUDA continuations")
    history.add_argument("--replay-input", type=Path, help="JSON emitted by tools.speculative_parity")
    history.add_argument("--output", type=Path)
    deepseek = subparsers.add_parser("deepseek", help="Diagnose a reduced CPU fixture")
    deepseek.add_argument("--checkpoint", type=Path)
    deepseek.add_argument("--quantized-fixture", action="store_true")
    deepseek.add_argument("--skip-independent", action="store_true", help="Run only same-engine cache and verifier comparisons")
    deepseek.add_argument("--new-tokens", type=int, default=16)
    deepseek.add_argument("--prompt", nargs="+", type=int, default=DEFAULT_PROMPT)
    deepseek.add_argument("--block-sizes", nargs="+", type=int, default=[2, 4, 8])
    deepseek.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "qwen3-archive":
        report = archived_qwen3(json.loads(args.replay_input.read_text()) if args.replay_input else None)
    else:
        from deepseek_fixture import write_fixture
        from engine import load_engine

        with TemporaryDirectory() as temp:
            checkpoint = args.checkpoint
            frozen = None
            if checkpoint is None:
                checkpoint = Path(temp)
                write_fixture(checkpoint, quantized=args.quantized_fixture)
                if args.prompt == DEFAULT_PROMPT and args.new_tokens == 16:
                    frozen = frozen_fixture(args.quantized_fixture)
            engine = load_engine(checkpoint)
            oracle = None
            if engine.config.quantization_config and not args.skip_independent:
                oracle = QuantizedOracle.from_checkpoint(checkpoint, engine.config, engine.token_map, engine.weights)
            report = deepseek_parity(
                engine, args.prompt, args.new_tokens, block_sizes=args.block_sizes,
                frozen_ids=frozen, independent=not args.skip_independent, oracle=oracle,
            )
            report["checkpoint"] = str(args.checkpoint) if args.checkpoint else (
                "reduced-quantized-default" if args.quantized_fixture else "reduced-dense-default"
            )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay recorded final-tiled BF16 disagreements on identical token prefixes."""

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from gpu_bench import FIXTURES, NEW_TOKENS, PINNED_CHECKPOINT
from harness import checkpoint_identity, digest, hf_oracle, workload


ROOT = Path(__file__).resolve().parents[1]
FINAL_TILED = ROOT / "results/cuda-tuning/final-tiled"
BLOCK_SIZES = (2, 4, 8)
TARGET_OPTIONS = {
    "fused_rms": True,
    "packed_projections": True,
    "unpacked_prefill": True,
    "dense_prefill": 4096,
    "decode_splits": 0,
    "fused_decode_pointwise": True,
    "tiled_decode": True,
}


def replay_cases(vocab_size):
    frozen = json.loads((FINAL_TILED / "frozen-128-output-diagnostic.json").read_text())
    sglang = json.loads((ROOT / "results/sglang-output-diagnostic.json").read_text())
    held_out = json.loads((FINAL_TILED / "held-out-bf16-128-greedy.json").read_text())
    frozen_steps = {}
    for step in frozen["hf_bf16_common_prefix"]:
        frozen_steps[step["step"]] = {"recorded_token_ids": {
            "cuda_scalar": step["cuda_tiled_id"],
            "cuda_packed": step["cuda_packed_id"],
            "hf_bf16": step["hf_bf16_argmax_id"],
        }}
    frozen_steps[frozen["first_packed_vs_tiled_divergence"]]["recorded_first_packed_vs_tiled_divergence"] = True
    sglang_difference = sglang["first_divergence"]["cuda_graph"]
    frozen_steps[sglang_difference["output_token_index"]]["recorded_token_ids"].update({
        "sglang": sglang_difference["sglang_token"],
        "hf_fp32": sglang_difference["hf_fp32_token"],
    })
    yield (
        json.loads(FIXTURES.read_text())[0],
        {
            "workload": "frozen-benchmark",
            "recorded_prompt_sha256": frozen["input_sha256"],
            "recorded_cuda_scalar_output_sha256": frozen["tiled_output_sha256"],
        },
        frozen_steps,
    )
    for case in held_out["cases"]:
        if not case["logit_failure_steps"]:
            continue
        prompt, _ = workload(held_out["seed"], case["prompt_tokens"], held_out["new_tokens"], vocab_size)
        yield (
            prompt.tolist(),
            {
                "workload": "held-out-greedy",
                "seed": held_out["seed"],
                "recorded_prompt_sha256": case["prompt_sha256"],
                "recorded_cuda_scalar_output_sha256": case["cuda_output_sha256"],
            },
            {step: {"recorded_hf_bf16_logit_failure": True} for step in case["logit_failure_steps"]},
        )


def token_score(logits, token):
    return {"logit": float(logits[token]), "rank": 1 + int(np.count_nonzero(logits > logits[token]))}


def compare_logits(actual, reference, atol, rtol):
    if actual.shape != reference.shape or not (np.isfinite(actual).all() and np.isfinite(reference).all()):
        raise ValueError("Logit vectors must have matching shapes and finite values")
    error = np.abs(actual - reference)
    excess = error - (atol + rtol * np.abs(reference))
    count = int(np.count_nonzero(excess > 0))
    actual_id, reference_id = int(actual.argmax()), int(reference.argmax())
    scores = token_score(reference, actual_id)
    top_two = np.partition(reference, -2)[-2:]
    max_abs_error = float(error.max())
    return {
        "actual_id": actual_id,
        "reference_id": reference_id,
        "exact_choice_match": actual_id == reference_id,
        "exact_logits_match": max_abs_error == 0,
        "actual_id_in_reference": scores,
        "reference_id_in_actual": token_score(actual, reference_id),
        "reference_top_margin": float(np.ptp(top_two)),
        "max_abs_error": max_abs_error,
        "max_abs_error_token_id": int(error.argmax()),
        "violation_count": count,
        "max_excess_over_tolerance": max(0.0, float(excess.max())),
        "max_excess_token_id": int(excess.argmax()) if count else None,
    }


def planned_probes(steps):
    probes = defaultdict(list)
    for step in steps:
        # A stage at generated prefix length start consumes row 0's input first.
        for size in BLOCK_SIZES:
            for row in range(min(size, step)):
                start = step - row - 1
                probes[start].append((step, size, row))
    return probes


def replay_case(engine, prompt, source, steps, oracles, atol, rtol, verification):
    from cuda_engine import BlockDecodeGraph, DecodeGraph

    if digest(prompt) != source["recorded_prompt_sha256"]:
        raise ValueError(f"{source['workload']} prompt hash differs at length {len(prompt)}")

    state = engine.allocate_cache(len(prompt) + NEW_TOKENS - 1)
    scalar = DecodeGraph(engine, state, prompt)
    blocks = {size: BlockDecodeGraph(engine, state, prompt, size, backend=verification) for size in BLOCK_SIZES}
    logits = scalar.prefill(prompt)
    generated = []
    scalar_logits = {}
    for step in range(NEW_TOKENS):
        generated.append(int(scalar.token.item()))
        if step in steps:
            scalar_logits[step] = logits.float().cpu().numpy()
        if step + 1 < NEW_TOKENS:
            logits = scalar.decode(scalar.token)

    output_sha256 = digest(generated)
    probes = planned_probes(steps)
    block_logits = defaultdict(list)
    replay_unchanged = {}
    logits = scalar.prefill(prompt)
    last_step = max(steps)
    for start in range(last_step + 1):
        if start in steps:
            replay_unchanged[start] = bool(np.array_equal(scalar_logits[start], logits.float().cpu().numpy()))
        for step, size, row in probes.get(start, ()):
            block = blocks[size]
            logits_for_block = block.stage(generated[start:start + size])
            logits_at_step = logits_for_block[row].float().cpu().numpy()
            block.commit(0)
            block_logits[step].append((size, row, start, logits_at_step))
        if start < last_step:
            logits = scalar.decode(generated[start])

    rows = []
    for step, metadata in sorted(steps.items()):
        prefix = prompt + generated[:step]
        expected = {name: oracle(prefix) for name, oracle in oracles.items()}
        actual = scalar_logits[step]
        entry = {
            "step": step,
            "prefix_tokens": len(prefix),
            "prefix_sha256": digest(prefix),
            **metadata,
            "scalar_replay_unchanged": replay_unchanged[step],
            "scalar": {name: compare_logits(actual, ref, atol, rtol) for name, ref in expected.items()},
            "block_rows": [],
        }
        if "recorded_token_ids" in metadata:
            entry["recorded_token_scores"] = {
                name: {source: token_score(values, token) for source, token in metadata["recorded_token_ids"].items()}
                for name, values in {"cuda_scalar": actual, **expected}.items()
            }
        for size, row, start, block_actual in sorted(block_logits[step]):
            references = {"cuda_scalar": actual, **expected}
            entry["block_rows"].append({
                "block_size": size,
                "row": row,
                "cache_prefix_tokens_before_stage": len(prompt) + start,
                **{name: compare_logits(block_actual, ref, atol, rtol) for name, ref in references.items()},
            })
        rows.append(entry)

    return {
        **source,
        "prompt_tokens": len(prompt),
        "prompt_ids": prompt,
        "scalar_output_ids": generated,
        "cuda_scalar_output_sha256": output_sha256,
        "recorded_cuda_scalar_output_reproduced": output_sha256 == source["recorded_cuda_scalar_output_sha256"],
        "steps": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification", choices=("flash", "tiled"), default="tiled",
                        help="Block verification attention backend (tiled by default)")
    args = parser.parse_args()

    import torch

    if checkpoint_identity(args.checkpoint) != PINNED_CHECKPOINT:
        raise ValueError("Checkpoint configuration or weights differ from the pinned revision")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Expose an NVIDIA CUDA GPU with native BF16")

    from cuda_engine import CudaQwen3Engine

    torch.set_float32_matmul_precision("highest")
    engine = CudaQwen3Engine.from_checkpoint(args.checkpoint, dtype=torch.bfloat16, **TARGET_OPTIONS)
    oracles = {f"hf_{name}": hf_oracle(args.checkpoint, "cuda", dtype) for name, dtype in (
        ("bf16", "bfloat16"), ("fp32", "float32"),
    )}
    held_out = json.loads((FINAL_TILED / "held-out-bf16-128-greedy.json").read_text())
    atol, rtol = held_out["atol"], held_out["rtol"]
    report = {
        "schema_version": 2,
        "checkpoint": PINNED_CHECKPOINT,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda": torch.version.cuda,
        "cuda_target_dtype": "bfloat16",
        "cuda_target_options": TARGET_OPTIONS,
        "verification_backend": args.verification,
        "scalar_greedy_output_tokens": NEW_TOKENS,
        "block_sizes": BLOCK_SIZES,
        "atol": atol,
        "rtol": rtol,
        "step_definition": "Zero-based output token index. Step 0 uses prefill and has no block row.",
        "row_definition": "Zero-based index of the consumed input token within a block. Each row predicts the next token.",
        "prefix_method": "Use the scalar CUDA greedy continuation for all models. Roll back each staged block before advancing the scalar cache.",
        "cases": [replay_case(engine, prompt, source, steps, oracles, atol, rtol, args.verification)
                  for prompt, source, steps in replay_cases(engine.config.vocab_size)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

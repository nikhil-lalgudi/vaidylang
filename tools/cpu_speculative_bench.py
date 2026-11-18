"""Measure held-out speculative acceptance on a reduced DeepSeek CPU checkpoint."""

import argparse
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np

from deepseek_draft import DSparkDraft
from harness import checkpoint_identity, digest
from speculative import NgramDraft
from speculative_bench import compare_to_baseline, measure_request, summarize_modes
from speculative_cpu import history_only, speculative_tokens, target_tokens
from speculative_learned_draft import LearnedDraft
from tools.train_speculative_draft import train_draft


BLOCK_SIZES = (2, 4, 8)
TRAINING_OPTIONS = {
    "context": 4, "embedding_size": 8, "hidden_size": 24, "epochs": 28, "confidence": 0.2,
}


def prompt_split(config, seed):
    """Use disjoint synthetic prompts for training, validation and acceptance."""
    splits = {}
    shape = {"train": (6, 9, 7, 12, 11, 8, 13, 10), "validation": (5, 7, 11), "test": (6, 10, 12)}
    for index, (split, lengths) in enumerate(shape.items()):
        rng = np.random.default_rng([seed, index])
        splits[split] = [rng.integers(config.vocab_size, size=length).tolist() for length in lengths]
    all_ids = [tuple(ids) for split in splits.values() for ids in split]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Repeated generated prompt across the CPU training and test inputs")
    return splits


def evaluate(engine, prompts, new_tokens, *, dspark, learned_path=None):
    """Measure each drafter against identical target-only continuations."""
    if not prompts or new_tokens < 2:
        raise ValueError("Supply a held-out prompt and at least two output tokens")
    drafts = {"ngram": lambda: history_only(NgramDraft())}
    if dspark:
        drafts["dspark"] = lambda: DSparkDraft(engine).propose
    if learned_path is not None:
        drafts["learned"] = lambda: history_only(LearnedDraft(learned_path))
    cases = []
    for prompt_id, ids in enumerate(prompts):
        target = measure_request(
            lambda cycles: target_tokens(engine, ids, new_tokens, cycles=cycles), new_tokens, 0
        )
        cases.append({"prompt_id": prompt_id, "input_ids": ids, "draft": "target", "block_size": 0, "samples": [target]})
        for draft_kind, factory in drafts.items():
            for size in BLOCK_SIZES:
                sample = measure_request(
                    lambda cycles: speculative_tokens(engine, ids, new_tokens, size, factory(), cycles=cycles),
                    new_tokens, size,
                )
                cases.append({
                    "prompt_id": prompt_id, "input_ids": ids, "draft": draft_kind,
                    "block_size": size, "samples": [sample],
                })
    parity = compare_to_baseline(cases)
    baseline = [case for case in cases if case["draft"] == "target"]
    summary = [{"draft": "target", **summarize_modes(baseline, [0])[0]}]
    for draft_kind in drafts:
        selected = baseline + [case for case in cases if case["draft"] == draft_kind]
        summary.extend({"draft": draft_kind, **row} for row in summarize_modes(selected, [0, *BLOCK_SIZES])[1:])

    report_cases = []
    for case in cases:
        sample, = case.pop("samples")
        cycles = sample.pop("cycles")
        report_cases.append({
            **case, **sample,
            "input_sha256": digest(case["input_ids"]),
            "proposed": sum(row["proposed"] for row in cycles),
            "accepted": sum(row["accepted"] for row in cycles),
            "max_proposal_tokens": max(row["proposed"] for row in cycles),
            "first_mismatch": sample.get("first_mismatch"),
        })
    return {"parity": parity, "summary": summary, "cases": report_cases}


def benchmark(engine, checkpoint, new_tokens, seed, *, train_learned=True):
    if new_tokens < 2 or engine.config.max_position_embeddings < 13 + new_tokens - 1:
        raise ValueError("Use at least two output tokens with space for the 13-token benchmark prompts")
    splits = prompt_split(engine.config, seed)
    with TemporaryDirectory() as temporary:
        learned_path = training = torch_version = None
        if train_learned:
            import torch

            torch_version = torch.__version__
            datasets = {
                split: [(ids, list(target_tokens(engine, ids, new_tokens))) for ids in splits[split]]
                for split in ("train", "validation")
            }
            options = {"seed": seed, "vocabulary_size": engine.config.vocab_size, **TRAINING_OPTIONS}
            weights, scores = train_draft(datasets["train"], datasets["validation"], **options)
            learned_path = Path(temporary) / "learned.npz"
            np.savez(learned_path, **weights)
            training = {
                "options": options,
                "scores": scores,
                "inputs": {
                    split: [{"input_ids": ids, "target_output_sha256": digest(out)} for ids, out in dataset]
                    for split, dataset in datasets.items()
                },
            }
        result = evaluate(engine, splits["test"], new_tokens, dspark=True, learned_path=learned_path)
    width = engine.config.dspark_block_size
    ids = splits["test"][0]
    logits, state = engine.prefill(ids, len(ids))
    seed_token = int(logits.argmax())
    draft = DSparkDraft(engine).forward(state, seed_token)
    result["dspark_probe"] = {
        "input_ids": ids,
        "input_sha256": digest(ids),
        "seed_token_id": seed_token,
        "output_ids": draft.ids.tolist(),
        "selected_logits": draft.logits[np.arange(width), draft.ids].tolist(),
        "confidence_logits": draft.confidence.tolist(),
    }
    result["run"] = {
        "checkpoint": checkpoint_identity(checkpoint),
        "model": "reduced DeepSeek V4.1 Flash text and DSpark",
        "device": "cpu",
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
        "dtype": "float32 CPU reference with checkpoint activation quantization",
        "seed": seed,
        "new_tokens": new_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "target": "scalar greedy",
        "verification": "transactional block with partial prefix replay and masked suffix slots",
        "dspark_fixed_draft_size": width,
        "dspark_max_proposals_by_verifier_width": {
            str(size): min(width, size - 1) for size in BLOCK_SIZES
        },
        "dspark_confidence": "raw logits recorded for diagnostics",
        "gpu_validation": [
            "Full checkpoint DSpark hidden-state and logit parity",
            "BF16/FP8/FP4 draft kernels and distributed draft cache",
            "Model-specific acceptance and latency on deployment hardware",
        ],
    }
    result["learned_training"] = training
    return result


def main(argv=None):
    from engine import load_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--skip-learned", action="store_true", help="Run the NumPy draft paths without Torch training")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed >= 2**64:
        parser.error("Use an unsigned 64-bit seed")
    try:
        engine = load_engine(args.checkpoint)
        if not getattr(engine.config, "dspark_block_size", 0):
            raise ValueError("Use a reduced DeepSeek checkpoint containing DSpark draft tensors")
        result = benchmark(engine, args.checkpoint, args.new_tokens, args.seed, train_learned=not args.skip_learned)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
        else:
            print(output, end="")
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0 if result["parity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

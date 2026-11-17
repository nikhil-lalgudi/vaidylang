"""Compare batch-1 speculative and target-only greedy CUDA graph decode."""

import argparse
from collections import deque
from contextlib import nullcontext
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

from gpu_bench import NEW_TOKENS, PINNED_CHECKPOINT
from harness import checkpoint_identity, host_fingerprint, percentiles, request_sample, summarize_case


PROMPTS = Path(__file__).with_name("fixtures") / "speculative-prompts.json"
BLOCK_SIZES = (0, 2, 4, 8)
TARGET_SETTINGS = {
    "fused_rms": True,
    "packed_projections": True,
    "unpacked_prefill": True,
    "dense_prefill": 4096,
    "fused_decode_pointwise": True,
    "tiled_decode": True,
}


def target_tokens(graph, ids, new_tokens, annotate):
    annotate = annotate or (lambda _: nullcontext())
    with annotate("prefill"):
        graph.prefill(ids)
        token = graph.token.item()
    yield token
    for _ in range(new_tokens - 1):
        with annotate("scalar"):
            graph.decode(graph.token)
            token = graph.token.item()
        yield token


def measure_request(generate, new_tokens, block_size):
    output, times, cycles = [], [], []
    started = perf_counter_ns()
    for token in generate(cycles):
        times.append(perf_counter_ns())
        output.append(token)
    if len(output) != new_tokens:
        raise ValueError(f"Run yielded {len(output)} tokens, expected {new_tokens}")

    sample = request_sample(started, times, output)
    if block_size == 0:
        cycles = [
            dict(draft_ms=0.0, verify_ms=0.0, scalar_ms=gap, cycle_ms=gap,
                 proposed=0, accepted=0, emitted=1, block_used=False)
            for gap in sample["itl_ms"]
        ]
    decode_tokens = new_tokens - 1
    if sum(cycle["emitted"] for cycle in cycles) != decode_tokens:
        raise ValueError("Decode cycle counts do not match the emitted token count")
    decode_ms = (times[-1] - times[0]) / 1e6
    sample.update(
        output_ids=output,
        decode_ms=decode_ms,
        decode_tokens_per_second=1000 * decode_tokens / decode_ms,
        streaming_gap_p95_ms=percentiles(sample["itl_ms"])["p95"],
        draft_ms=sum(cycle["draft_ms"] for cycle in cycles),
        verify_ms=sum(cycle["verify_ms"] for cycle in cycles),
        scalar_ms=sum(cycle["scalar_ms"] for cycle in cycles),
        emitted_tokens_per_cycle=decode_tokens / len(cycles),
        cycles=cycles,
    )
    return sample


def compare_to_baseline(cases):
    baseline = {case["prompt_id"]: case["samples"] for case in cases if case["block_size"] == 0}
    matched = compared = 0
    for case in cases:
        if case["block_size"] == 0 or case["prompt_id"] not in baseline:
            continue
        for expected, sample in zip(baseline[case["prompt_id"]], case["samples"], strict=True):
            expected_ids, actual_ids = expected["output_ids"], sample["output_ids"]
            exact = expected_ids == actual_ids
            sample["matches_baseline"] = exact
            sample.pop("first_mismatch", None)
            compared += 1
            matched += exact
            if not exact:
                index, (left, right) = next(
                    (index, pair) for index, pair in enumerate(zip(expected_ids, actual_ids, strict=True))
                    if pair[0] != pair[1]
                )
                sample["first_mismatch"] = {"output_index": index, "expected_id": left, "actual_id": right}
    if compared:
        return {"matched": matched, "compared": compared, "passed": matched == compared}
    return None


def summarize_modes(cases, block_sizes):
    modes = []
    for block_size in block_sizes:
        samples = [sample for case in cases if case["block_size"] == block_size for sample in case["samples"]]
        cycles = [cycle for sample in samples for cycle in sample["cycles"]]
        proposed = sum(cycle["proposed"] for cycle in cycles)
        accepted = sum(cycle["accepted"] for cycle in cycles)
        emitted = sum(cycle["emitted"] for cycle in cycles)
        modes.append({
            "block_size": block_size,
            "requests": len(samples),
            "ttft_ms": percentiles([sample["ttft_ms"] for sample in samples]),
            "streaming_gap_p95_ms": percentiles([gap for sample in samples for gap in sample["itl_ms"]])["p95"],
            "decode_tokens_per_second": 1000 * emitted / sum(sample["decode_ms"] for sample in samples),
            "output_tokens_per_second": 1000 * (emitted + len(samples)) / sum(sample["e2e_ms"] for sample in samples),
            "draft_ms": sum(sample["draft_ms"] for sample in samples),
            "verify_ms": sum(sample["verify_ms"] for sample in samples),
            "scalar_ms": sum(sample["scalar_ms"] for sample in samples),
            "emitted_tokens_per_cycle": emitted / len(cycles),
            "block_cycles": sum(cycle["block_used"] for cycle in cycles),
            "cycles": len(cycles),
            "proposed": proposed,
            "accepted": accepted,
            "draft_acceptance": accepted / proposed if proposed else None,
            "exact_matches": sum(sample["matches_baseline"] for sample in samples)
            if "matches_baseline" in samples[0] else None,
        })
    baseline = next((mode for mode in modes if mode["block_size"] == 0), None)
    if baseline is not None:
        for mode in modes:
            mode["decode_speedup"] = mode["decode_tokens_per_second"] / baseline["decode_tokens_per_second"]
    return modes


def load_draft_factory(specification, default):
    if specification is None:
        return default
    module, sep, attribute = specification.partition(":")
    if not (module and sep and attribute):
        raise ValueError("Draft factory must be specified as module:attribute")
    factory = getattr(import_module(module), attribute, None)
    if not callable(factory):
        raise ValueError(f"Draft factory {specification!r} is not callable")
    return factory


def benchmark(args, prompts):
    import torch

    if torch.version.cuda is None or torch.cuda.device_count() != 1:
        raise ValueError("Expose one NVIDIA CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("The GPU must support native BF16")
    if checkpoint_identity(args.checkpoint) != PINNED_CHECKPOINT:
        raise ValueError("Checkpoint configuration or weights differ from the pinned revision")

    from transformers import AutoTokenizer

    from cuda_engine import BlockDecodeGraph, CudaQwen3Engine, DecodeGraph
    from speculative import NgramDraft, speculative_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--id", f"GPU-{gpu.uuid}", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    annotate = torch.cuda.nvtx.range if args.profile else None
    draft_factory = load_draft_factory(args.draft_factory, NgramDraft)
    engine = CudaQwen3Engine.from_checkpoint(args.checkpoint, dtype=torch.bfloat16, **TARGET_SETTINGS)
    cases = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.apply_chat_template(
                prompt["messages"], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            if prompt.get("continuation"):
                ids.extend(tokenizer.encode(prompt["continuation"], add_special_tokens=False))
            state = engine.allocate_cache(len(ids) + max(args.new_tokens - 1, *args.block_sizes))
            single = DecodeGraph(engine, state, ids)
            for block_size in args.block_sizes:
                block = (
                    BlockDecodeGraph(engine, state, ids, block_size, backend=args.verification)
                    if block_size else None
                )

                def generate(cycles=None, annotate=None):
                    if block is None:
                        return target_tokens(single, ids, args.new_tokens, annotate)
                    return speculative_tokens(
                        single, block, ids, args.new_tokens, draft_factory, cycles=cycles, annotate=annotate,
                    )

                for _ in range(args.warmup):
                    deque(generate(), maxlen=0)
                samples = []
                for _ in range(args.repeats):
                    label = f"request/{prompt['id']}/block-{block_size}"
                    with (annotate(label) if annotate else nullcontext()):
                        sample = measure_request(
                            lambda cycles: generate(cycles, annotate), args.new_tokens, block_size,
                        )
                    samples.append(sample)
                case = summarize_case(ids, args.new_tokens, samples)
                case.update(prompt_id=prompt["id"], category=prompt["category"], block_size=block_size)
                cases.append(case)

    parity = compare_to_baseline(cases)
    return {
        "schema_version": 1,
        "run": {
            "checkpoint": PINNED_CHECKPOINT,
            "gpu": gpu.name,
            "gpu_uuid": str(gpu.uuid),
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "host": host_fingerprint(),
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "batch_size": 1,
            "temperature": 0,
            "ignore_eos": True,
            "enable_thinking": False,
            "timing": "token_ready_wall_time",
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "profile": args.profile,
            "block_sizes": args.block_sizes,
            "verification": args.verification,
            "draft_factory": args.draft_factory or "speculative:NgramDraft",
            "target_settings": TARGET_SETTINGS,
        },
        "parity": parity,
        "summary": summarize_modes(cases, args.block_sizes),
        "cases": cases,
    }


def main():
    prompts = {prompt["id"]: prompt for prompt in json.loads(PROMPTS.read_text())}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-sizes", type=int, nargs="+", choices=BLOCK_SIZES, default=BLOCK_SIZES,
                        help="Use 0 for target-only and 2, 4, or 8 for speculative verification")
    parser.add_argument("--verification", choices=("flash", "tiled"), default="tiled",
                        help="Attention backend for speculative verification (tiled by default)")
    parser.add_argument("--prompt", choices=prompts, action="append", help="Limit to these prompt IDs")
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true", help="Label measured requests with NVTX ranges")
    parser.add_argument("--draft-factory", help="Override the default n-gram draft with module:attribute")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    if args.new_tokens < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error("Use at least two new tokens, one repeat, and nonnegative warmups")
    if len(set(args.block_sizes)) != len(args.block_sizes):
        parser.error("Block sizes must be unique")
    selected = args.prompt or prompts
    if len(set(selected)) != len(selected):
        parser.error("Prompt IDs must be unique")
    try:
        result = benchmark(args, [prompts[name] for name in selected])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ImportError, OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")
    if result["parity"] is not None and not result["parity"]["passed"]:
        parser.exit(1, f"Greedy output mismatch. See {args.output}\n")


if __name__ == "__main__":
    main()

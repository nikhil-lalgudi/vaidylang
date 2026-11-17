"""Verify cached decode and measure in-process token latency."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np

from engine import Qwen3Engine, tiny_weights


def digest(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def host_fingerprint() -> str:
    return hashlib.sha256(platform.node().encode()).hexdigest()


def workload(seed: int, length: int, new_tokens: int, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, length])
    ids = rng.integers(0, min(vocab, 32768), size=length + new_tokens - 1)
    return ids[:length], ids[length:]


def hf_oracle(checkpoint: Path, device: str, dtype: str = "float32"):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ValueError("Install PyTorch and transformers for the HF oracle") from error
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=getattr(torch, dtype), local_files_only=True, attn_implementation="eager",
        use_safetensors=True,
    ).eval().to(device)

    @torch.inference_mode()
    def logits(ids):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().numpy()

    return logits


def verify_sequence(engine, oracle, prompt, tail, greedy, atol, rtol) -> float:
    history = prompt.tolist()
    actual, state = engine.prefill(history, len(history) + len(tail))
    max_error = 0.0
    path = "greedy" if greedy else "teacher-forced"
    for step in range(len(tail) + 1):
        expected = oracle(history)
        if not isinstance(actual, np.ndarray):
            actual = actual.float().cpu().numpy()
        if not isinstance(expected, np.ndarray):
            expected = expected.float().cpu().numpy()
        if actual.shape != expected.shape:
            raise ValueError(f"Logit shapes differ for {path} at step {step}")
        if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
            raise ValueError(f"Nonfinite logits for {path} at prompt length {len(prompt)}, step {step}")
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        expected_id = int(expected.argmax())
        actual_id = int(actual.argmax())
        if not np.allclose(actual, expected, atol=atol, rtol=rtol) or actual_id != expected_id:
            top_two = np.partition(expected, -2)[-2:]
            raise ValueError(
                f"{path} parity failed at prompt length {len(prompt)}, step {step}. "
                f"Max absolute error {error:.3g}, candidate token {actual_id}, "
                f"oracle token {expected_id}, oracle top-2 margin {float(np.ptp(top_two)):.3g}."
            )
        if step < len(tail):
            token = actual_id if greedy else int(tail[step])
            history.append(token)
            actual = engine.decode(token, state)
    return max_error


def verify(args, engine) -> dict:
    oracle_name = args.oracle or ("full" if args.tiny else "hf")
    if oracle_name == "hf":
        if args.checkpoint is None:
            raise ValueError("The HF oracle requires a checkpoint directory")
        oracle = hf_oracle(args.checkpoint, args.oracle_device, args.dtype)
    else:
        def oracle(ids):
            return engine.prefill(ids, len(ids))[0]

    cases = []
    for length in args.prompt_lengths:
        prompt, tail = workload(args.seed, length, args.new_tokens, engine.config.vocab_size)
        forced = verify_sequence(engine, oracle, prompt, tail, False, args.atol, args.rtol)
        greedy = verify_sequence(engine, oracle, prompt, tail, True, args.atol, args.rtol)
        cases.append({"prompt_tokens": length, "teacher_forced_max_abs": forced, "greedy_max_abs": greedy})
    return {
        "passed": True, "oracle": oracle_name, "steps_per_path": args.new_tokens, "cases": cases,
        "device": "cuda" if args.cuda else "cpu", "dtype": args.dtype,
        "atol": args.atol, "rtol": args.rtol, "cuda_graph": args.cuda_graph,
        "fused_rms": args.fused_rms,
    }


def timed_request(engine, ids, new_tokens: int) -> dict:
    started = perf_counter_ns()
    logits, state = engine.prefill(ids, len(ids) + new_tokens - 1)
    token = int(logits.argmax())
    times = [perf_counter_ns()]
    output = [token]
    for _ in range(new_tokens - 1):
        token = int(engine.decode(token, state).argmax())
        times.append(perf_counter_ns())
        output.append(token)
    return request_sample(started, times, output)


def request_sample(started, times, output) -> dict:
    gaps = (np.diff(times) / 1e6).tolist()
    return {
        "ttft_ms": (times[0] - started) / 1e6,
        "tpot_ms": (times[-1] - times[0]) / ((len(output) - 1) * 1e6),
        "itl_ms": gaps,
        "e2e_ms": (times[-1] - started) / 1e6,
        "output_sha256": digest(output),
    }


def percentiles(values) -> dict:
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def checkpoint_identity(directory: Path) -> str:
    with (directory / "config.json").open("rb") as config:
        config_hash = hashlib.file_digest(config, "sha256").hexdigest()
    with (directory / "model.safetensors").open("rb") as weights:
        weight_hash = hashlib.file_digest(weights, "sha256").hexdigest()
    return f"{config_hash}/{weight_hash}"


def summarize_case(ids, new_tokens, samples) -> dict:
    return {
        "prompt_tokens": len(ids),
        "input_sha256": digest(ids),
        "ttft_ms": percentiles([sample["ttft_ms"] for sample in samples]),
        "tpot_ms": percentiles([sample["tpot_ms"] for sample in samples]),
        "itl_ms": percentiles([gap for sample in samples for gap in sample["itl_ms"]]),
        "sequential_output_tps": 1000 * new_tokens * len(samples) / sum(s["e2e_ms"] for s in samples),
        "samples": samples,
    }


def benchmark(args, engine) -> dict:
    identity = "tiny-seed-7" if args.checkpoint is None else checkpoint_identity(args.checkpoint)

    cases = []
    for length in args.prompt_lengths:
        ids, _ = workload(args.seed, length, args.new_tokens, engine.config.vocab_size)
        for _ in range(args.warmup):
            timed_request(engine, ids, args.new_tokens)
        samples = [timed_request(engine, ids, args.new_tokens) for _ in range(args.repeats)]
        cases.append(summarize_case(ids, args.new_tokens, samples))

    return {
        "schema_version": 1,
        "source_tag": args.source_tag,
        "run": {
            "checkpoint": identity,
            "device": "cpu",
            "dtype": "float32",
            "machine": platform.machine(),
            "host": host_fingerprint(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "thread_env": {name: os.getenv(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )},
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "cases": cases,
    }


def compare(args) -> dict:
    baseline, candidate = [json.loads(path.read_text()) for path in (args.baseline, args.candidate)]
    if baseline["schema_version"] != candidate["schema_version"] or baseline["run"] != candidate["run"]:
        raise ValueError("Results need matching checkpoints, devices, software and benchmark settings")
    base_cases = {case["prompt_tokens"]: case for case in baseline["cases"]}
    cand_cases = {case["prompt_tokens"]: case for case in candidate["cases"]}
    if base_cases.keys() != cand_cases.keys():
        raise ValueError("Results need matching prompt lengths")

    cases = []
    for length, base in base_cases.items():
        cand = cand_cases[length]
        if base["input_sha256"] != cand["input_sha256"]:
            raise ValueError(f"Results used different token IDs at prompt length {length}")
        same_tokens = [s["output_sha256"] for s in base["samples"]] == [s["output_sha256"] for s in cand["samples"]]
        base_tpot = base["tpot_ms"]["p50"]
        base_ttft = base["ttft_ms"]["p95"]
        # Ignore floating-point noise at the inclusive percentage limits.
        tpot_improvement = round(100 * (base_tpot - cand["tpot_ms"]["p50"]) / base_tpot, 12)
        ttft_regression = round(100 * (cand["ttft_ms"]["p95"] - base_ttft) / base_ttft, 12)
        cases.append({
            "prompt_tokens": length,
            "matching_output": same_tokens,
            "p50_tpot_improvement_pct": tpot_improvement,
            "p95_ttft_regression_pct": ttft_regression,
            "passed": (
                same_tokens and tpot_improvement >= args.min_tpot_improvement_pct
                and ttft_regression <= args.max_ttft_regression_pct
            ),
        })
    return {"passed": all(case["passed"] for case in cases), "cases": cases}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "bench"):
        command = commands.add_parser(name)
        model = command.add_mutually_exclusive_group(required=True)
        model.add_argument("--tiny", action="store_true", help="Use a deterministic synthetic checkpoint")
        model.add_argument("--checkpoint", type=Path, help="Directory containing config.json and model.safetensors")
        command.add_argument("--prompt-lengths", nargs="+", type=int, default=[8, 32])
        command.add_argument("--new-tokens", type=int, default=8)
        command.add_argument("--seed", type=int, default=123)
        command.add_argument("--output", type=Path)
        if name == "verify":
            command.add_argument("--oracle", choices=("full", "hf"))
            command.add_argument("--oracle-device", default="cpu")
            command.add_argument("--cuda", action="store_true", help="Verify the CUDA engine")
            command.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
            command.add_argument("--cuda-graph", action="store_true", help="Verify CUDA Graph replay")
            command.add_argument("--fused-rms", action="store_true", help="Verify fused BF16 RMSNorm")
            command.add_argument("--atol", type=float, default=3e-4)
            command.add_argument("--rtol", type=float, default=3e-4)
        else:
            command.add_argument("--warmup", type=int, default=3)
            command.add_argument("--repeats", type=int, default=20)
            command.add_argument("--source-tag", default="working-tree")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--min-tpot-improvement-pct", type=float, default=5.0)
    comparison.add_argument("--max-ttft-regression-pct", type=float, default=5.0)
    comparison.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(args)
        else:
            if args.seed < 0 or args.new_tokens < 2 or min(args.prompt_lengths) < 1:
                raise ValueError("Use a nonnegative seed, positive prompt lengths, and at least two output tokens")
            if args.command == "bench" and (args.warmup < 0 or args.repeats < 1):
                raise ValueError("Use nonnegative warmups and at least one measured repeat")
            if args.command == "verify" and args.cuda:
                import torch
                from cuda_engine import CudaQwen3Engine, GraphQwen3Engine

                dtype = getattr(torch, args.dtype)
                if args.fused_rms and dtype != torch.bfloat16:
                    raise ValueError("Fused RMSNorm requires BF16")
                if args.tiny and dtype != torch.float32:
                    raise ValueError("Use the real checkpoint for BF16 attention verification")
                engine_type = GraphQwen3Engine if args.cuda_graph else CudaQwen3Engine
                engine = (
                    engine_type(*tiny_weights(), dtype=dtype) if args.tiny
                    else engine_type.from_checkpoint(args.checkpoint, dtype=dtype, fused_rms=args.fused_rms)
                )
            else:
                if args.command == "verify" and (args.dtype != "float32" or args.cuda_graph or args.fused_rms):
                    raise ValueError("BF16, CUDA Graphs, and fused RMSNorm require --cuda")
                engine = Qwen3Engine(*tiny_weights()) if args.tiny else Qwen3Engine.from_checkpoint(args.checkpoint)
            if max(args.prompt_lengths) + args.new_tokens - 1 > engine.config.max_position_embeddings:
                raise ValueError("Prompt and generated tokens exceed the model context limit")
            result = verify(args, engine) if args.command == "verify" else benchmark(args, engine)
        output = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n")
        print(output)
        return 1 if result.get("passed") is False else 0
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

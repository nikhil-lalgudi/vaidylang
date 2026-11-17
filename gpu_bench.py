"""Measure batch-1 BF16 prefill and decode on fixed Qwen3 token fixtures."""

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

from harness import checkpoint_identity, host_fingerprint, request_sample, summarize_case


FIXTURES = Path(__file__).with_name("fixtures") / "qwen3-0.6b-batch1.json"
PINNED_CHECKPOINT = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd/"
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
)
PROMPT_LENGTHS = (128, 1024, 4096)
NEW_TOKENS = 128
SGLANG_VERSION = "0.5.23"


def completed_sample(start, new_tokens):
    times = []
    output = []
    started = perf_counter_ns()
    for token in start():
        output.append(token.item())
        times.append(perf_counter_ns())
    if len(output) != new_tokens:
        raise ValueError(f"Run yielded {len(output)} tokens, expected {new_tokens} tokens")
    return request_sample(started, times, output)


def benchmark_cases(args, prompts, generate):
    cases = []
    for ids in prompts:
        def start():
            return generate(ids)

        for _ in range(args.warmup):
            completed_sample(start, NEW_TOKENS)
        samples = [completed_sample(start, NEW_TOKENS) for _ in range(args.repeats)]
        cases.append(summarize_case(ids, NEW_TOKENS, samples))
    return cases


def sglang_runner(checkpoint, settings):
    # Reuse the pinned static ModelRunner used by the official one-batch tool.
    # https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/benchmark/one_batch.py
    from sglang.benchmark import one_batch

    server_args = one_batch.ServerArgs(model_path=str(checkpoint.resolve()), **settings)
    server_args.resolve_once()
    one_batch._set_envs_and_config(server_args)
    port_args = one_batch.PortArgs.init_new(server_args)
    one_batch.publish(
        server_args, role="scheduler",
        ranks=one_batch.SpawnRanks(world_rank=0, gpu_id=0),
    )
    one_batch._init_process_global_configs()
    runner, _ = one_batch.load_model(server_args, port_args, gpu_id=0, tp_rank=0)
    return runner, one_batch.prepare_synthetic_inputs_for_latency_test


def sglang_tokens(runner, prepare_inputs, ids, annotate):
    with annotate("prefill"):
        runner.clear()
        requests = prepare_inputs(1, len(ids), custom_inputs=[ids])
        token, _, state = runner.extend(requests)
        yield token
    try:
        for _ in range(NEW_TOKENS - 1):
            with annotate("decode"):
                token, _ = runner.decode(token, state)
                yield token
    finally:
        runner.cleanup(state)

def cuda_tokens(engine, state, ids, annotate):
    with annotate("prefill"):
        token = engine.prefill_into(ids, state).argmax().reshape(1)
        yield token
    for _ in range(NEW_TOKENS - 1):
        with annotate("decode"):
            token = engine.decode(token, state).argmax().reshape(1)
            yield token


def benchmark_gpu(args):
    import torch

    if torch.version.cuda is None or torch.cuda.device_count() != 1:
        raise ValueError("Expose one NVIDIA CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("The GPU must support native BF16")

    if args.engine == "sglang" and version("sglang") != SGLANG_VERSION:
        raise ValueError(f"Install the pinned SGLang {SGLANG_VERSION} release")
    if checkpoint_identity(args.checkpoint) != PINNED_CHECKPOINT:
        raise ValueError("Checkpoint configuration or weights differ from the pinned revision")

    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--id", f"GPU-{gpu.uuid}", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    prompts = json.loads(FIXTURES.read_text())
    if tuple(map(len, prompts)) != PROMPT_LENGTHS:
        raise ValueError("Expected fixed input fixtures of 128, 1024, and 4096 tokens")
    annotate = torch.cuda.nvtx.range if args.profile else lambda _: nullcontext()
    if args.engine == "sglang":
        settings = {
            "device": "cuda",
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "tp_size": 1,
            "max_running_requests": 1,
            "disable_radix_cache": True,
            "chunked_prefill_size": -1,
            "speculative_algorithm": None,
            "cuda_graph_max_bs_decode": 1,
            "log_level": "warning",
        }
        runner, prepare_inputs = sglang_runner(args.checkpoint, settings)
        try:
            cases = benchmark_cases(
                args, prompts, lambda ids: sglang_tokens(runner, prepare_inputs, ids, annotate),
            )
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        tag = f"sglang-{SGLANG_VERSION}"
        implementation = {"engine": "sglang", "version": SGLANG_VERSION, "settings": settings}
    else:
        from cuda_engine import CudaQwen3Engine

        engine = CudaQwen3Engine.from_checkpoint(args.checkpoint, dtype=torch.bfloat16)
        states = {len(ids): engine.allocate_cache(len(ids) + NEW_TOKENS - 1) for ids in prompts}
        with torch.inference_mode():
            cases = benchmark_cases(
                args, prompts, lambda ids: cuda_tokens(engine, states[len(ids)], ids, annotate),
            )
        tag = "cuda-eager"
        implementation = {"engine": "cuda", "decode": "eager", "attention": "flash-attention-3"}

    return {
        "schema_version": 1,
        "source_tag": args.source_tag or tag,
        "implementation": implementation,
        "run": {
            "checkpoint": PINNED_CHECKPOINT,
            "device": "cuda",
            "gpu": gpu.name,
            "gpu_uuid": str(gpu.uuid),
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "host": host_fingerprint(),
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "prefix_cache": False,
            "chunked_prefill": False,
            "speculative_decoding": False,
            "batch_size": 1,
            "temperature": 0,
            "ignore_eos": True,
            "timing": "token_ready_wall_time",
            "new_tokens": NEW_TOKENS,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("sglang", "cuda"), default="sglang")
    parser.add_argument("--source-tag")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="Label prefill and decode in an Nsight trace")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("Use at least one repeat and nonnegative warmups")
    try:
        result = benchmark_gpu(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ImportError, KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

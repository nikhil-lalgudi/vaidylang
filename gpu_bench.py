"""Benchmark batch-1 BF16 streaming with the fixed Qwen3 token fixtures."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

from harness import checkpoint_identity, request_sample, summarize_case


FIXTURES = Path(__file__).with_name("fixtures") / "qwen3-0.6b-batch1.json"
PINNED_CHECKPOINT = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd/"
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
)
PROMPT_LENGTHS = (128, 1024, 4096)
NEW_TOKENS = 128
SGLANG_VERSION = "0.5.23"


def stream_sample(start_stream, prompt_tokens, new_tokens):
    times = []
    output = []
    started = perf_counter_ns()
    for chunk in start_stream():
        arrived = perf_counter_ns()
        meta = chunk["meta_info"]
        if meta["prompt_tokens"] != prompt_tokens:
            raise ValueError("SGLang changed the prompt length")
        if meta["cached_tokens"] != 0:
            raise ValueError("Unexpected prefix cache hit")
        ids = chunk.get("output_ids", [])
        count = meta["completion_tokens"]
        if count == len(output) and not ids:
            continue
        if len(ids) != 1 or count != len(output) + 1:
            raise ValueError("Expected one token per stream update to measure raw token gaps")
        output.append(ids[0])
        times.append(arrived)
    if len(output) != new_tokens:
        raise ValueError(f"Stream yielded {len(output)} tokens, expected {new_tokens} tokens")
    return request_sample(started, times, output)


def benchmark_sglang(args):
    import torch

    if torch.version.cuda is None or torch.cuda.device_count() != 1:
        raise ValueError("Expose one NVIDIA CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("The GPU must support native BF16")

    if version("sglang") != SGLANG_VERSION:
        raise ValueError(f"Install the pinned SGLang {SGLANG_VERSION} release")
    if checkpoint_identity(args.checkpoint) != PINNED_CHECKPOINT:
        raise ValueError("Checkpoint configuration or weights differ from the pinned revision")

    import sglang

    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--id", f"GPU-{gpu.uuid}", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    prompts = json.loads(FIXTURES.read_text())
    if tuple(map(len, prompts)) != PROMPT_LENGTHS:
        raise ValueError("Expected fixed input fixtures of 128, 1024, and 4096 tokens")
    sampling = {"temperature": 0, "max_new_tokens": NEW_TOKENS, "ignore_eos": True}
    # Engine accepts the ServerArgs fields in the pinned release.
    # https://github.com/sgl-project/sglang/tree/v0.5.23/python/sglang/srt/arg_groups/fields
    settings = {
        "device": "cuda",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tp_size": 1,
        "max_running_requests": 1,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "speculative_algorithm": None,
        "skip_tokenizer_init": True,
        "stream_interval": 1,
        "incremental_streaming_output": True,
    }
    engine = sglang.Engine(model_path=str(args.checkpoint.resolve()), **settings)
    try:
        cases = []
        for ids in prompts:
            def start_stream():
                return engine.generate(input_ids=ids, sampling_params=sampling, stream=True)

            for _ in range(args.warmup):
                stream_sample(start_stream, len(ids), NEW_TOKENS)
            samples = [
                stream_sample(start_stream, len(ids), NEW_TOKENS)
                for _ in range(args.repeats)
            ]
            cases.append(summarize_case(ids, NEW_TOKENS, samples))
    finally:
        engine.shutdown()

    return {
        "schema_version": 1,
        "source_tag": f"sglang-{SGLANG_VERSION}",
        "implementation": {"engine": "sglang", "version": SGLANG_VERSION, "settings": settings},
        "run": {
            "checkpoint": PINNED_CHECKPOINT,
            "device": "cuda",
            "gpu": gpu.name,
            "gpu_uuid": str(gpu.uuid),
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "host": platform.node(),
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "prefix_cache": False,
            "chunked_prefill": False,
            "speculative_decoding": False,
            "batch_size": 1,
            "temperature": 0,
            "ignore_eos": True,
            "timing": "stream_wall_time",
            "new_tokens": NEW_TOKENS,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("Use at least one repeat and nonnegative warmups")
    try:
        result = benchmark_sglang(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ImportError, KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

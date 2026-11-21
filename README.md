# VaidyLang

[![CPU inference](https://github.com/nikhil-lalgudi/vaidylang/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/nikhil-lalgudi/vaidylang/actions/workflows/cpu.yml)

An experimental inference engine and reproducibility harness for Qwen3 and
DeepSeek V4.1 Flash. It includes CPU reference execution, a single-request
Qwen3 CUDA path, and tools for comparing logits, cache state, and latency.

## What runs today

| Path | Status |
| --- | --- |
| **Qwen3 CPU** | The pinned Qwen3-0.6B checkpoint generates from text or token IDs. Cached decode was checked against a full-prefix Transformers oracle. [CPU parity](results/hf-parity-extended.json) |
| **Qwen3 CUDA** | On one H100 NVL at batch size 1, the tuned path had 20–27% lower median time per output token than the recorded SGLang 0.5.23 baseline. Greedy output hashes matched at 1,024 and 4,096 prompt tokens. The 128-token case diverged on the first token. [Comparison](results/cuda-tuning/final-tiled/compare-sglang.json) |
| **DeepSeek CPU** | Reduced dense and FP8/FP4 checkpoints ran all 100 prompts, including images. The repo also records a 32,743-token reduced run and a CPU probe of the published BF16 vision tower through its aligner. [CPU evidence](docs/deepseek-cpu-reference.md#100-prompt-workload) |
| **DeepSeek CUDA** | A capture adapter targets the pinned upstream CUDA runtime. Published-weight capture, full-checkpoint execution in VaidyLang, and hardware-matched benchmarks are pending. [Reference replay](docs/deepseek-cpu-reference.md#capture-and-replay-reference-logits) · [GPU validation](docs/deepseek-cpu-reference.md#gpu-validation-pending) |

## Quickstart

Run from the repository root with Python 3.11 or newer.

**Verify cached Qwen3 decoding** with a deterministic fixture and no model download.

```bash
python3 -m pip install 'numpy>=2,<3'
OPENBLAS_NUM_THREADS=1 python3 harness.py verify --tiny \
  --prompt-lengths 1 8 32 128 --new-tokens 12
```

**Generate from a reduced DeepSeek checkpoint** using fixed input IDs.

```bash
fixture_dir=$(mktemp -d)
python3 -m deepseek_fixture "$fixture_dir"
OPENBLAS_NUM_THREADS=1 python3 harness.py generate \
  --checkpoint "$fixture_dir" --ids 2 11 3 7 9 --new-tokens 16
```

The [CPU container guide](docker/README.md) covers Docker and VS Code Dev
Containers. The [Claude](.claude/settings.json) and [Codex](.codex/config.toml)
project settings use the CPU workflow’s thread limits. See the
[CPU validation skill](.skills/cpu-validation/SKILL.md) for choosing and
interpreting a check.

## Guides and reports

- [DeepSeek CPU reference](docs/deepseek-cpu-reference.md) covers checkpoint inspection, text and image inputs, numerical parity, and distributed placement.
- [DeepSeek reference replay](docs/deepseek-cpu-reference.md#capture-and-replay-reference-logits) covers optional full-logit capture and follows the reference prefix after greedy choices diverge. [Prompt corpus](fixtures/deepseek-prompts.jsonl) · [Answer scoring and CPU CI](docs/deepseek-cpu-reference.md#score-answers-and-run-cpu-ci)
- [Qwen3 benchmark guide](docs/qwen3-benchmarks.md) covers the public checkpoint, CUDA and SGLang comparisons, profiling, and speculative verification.
- [Report index](results/README.md) links to the archived baseline, parity, speculation, and DeepSeek workload reports.

## Repository layout

| Location | Contents |
| --- | --- |
| [`harness.py`](harness.py) | CPU verification, generation, checkpoint inspection, and benchmark comparison |
| [`engine.py`](engine.py), [`cuda_engine.py`](cuda_engine.py), [`deepseek.py`](deepseek.py) | Qwen3 CPU and CUDA execution, plus the reduced DeepSeek CPU engine |
| [`tools/`](tools/) | Reference replay, checkpoint placement, and benchmark tools |
| [`fixtures/`](fixtures/), [`results/`](results/) | Pinned inputs and archived evidence |
| [`docs/`](docs/), [`third_party/deepseek_v41/`](third_party/deepseek_v41/) | Reproduction guides and vendored-source attribution |

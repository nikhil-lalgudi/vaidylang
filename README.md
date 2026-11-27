# VaidyLang

[![CPU inference](https://github.com/nikhil-lalgudi/vaidylang/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/nikhil-lalgudi/vaidylang/actions/workflows/cpu.yml)

An experimental inference engine and reproducibility harness for Qwen3,
Inkling, and DeepSeek V4.1 Flash. It includes CPU reference execution, a
single-request Qwen3 CUDA path, and tools for comparing logits, cache state,
and latency.

## What runs today

| Path | Status |
| --- | --- |
| **Qwen3 CPU** | A C++ engine runs the pinned Qwen3-0.6B checkpoint. The harness supports text and token IDs and retains the NumPy reference. [Build instructions](#build-the-c-cpu-engine) · [CPU parity](results/hf-parity-extended.json) |
| **Qwen3 CUDA** | On one H100 NVL at batch size 1, the tuned path had 20–27% lower median time per output token than the recorded SGLang 0.5.23 baseline. Greedy output hashes matched at 1,024 and 4,096 prompt tokens. The 128-token case diverged on the first token. [Comparison](results/cuda-tuning/final-tiled/compare-sglang.json) |
| **Inkling CPU** | The C++ text decoder reads the original BF16 Safetensors layout and runs FP32 inference. Reduced weights matched a Transformers oracle across full and sliding attention, convolution, and expert routing. [Serving guide](#serve-a-local-checkpoint) |
| **DeepSeek CPU** | Reduced dense and FP8/FP4 checkpoints ran all 100 prompts, including images. The C++ library accelerates FP8/FP4 activation quantization. [CPU evidence](docs/deepseek-cpu-reference.md#100-prompt-workload) |
| **DeepSeek CUDA** | A capture adapter targets the pinned upstream CUDA runtime. Published-weight capture, full-checkpoint execution in VaidyLang, and hardware-matched benchmarks are pending. [Reference replay](docs/deepseek-cpu-reference.md#capture-and-replay-reference-logits) · [GPU validation](docs/deepseek-cpu-reference.md#gpu-validation-pending) |

## Build the C++ CPU engine

Use CMake 3.24 or newer and a C++20 toolchain. macOS uses Accelerate. On
Debian or Ubuntu, install `pkg-config`, `libopenblas-dev`, and
`nlohmann-json3-dev`. CMake fetches a pinned JSON library when a system
package is unavailable. The parity tests use Python and NumPy. Set
`-DBUILD_TESTING=OFF` for a build without Python.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use any supported local Qwen3 Safetensors checkpoint, including the pinned
revision in the [Qwen3 guide](docs/qwen3-benchmarks.md).

```bash
build/vaidylang-qwen3 checkpoints/qwen3-0.6b 8 3 1 4 1 5
python3 harness.py generate --checkpoint checkpoints/qwen3-0.6b \
  --text 'The capital of France is' --new-tokens 8
```

The standalone executable accepts input IDs and prints a JSON array of
generated IDs. The harness automatically uses the C++ library in `build/`
for Qwen3 checkpoints. Use `--cpu-backend python` for the NumPy reference,
`--cpu-backend native` to require C++, or `VAIDYLANG_CPU_LIBRARY` to point
to a library in a different build directory. Text input requires the
checkpoint tokenizer and the Python `tokenizers` package.

Reduced DeepSeek inference automatically uses the C++ activation quantizer
when the library is available. Set `VAIDYLANG_CPU_LIBRARY=off` to reproduce
NumPy-only runs.

## Python quickstart

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

## Serve a local checkpoint

[Build the C++ library](#build-the-c-cpu-engine) for Inkling or native Qwen3.
Keep the tokenizer and chat template with the checkpoint. The server also
accepts reduced DeepSeek V4.1 checkpoints with their compatible tokenizer.

```bash
python3 -m pip install 'numpy>=2,<3' 'tokenizers>=0.22,<0.23' 'jinja2>=3.1,<4'
python3 harness.py serve --checkpoint checkpoints/Inkling \
  --name inkling --cpu-backend native --port 8000
```

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":32}'
```

For Qwen3, point the same command at `checkpoints/qwen3-0.6b` and set
`--name qwen3`. Use another port to run a second checkpoint alongside it.
To try reduced DeepSeek, create a checkpoint with `python3 -m deepseek_fixture
checkpoints/deepseek-fixture --with-tokenizer` and serve it with `--cpu-backend auto`.
Each process serves one checkpoint on `127.0.0.1` and accepts one request
at a time. The chat endpoint supports single, non-streaming greedy text
completions. `--max-context` limits prompt and completion tokens together,
with a default of 4,096 or the model limit if it is lower.

The Inkling engine implements the original BF16 checkpoint's text decoder.
It maps Safetensors shards and converts slices to FP32 when used. The full
checkpoint contains about 1.90 TB of tensors and requires a host sized for
its storage and inference workload. Full-weight execution, quantized variants,
Inkling 2, and image and audio input are outside the validated scope.
[Published Inkling checkpoint](https://huggingface.co/thinkingmachines/Inkling/tree/828496eeae4c243ff1a22f7f28ff83694f2f7bc9) ·
[Transformers text oracle](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/models/inkling/modeling_inkling.py)

Run the reduced native Inkling test with
`ctest --test-dir build -R native_inkling --output-on-failure`.

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
| [`native/`](native/) | C++ Qwen3 and Inkling CPU engines, standalone Qwen3 runner, and harness bindings |
| [`tools/`](tools/) | Reference replay, checkpoint placement, and benchmark tools |
| [`fixtures/`](fixtures/), [`results/`](results/) | Pinned inputs and archived evidence |
| [`docs/`](docs/), [`third_party/deepseek_v41/`](third_party/deepseek_v41/) | Reproduction guides and vendored-source attribution |

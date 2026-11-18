# Qwen3 inference engine starter

The engine runs the dense Qwen3 forward pass in NumPy and takes integer token IDs. `prefill` returns first-token logits and a request-local KV cache. Each `decode` call appends a token and returns logits for its successor. The starter uses one sequence at a time and FP32 computation.

## Run the harness

Use Python 3.11 or newer. The synthetic checkpoint is generated from a fixed seed.

```bash
python -m pip install 'numpy>=2,<3'
OPENBLAS_NUM_THREADS=1 python -m unittest -v
OPENBLAS_NUM_THREADS=1 python harness.py verify --tiny \
  --prompt-lengths 1 8 32 128 --new-tokens 12
OPENBLAS_NUM_THREADS=1 python harness.py bench --tiny \
  --prompt-lengths 16 64 256 --new-tokens 32 --repeats 30 \
  --source-tag initial --output results/baseline.json
```

`verify` compares logits and greedy token IDs at every step. It checks a teacher-forced path, which feeds preset continuation tokens, and a greedy path, which feeds back the engine's predictions. On the tiny checkpoint it compares cached decode with full-prefix recomputation. The unit tests also include independently calculated FP64 logits.

`bench` uses fixed pretokenized inputs and generates exactly the requested number of tokens, including after EOS. TTFT is wall-clock time through the first token, including cache allocation and prefill. TPOT is `(last token time - first token time) / (output tokens - 1)`. ITL records every successive token gap. Runs report p50 and p95, sequential output throughput, raw request times, and input and output token hashes. Each measured request starts with a fresh cache. Model load and warmup happen before measurement.

Save a candidate run with the same arguments and a different `--source-tag` and `--output`. `compare` checks that fixtures and run settings match. It exits with status 1 when a shape fails any threshold or its greedy outputs change.

```bash
python harness.py compare --baseline results/baseline.json \
  --candidate results/candidate.json \
  --min-tpot-improvement-pct 5 --max-ttft-regression-pct 5
```

## Inspect a checkpoint

The shared checkpoint reader accepts the published DeepSeek V4.1 Flash config and Safetensors index. Inspect its metadata before downloading the weight shards.

```bash
python harness.py inspect-checkpoint --checkpoint checkpoints/DeepSeek-V4.1-Flash --metadata-only
```

`Checkpoint.open(path).read_tensor(name)` reads a local tensor using the native PyTorch dtype. The CPU DeepSeek loader can also interpret stored E8M0 and packed FP4 bytes directly with NumPy.

## Run reduced DeepSeek V4.1 Flash on CPU

`engine.load_engine` dispatches from the checkpoint's root model type. The same generation command loads Qwen3 and DeepSeek checkpoints and takes integer token IDs. The deterministic DeepSeek fixture uses the published text topology with smaller dimensions and stores FP8 and packed FP4 weights in two local shards.

```bash
fixture_dir=$(mktemp -d)
python3 -m deepseek_fixture "$fixture_dir"
python3 harness.py generate --checkpoint "$fixture_dir" --ids 2 11 3 7 9 --new-tokens 16
```

Install `tokenizers` for text input, then use the checkpoint's local `tokenizer.json`.
DeepSeek wraps plain text as a user turn with its pinned V4.1 encoder. Qwen3
tokenizes plain text as a raw completion prefix. The reduced tokenizer below
uses a synthetic vocabulary and exercises the CLI on the reduced weights.

```bash
python3 -m pip install tokenizers
python3 -m deepseek_fixture "$fixture_dir" --with-tokenizer
python3 harness.py generate --checkpoint "$fixture_dir" --text 'Hello, world!' --new-tokens 4
python3 harness.py generate --checkpoint "$fixture_dir" --text-file prompt.txt
python3 harness.py generate --checkpoint "$fixture_dir" --messages messages.json
```

`messages.json` is a JSON array of DeepSeek role and content objects, such as
`[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello, world!"}]`.
Use `--thinking` and optionally `--reasoning-effort 1–100` to use the pinned
reasoning format. Text and chat stop at checkpoint EOS and return decoded text,
the tokenizer SHA-256, and the prompt and continuation IDs with their hashes.
The continuation IDs include EOS when emitted. `--ids` replays a fixed number
of steps and works with checkpoints that have no tokenizer.

For an image prompt, install PyTorch and Pillow and build the reduced vision
fixture. Use `<image>path.png</image>` in plain text or a structured
`{"type":"image_url","image_url":{"url":"path.png"}}` block in a message.
Both accept local paths and base64 data URLs. Files named inside a prompt
file are resolved relative to that file.

```bash
vision_dir=$(mktemp -d)
python3 -m deepseek_fixture "$vision_dir" --vision
python3 harness.py generate --checkpoint "$vision_dir" \
  --text 'What is shown? <image>path.png</image>' --new-tokens 8
python3 harness.py prepare-prompt --checkpoint "$vision_dir" --messages messages.json
```

`prepare-prompt` also works with just the published `config.json` and
`tokenizer.json`. It emits expanded IDs, per-token image types, image span
positions, grids, source hashes and BF16 patch hashes without loading weights.
Reduced vision generation uses synthetic weights, so it tests pixel input,
embedding replacement, routing, Engram masking and decode state.

The CPU reference materializes reduced weights and uses scalar cache updates, including when verifying a block of draft tokens. The published checkpoint is available to the metadata and placement tools. CPU reproduction commands, numerical results, and pending GPU checks are in the [DeepSeek CPU reference handoff](deepseek-cpu-reference.md).

## Check the public Qwen3 checkpoint

On a machine with access to Hugging Face, download the pinned Qwen3-0.6B revision. An FP32 CPU oracle and candidate in the same process need several GB of RAM.

```bash
python -m pip install 'huggingface_hub>=0.34,<1' 'transformers>=4.52,<5' torch
hf download Qwen/Qwen3-0.6B \
  --revision 167b8104f88905a951069f5f95f9776908da5f68 \
  --local-dir checkpoints/Qwen3-0.6B
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python harness.py verify \
  --checkpoint checkpoints/Qwen3-0.6B --oracle hf \
  --prompt-lengths 16 64 --new-tokens 4
```

The Transformers oracle recomputes every prefix in FP32. `--oracle-device cuda` puts the oracle on a CUDA GPU. The starter engine itself runs on CPU. The local FP32 run passed on both forced and greedy paths at prompt lengths 16 and 64 with four output tokens, 128 with 32 output tokens, and 1,024 with two output tokens. See `results/hf-parity.json`, `results/hf-parity-extended.json`, and `results/hf-parity-1024.json`.

## Build the GPU engine in this order

1. **Capture the baseline.** Use a Linux host with one visible NVIDIA GPU and native BF16. Install SGLang 0.5.23 in a GPU environment with a CUDA 13.0 toolkit, a compatible NVIDIA driver, a C++ compiler, and Ninja. Set `CUDA_HOME` and prepend its `bin` directory to `PATH` so JIT compilation uses that toolkit. The benchmark validates both checkpoint file hashes, runs one request at a time with BF16 weights and KV cache, disables prefix reuse, chunked prefill, and speculative decoding, and generates 128 greedy tokens including after EOS.

   ```bash
   uv pip install --prerelease=allow 'sglang==0.5.23'
   CUDA_VISIBLE_DEVICES=0 python gpu_bench.py \
     --checkpoint checkpoints/Qwen3-0.6B --output results/sglang-baseline.json
   ```

   The runner uses the serialized input IDs in `fixtures/qwen3-0.6b-batch1.json` at 128, 1,024, and 4,096 tokens. It saves GPU UUID, driver and runtime versions, request percentiles, all 127 token gaps per request, and input and output hashes. TTFT and TPOT measure wall time in SGLang’s static batch runner. Each timestamp follows transfer of a single completed token to the host. The next decode uses the device token.

   Capture a separate trace so profiling overhead stays out of the baseline JSON.

   ```bash
   CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
     --output=results/sglang-baseline python gpu_bench.py \
     --checkpoint checkpoints/Qwen3-0.6B --repeats 1 --profile \
     --output results/sglang-profile-samples.json
   ```

   [SGLang 0.5.23 release and install instructions](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) · [SGLang static batch runner](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/benchmark/one_batch.py) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

2. **Verify the eager CUDA engine.** The engine uses PyTorch CUDA tensor operations with a device KV cache and position tensor. BF16 uses FlashAttention 3 from the CUDA environment installed above. Use the full-prefix Transformers oracle for both precisions.

   ```bash
   python harness.py verify --cuda --checkpoint checkpoints/Qwen3-0.6B \
     --oracle-device cuda --prompt-lengths 16 64 128 1024 4096 --new-tokens 4
   python harness.py verify --cuda --dtype bfloat16 --checkpoint checkpoints/Qwen3-0.6B \
     --oracle-device cuda --prompt-lengths 16 64 128 1024 4096 --new-tokens 4 \
     --atol 0.5 --rtol 0.025
   ```

   The BF16 check allows for the logit differences observed between fused and eager attention. At each checked step, the CUDA choice must be among the oracle's exact logit maxima. Use the same BF16 benchmark command with `--engine cuda` and a new output path. Token selection and the next decode input stay on the GPU, with a host copy used to time each emitted token.

3. **Profile and test one change per run.** Add `--cuda-graph` to the CUDA verification and benchmark commands to replay decode with stable cache and token addresses. Capture each variant with Nsight Systems and `--profile` to mark the entire decode step.

   ```bash
   nsys export --type=sqlite --output=results/candidate/decode.sqlite results/candidate/decode.nsys-rep
   python tools/decode_trace.py results/candidate/decode.sqlite > results/candidate/trace-summary.json
   ```

   The summary measures GPU kernel activity inside each decode range. The unprofiled benchmark JSON is the source of latency percentiles. Add `--fused-rms` to the BF16 verification and CUDA benchmark to use the Triton RMSNorm experiment. Add `--packed-projections` to combine Q/K/V and gate/up linear operations that use the same input.

   With packing enabled, `--unpacked-prefill` uses separate projections for prefill. `--dense-prefill 4096` uses dense causal attention and separate KV writes starting at 4,096 input tokens. `--decode-splits 16` requests 16 FlashAttention KV splits for decode, and `--decode-splits 0` uses automatic scheduling. With fused RMSNorm enabled, `--fused-decode-pointwise` fuses residual adds with RMSNorm, Q and K normalization, and SwiGLU in decode.

   `--tiled-decode-attention` uses an experimental two-kernel Triton path for BF16 decode. The first kernel applies RoPE, appends K/V, and computes partial attention. The second combines the partial results. This option uses its own tile sizes and requires `--decode-splits 0`.

4. **Validate each patch.** Use the previous trace to choose one change. Save the patch, its hypothesis, forced and greedy parity output, unprofiled benchmark JSON, and an Nsight Systems trace in a directory named for the patch. Keep the baseline JSON and input IDs fixed. Use the same GPU and token timing boundary for the candidate and keep its run metadata and JSON fields compatible with the baseline. Run the existing comparison against the frozen baseline.

   ```bash
   python harness.py compare --baseline results/sglang-baseline.json \
     --candidate results/candidate/bench.json \
     --min-tpot-improvement-pct 10 --max-ttft-regression-pct 5 \
     --output results/candidate/compare.json
   ```

   Each input length must meet both latency thresholds and match the baseline output token hashes. Validate the selected patch on held-out prompts as well.

## Run speculative verification

The speculative benchmark uses the same pinned Qwen3 checkpoint. It compares target-only decode with block sizes 2, 4, and 8 on six coding and chat prompts and writes the first token mismatch if a continuation diverges. The tiled verifier shares the scalar attention reduction and masks staged tokens after the current query row. The graph advances the logical cache only for the verified prefix.

```bash
python speculative_bench.py --checkpoint checkpoints/Qwen3-0.6B \
  --verification tiled --repeats 3 --warmup 2 --output results/speculative/ngram-tiled.json
python -m tools.speculative_parity --checkpoint checkpoints/Qwen3-0.6B \
  --verification tiled --output results/speculative/parity-tiled.json
python -m tools.train_speculative_draft --checkpoint checkpoints/Qwen3-0.6B \
  --output results/speculative/learned-draft.npz --report results/speculative/learned-training.json
QWEN3_SPECULATIVE_DRAFT_WEIGHTS=results/speculative/learned-draft.npz \
  python speculative_bench.py --checkpoint checkpoints/Qwen3-0.6B --verification tiled \
  --draft-factory speculative_learned_draft:LearnedDraft --repeats 3 --warmup 2 \
  --output results/speculative/learned-tiled.json
```

[Baseten engine](https://github.com/basetenlabs/deepseek-ultrafast) · [Qwen3-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/167b8104f88905a951069f5f95f9776908da5f68) · [SGLang benchmarks](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

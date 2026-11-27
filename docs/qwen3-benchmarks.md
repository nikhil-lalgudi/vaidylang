# Qwen3 correctness and benchmarks

Run these commands from the repository root. See the [report index](../results/README.md) for archived runs.

## Verify and benchmark a synthetic checkpoint

The synthetic checkpoint is generated from a fixed seed. Use Python 3.11 or newer.

```bash
python -m pip install 'numpy>=2,<3'
report_dir=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 python harness.py verify --tiny \
  --prompt-lengths 1 8 32 128 --new-tokens 12
OPENBLAS_NUM_THREADS=1 python harness.py bench --tiny \
  --prompt-lengths 16 64 256 --new-tokens 32 --repeats 30 \
  --source-tag initial --output "$report_dir/baseline.json"
```

`verify` compares logits and greedy token IDs at every step. It checks a teacher-forced path, which feeds preset continuation tokens, and a greedy path, which feeds back the engine's predictions. On the tiny checkpoint it compares cached decode with full-prefix recomputation. The unit tests also include independently calculated FP64 logits.

`bench` uses fixed pretokenized inputs and generates exactly the requested number of tokens, including after EOS. TTFT is wall-clock time through the first token, including cache allocation and prefill. TPOT is `(last token time - first token time) / (output tokens - 1)`. ITL records every successive token gap. Runs report p50 and p95, sequential output throughput, raw request times, and input and output token hashes. Each measured request starts with a fresh cache. Model load and warmup happen before measurement.

Save a candidate run with the same arguments and a different `--source-tag` and `--output`. `compare` checks that fixtures and run settings match. It exits with status 1 when a shape fails any threshold or its greedy outputs change.

```bash
python harness.py compare --baseline "$report_dir/baseline.json" \
  --candidate "$report_dir/candidate.json" \
  --min-tpot-improvement-pct 5 --max-ttft-regression-pct 5
```

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

The Transformers oracle recomputes every prefix in FP32. `--oracle-device cuda` puts the oracle on a CUDA GPU. This command runs VaidyLang on CPU. The local FP32 run passed on both forced and greedy paths at prompt lengths 16 and 64 with four output tokens, 128 with 32 output tokens, and 1,024 with two output tokens. [Short prompts](../results/hf-parity.json) · [128 tokens](../results/hf-parity-extended.json) · [1,024 tokens](../results/hf-parity-1024.json)

## Benchmark the C++ CPU engine

Build the [C++ CPU engine](../README.md#build-the-c-cpu-engine) and use the
same checkpoint, seed, and BLAS settings for both runs.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
report_dir=$(mktemp -d)
for backend in python native; do
  python harness.py bench --checkpoint checkpoints/Qwen3-0.6B \
    --cpu-backend "$backend" --prompt-lengths 8 128 512 --new-tokens 8 \
    --seed 123 --warmup 2 --repeats 7 --output "$report_dir/$backend.json"
done
python harness.py compare --baseline "$report_dir/python.json" \
  --candidate "$report_dir/native.json"
```

The native runner packs Q/K/V and gate/up projections, reuses request-local
buffers, and processes prefill in blocks of 128 tokens. On Apple Silicon,
large BF16 token tables stay memory-mapped in their original format. The
output projection uses up to four dispatch work items with FP32 accumulation.

On an Apple M2 Pro, paired runs on the pinned Qwen3-0.6B checkpoint
produced identical greedy token sequences. The maximum absolute difference
across teacher-forced logits was `9.23e-5`.

| Prompt tokens | Median first token, NumPy → C++ | Median output token, NumPy → C++ |
| ---: | ---: | ---: |
| 8 | 398.6 → 267.2 ms | 115.5 → 87.9 ms |
| 128 | 506.6 → 445.4 ms | 104.0 → 72.9 ms |
| 512 | 2,054.5 → 2,019.2 ms | 94.0 → 77.5 ms |

The 21-request repeat at 8 prompt tokens measured 125.4 → 106.0 ms
median first-token latency and 164.8 → 243.6 ms p95. Accelerate vector
exponentials were tested separately and reverted after increasing median
prefill latency for the 128- and 512-token cases.

### Long-context CPU attention

The Accelerate path vectorizes softmax for attention rows with at least
128 visible tokens. In paired runs against `4cad2d4` on an Apple M2 Pro,
median time to first token improved by 22–30% on longer prompts. The
runs used the pinned Qwen3-0.6B checkpoint, seed 123, four output tokens,
one warmup, five measured requests per build, and one BLAS thread.

| Prompt tokens | Median first token, before → after |
| ---: | ---: |
| 512 | 1,745 → 1,364 ms |
| 1,024 | 3,858 → 2,920 ms |
| 2,048 | 9,609 → 6,733 ms |

Greedy tokens matched at 8, 128, 512, 1,024, and 2,048 prompt tokens.
The maximum teacher-forced logit difference was `5.37e-5`.

## Reproduce the batch-1 CUDA benchmark

1. **Capture the baseline.** Use a Linux host with one visible NVIDIA GPU and native BF16. Install SGLang 0.5.23 in a GPU environment with a CUDA 13.0 toolkit, a compatible NVIDIA driver, a C++ compiler, and Ninja. Set `CUDA_HOME` and prepend its `bin` directory to `PATH` so JIT compilation uses that toolkit. The benchmark validates both checkpoint file hashes, runs one request at a time with BF16 weights and KV cache, disables prefix reuse, chunked prefill, and speculative decoding, and generates 128 greedy tokens including after EOS.

   ```bash
   report_dir=$(mktemp -d)
   uv pip install --prerelease=allow 'sglang==0.5.23'
   CUDA_VISIBLE_DEVICES=0 python gpu_bench.py \
     --checkpoint checkpoints/Qwen3-0.6B --output "$report_dir/sglang-baseline.json"
   ```

   The runner uses the [frozen input IDs](../fixtures/qwen3-0.6b-batch1.json) at 128, 1,024, and 4,096 tokens. It saves GPU UUID, driver and runtime versions, request percentiles, all 127 token gaps per request, and input and output hashes. TTFT and TPOT measure wall time in SGLang’s static batch runner. Each timestamp follows transfer of a single completed token to the host. The next decode uses the device token.

   Capture a separate trace to keep profiling overhead out of the baseline JSON.

   ```bash
   CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
     --output="$report_dir/sglang-baseline" python gpu_bench.py \
     --checkpoint checkpoints/Qwen3-0.6B --repeats 1 --profile \
     --output "$report_dir/sglang-profile-samples.json"
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

   The BF16 check allows for the logit differences observed between fused and eager attention. At each checked step, the CUDA choice must be among the oracle's exact logit maxima. Token selection and the next decode input stay on the GPU, with a host copy used to time each emitted token.

3. **Profile and test one change per run.** Add `--cuda-graph` to the CUDA verification and benchmark commands to replay decode with stable cache and token addresses. Capture each variant with Nsight Systems and `--profile` to mark the entire decode step.

   ```bash
   nsys export --type=sqlite --output="$report_dir/decode.sqlite" "$report_dir/decode.nsys-rep"
   python tools/decode_trace.py "$report_dir/decode.sqlite" > "$report_dir/trace-summary.json"
   ```

   The summary measures GPU kernel activity inside each decode range. The unprofiled benchmark JSON is the source of latency percentiles. Add `--fused-rms` to the BF16 verification and CUDA benchmark to use the Triton RMSNorm experiment. Add `--packed-projections` to combine Q/K/V and gate/up linear operations that use the same input.

   With packing enabled, `--unpacked-prefill` uses separate projections for prefill. `--dense-prefill 4096` uses dense causal attention and separate KV writes starting at 4,096 input tokens. `--decode-splits 16` requests 16 FlashAttention KV splits for decode, and `--decode-splits 0` uses automatic scheduling. With fused RMSNorm enabled, `--fused-decode-pointwise` fuses residual adds with RMSNorm, Q and K normalization, and SwiGLU in decode.

   `--tiled-decode-attention` uses an experimental two-kernel Triton path for BF16 decode. The first kernel applies RoPE, appends K/V, and computes partial attention. The second combines the partial results. This option uses its own tile sizes and requires `--decode-splits 0`.

   The archived tuned run used these settings.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python gpu_bench.py --engine cuda \
     --checkpoint checkpoints/Qwen3-0.6B --output "$report_dir/bench.json" \
     --cuda-graph --fused-rms --packed-projections --unpacked-prefill \
     --dense-prefill 4096 --fused-decode-pointwise --tiled-decode-attention
   ```

4. **Validate each patch.** Use the previous trace to choose one change. Save the patch, its hypothesis, forced and greedy parity output, unprofiled benchmark JSON, and an Nsight Systems trace in a directory named for the patch. Keep the baseline JSON and input IDs fixed. Use the same GPU and token timing boundary for the candidate and keep its run metadata and JSON fields compatible with the baseline. Run the existing comparison against the frozen baseline.

   ```bash
   python harness.py compare --baseline "$report_dir/sglang-baseline.json" \
     --candidate "$report_dir/bench.json" \
     --min-tpot-improvement-pct 10 --max-ttft-regression-pct 5 \
     --output "$report_dir/compare.json"
   ```

   Each input length must meet both latency thresholds and match the baseline output token hashes. Validate the selected patch on held-out prompts as well. The archived comparison passed at 1,024 and 4,096 tokens. At 128 tokens, the first greedy token differed. [Tuned CUDA comparison](../results/cuda-tuning/final-tiled/compare-sglang.json) · [First-token diagnostic](../results/cuda-tuning/final-tiled/frozen-128-output-diagnostic.json)

## Run speculative verification

The speculative benchmark uses the same pinned Qwen3 checkpoint. It compares target-only decode with block sizes 2, 4, and 8 on six coding and chat prompts and writes the first token mismatch if a continuation diverges. The tiled verifier shares the scalar attention reduction and masks staged tokens after the current query row. The graph advances the logical cache only for the verified prefix.

```bash
report_dir=$(mktemp -d)
python speculative_bench.py --checkpoint checkpoints/Qwen3-0.6B \
  --verification tiled --repeats 3 --warmup 2 --output "$report_dir/ngram-tiled.json"
python -m tools.speculative_parity --checkpoint checkpoints/Qwen3-0.6B \
  --verification tiled --output "$report_dir/parity-tiled.json"
python -m tools.train_speculative_draft --checkpoint checkpoints/Qwen3-0.6B \
  --output "$report_dir/learned-draft.npz" --report "$report_dir/learned-training.json"
QWEN3_SPECULATIVE_DRAFT_WEIGHTS="$report_dir/learned-draft.npz" \
  python speculative_bench.py --checkpoint checkpoints/Qwen3-0.6B --verification tiled \
  --draft-factory speculative_learned_draft:LearnedDraft --repeats 3 --warmup 2 \
  --output "$report_dir/learned-tiled.json"
```

[Baseten engine](https://github.com/basetenlabs/deepseek-ultrafast) · [Qwen3-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/167b8104f88905a951069f5f95f9776908da5f68) · [SGLang benchmarks](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

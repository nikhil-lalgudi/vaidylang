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

## Check the public checkpoint

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

The Transformers oracle recomputes every prefix in FP32. `--oracle-device cuda` puts the oracle on a CUDA GPU. The starter engine itself runs on CPU. The checkpoint and GPU commands are provided for the next run.

## Build the GPU engine in this order

1. **Capture the baseline.** On one CUDA GPU with BF16 support, pin the model, SGLang revision, driver, and CUDA version. Collect a static batch-1 profile first. A SGLang installation provides the following entry point.

   ```bash
   python -m sglang.benchmark.one_batch \
     --model-path checkpoints/Qwen3-0.6B --dtype bfloat16 \
     --batch-size 1 --input-len 128 1024 4096 --output-len 128
   ```

   Use the static runs to locate expensive operators. Use the same serialized prompts and decode budgets for the eventual head-to-head comparison. Repeat on the same GPU with matching precision, cache reuse, and speculative decoding settings.

2. **Move the forward pass to CUDA.** Preserve the `prefill` and `decode` contract. Start with CUDA tensor ops, a preallocated device KV cache, and fused attention. Verify GPU FP32 against the reference, then introduce BF16 and compare against a BF16 reference with tolerances calibrated on real prompts. Keep token selection and position updates on the device when measuring decode.

3. **Profile and test one change per run.** Use Nsight Systems to see launch gaps, copies, and synchronization. Measure kernel bottlenecks with Nsight Compute. Test CUDA Graph replay with stable buffer addresses for launch-bound decode. Test packed QKV projections or fused residual and RMSNorm if the trace implicates those operations. Measure wall time for request latency and CUDA events for GPU time.

4. **Validate a result on held-out prompts.** Preserve the baseline and run the same committed workloads before and after each patch. Save the change, hypothesis, parity output, benchmark JSON, and trace. A starting success target is at least 10% lower p50 TPOT at each batch-1 context length, with at most 5% worse p95 TTFT, compared with SGLang on the same GPU. For the serving comparison, use the same streaming client, prompt IDs, output caps, and request schedule on both servers. Measure concurrency and p95 inter-token latency before adding continuous batching or paged KV.

[Baseten engine](https://github.com/basetenlabs/deepseek-ultrafast) · [Qwen3-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/167b8104f88905a951069f5f95f9776908da5f68) · [SGLang benchmarks](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

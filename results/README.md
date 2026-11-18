# Experiment reports

These are archived runs. The reproduction guides describe the checkpoint,
hardware, input IDs, and settings that make each comparison meaningful.
[Qwen3 guide](../docs/qwen3-benchmarks.md) · [DeepSeek reference](../docs/deepseek-cpu-reference.md)

## Qwen3

| Report | What it measures |
| --- | --- |
| [CPU parity at 128 tokens](hf-parity-extended.json) · [1,024 tokens](hf-parity-1024.json) | Cached decoding against full-prefix Transformers logits with the pinned Qwen3-0.6B checkpoint |
| [SGLang baseline](sglang-baseline.json) · [Tuned CUDA run](cuda-tuning/final-tiled/benchmark.json) · [Comparison](cuda-tuning/final-tiled/compare-sglang.json) | Batch-1 H100 NVL latency and greedy-output hashes on identical input IDs |
| [CUDA tuning summary](cuda-tuning/summary.json) · [First-token diagnostic](cuda-tuning/final-tiled/frozen-128-output-diagnostic.json) | Optimization passes, held-out checks, and evidence for the 128-token disagreement |
| [Speculation summary](speculative/summary.json) · [Tiled verifier parity](speculative/parity-tiled.json) | Greedy agreement, draft acceptance, and timing for the n-gram and learned drafts |

## DeepSeek V4.1 Flash

| Report | What it measures |
| --- | --- |
| [Dense parity](deepseek-cpu/parity-dense.json) · [FP8/FP4 parity](deepseek-cpu/parity-quantized.json) · [Speculation](deepseek-cpu/speculative.json) | Reduced-model logits against the independent FP64 oracle and speculative decoding against scalar execution |
| [Frozen prompts](deepseek-cpu/prompts-frozen.json.gz) · [Dense smoke](deepseek-cpu/prompts-smoke-dense.json.gz) · [Quantized smoke](deepseek-cpu/prompts-smoke-quantized.json.gz) | Archived 100-case prompt compilation and reduced-weight generation. These reports predate the long-document and tool-prompt revisions. |
| [32,743-token dense run](deepseek-cpu/prompt-32743-dense.json.gz) · [Quantized run](deepseek-cpu/prompt-32743-quantized.json.gz) | Reduced-model generation after a long prompt |
| [Upstream checks](deepseek-cpu/upstream-checks.json) · [Published BF16 vision](deepseek-cpu/vision-bf16-published-resize.json) | Pinned tokenizer, image preprocessing and Engram checks, plus the published vision tower and aligner on CPU |
| [Placement summary](deepseek-cpu/placement.md) · [Per-rank data](deepseek-cpu/placement.json) | Published checkpoint storage and rank memory estimates derived from Safetensors headers |

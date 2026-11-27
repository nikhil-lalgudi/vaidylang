# DeepSeek V4.1 Flash CPU reference

Run the commands below from the repository root. See the [report index](../results/README.md) for archived runs.

The text topology and storage encodings follow DeepSeek V4.1 Flash revision `314113f8a7dadb388215396ba5fbee2c610a04d6`. The reduced CPU engine uses NumPy and can run activation quantization and hyper-connections in C++. [Release repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6) · [DeepSeek MIT license](../LICENSE.DeepSeek)

## C++ CPU performance

Build the [C++ CPU library](../README.md#build-the-c-cpu-engine) to enable the
native kernels automatically. Set `VAIDYLANG_CPU_LIBRARY=off` to use NumPy for
the same workload. Both paths run the reduced checkpoint and produce the
same greedy output in the comparisons below.

On an Apple M2 Pro, native kernels and reuse of quantized expert
inputs reduced median latency by 57–60% on the reduced FP8/FP4 fixture.
Against the native kernels alone, reusing the expert input cut median
prefill by another 6–8%. Native hyper-connections reduced median latency by
20–23% on the reduced dense fixture. The paired runs used seed 123, 16
output tokens, two warmups, and seven measured requests per backend.

| Prompt tokens | First token, original → optimized | Output token, original → optimized |
| ---: | ---: | ---: |
| 8 | 38.0 → 15.8 ms | 4.96 → 2.00 ms |
| 128 | 619.9 → 248.6 ms | 4.89 → 2.01 ms |
| 512 | 2,448.0 → 1,032.6 ms | 4.86 → 2.08 ms |

Both commands below reuse the expert input. Compare with the pre-migration
revision to measure the combined speedup.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
fixture_dir=$(mktemp -d)
python3 -m deepseek_fixture "$fixture_dir" --context 1024
VAIDYLANG_CPU_LIBRARY=off python3 harness.py bench --checkpoint "$fixture_dir" \
  --prompt-lengths 8 128 512 --new-tokens 16 --warmup 2 --repeats 7
python3 harness.py bench --checkpoint "$fixture_dir" \
  --prompt-lengths 8 128 512 --new-tokens 16 --warmup 2 --repeats 7
```

| Pinned file | SHA-256 |
| --- | --- |
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `inference/model.py` | `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `inference/convert.py` | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `encoding/encoding.py` | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |
| `LICENSE` | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |

## Tensor mapping

The public index already uses the post-conversion text names. `embed.weight`, `head.weight` and `norm.weight` are global. Backbone blocks use `layers.N`; draft stages use `mtp.N` and tie the embedding and output head to the backbone. Each block has latent attention projections, normalization, hyper-connection parameters, router, routed experts and one shared expert. KV source blocks additionally own a compressor and index key projection, and index source blocks own the query side of indexing. Engram layers own an embedding table, a gated key/value projection, and per-stream scale vectors. Image-only tensors include `vision.*`, `aligner.*`, `image_*` and the alternate `.gate.bias_vl` router biases.

Separate operation tests derive expected values by hand from the pinned formulas. These include adjacent-pair RoPE, a softmax sink with zero value, featurewise compressor softmax, a forced newest candidate block, input-to-output hyper-connection orientation, and bias-corrected expert selection with unbiased weights.

## Commits and evidence

| Commit | Changed modules and evidence |
| --- | --- |
| `92445b1` | `speculative_cpu.py` stages and commits a complete request-state snapshot. |
| `6aab8f5` | `deepseek_quant.py` decodes stored bytes and computes scaled reference linears. |
| `9f1a42b` | `tools/cpu_parity.py`, `tools/deepseek_oracle.py`, the frozen fixture, and the parity and archive reports. |
| `e2148b3` | `deepseek_placement.py`, ranged header reader, placement CLI, and rank and collective tests. |
| `469273a` | Shared loader and CLI, checkpoint storage reader, DeepSeek config, token state, text backbone, and reduced fixture. |
| `615376a` | `deepseek_draft.py`, DSpark and transaction regressions, learned draft adapter, and CPU benchmark. |

The reduced checkpoint has six backbone layers, three DSpark stages, vocabulary 96, hidden size 32, and context 128. Its two quantized shards contain 733 tensors and 514,564 tensor bytes. Ratios 0, 2, and 1, separate KV and index sources, two-level candidate filtering, and two Engram tables execute in the fixture. Fresh-prefix and cached teacher-forced and greedy generation matched bit for bit at prompt lengths 1, 6, and 17 over 16 steps. The standalone generation command returned these 16 IDs for prompt `2 11 3 7 9`.

```text
53 11 82 71 12 2 55 6 39 55 35 14 75 50 84 50
```

| Comparison | Dense fixture | FP8/FP4 fixture |
| --- | ---: | ---: |
| Block verification versus scalar target | 48/48 logit vectors bit-identical | 48/48 logit vectors bit-identical |
| Scalar target versus independent cache-free FP64 oracle, max absolute logit error | `1.428731e-6` | `5.132859e-7` |
| Scalar target versus frozen oracle IDs | 16/16 | 16/16 |

The independent oracle rebuilds raw and compressed prefixes from scratch and separately interprets weight bytes, computes activation rounding, and evaluates the attention, Engram, hyper-connection, and expert formulas. The comparison uses `atol=3e-5, rtol=2e-5` for FP32 projection and residual accumulation versus FP64. Verifier rows use exact comparison. The report preserves prefix IDs and hash, selected token and rank, error, tolerance violations, and first differing operation. Both fixtures had zero tolerance violations. The injected rotary regression localized the first difference to position 5, layer 0, operation `raw_kv`, and captures the two tensor values. [Dense parity](../results/deepseek-cpu/parity-dense.json) · [Quantized parity](../results/deepseek-cpu/parity-quantized.json)

DSpark reads the mean across hyper-connection streams immediately before each configured target layer. Its fixed five-slot draft uses the target feature window, a verified seed token, and four noise-token slots. It adds a Markov logit bias conditioned on the preceding token and computes confidence from the matching Markov embedding and draft hidden state. An independent FP64 comparison of all three stages yielded the same draft IDs and max absolute errors of `6.455e-7` for logits and `8.726e-8` for confidence, using `atol=rtol=1e-5`. Draft KV is recomputed from committed target features. The target verifier clones the request for staging and replays only verified inputs for a partial commit, including the raw ring, pending compressor group, compressed KV, index keys, Engram history, and target features. Padding rows never advance cache state.

All 27 held-out CPU speculative requests matched target-only greedy output exactly across n-gram, learned, and DSpark drafts. Learned training, validation, and acceptance prompts were distinct. Timings below are aggregate over three 16-token requests per setting. This synthetic workload measures correctness and how often the controller can use a draft.

| DSpark verifier width | Maximum proposals | Accepted / proposed | Emitted / cycle | Draft time | Verification time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 2 / 40 | 1.05 | 261 ms | 596 ms |
| 4 | 3 | 3 / 108 | 1.07 | 258 ms | 972 ms |
| 8 | 5 | 3 / 165 | 1.07 | 258 ms | 1,222 ms |

The width includes the previous token consumed by the verifier. The fixed draft head still computes all five slots at each supported width. It supplies at most five proposals even at width eight. All 27 comparisons report `first_mismatch: null`. [Per-request IDs, timings, and draft acceptance](../results/deepseek-cpu/speculative.json)

Placement accounts for 510,286,023,000 stored tensor bytes across 48 shards from 10,684,928 bytes of Safetensors JSON headers. All 96,085 entries have an assigned placement. At eight ranks, batch size 1, and 1,048,576 context tokens, estimated text weights, scales, and persistent state total 70.30 GB per rank. Selected one-query scratch buffers add about 1.10 GB per rank. Vision tensors add up to 0.97 GB per rank if retained. The placement report includes 1, 2, 4 and 8 ranks, dtype and shape verification, and the allocation assumptions. [Placement and memory detail](../results/deepseek-cpu/placement.md)

## Reproduce on CPU

Verified on macOS 26.6.2 arm64 with Python 3.11.5, NumPy 2.4.4 and PyTorch 2.4.1. PyTorch is used by the learned draft training, native dtype probe, and Gloo test. Safetensors, Triton, TileLang, and SGLang kernels were absent. The NumPy decoder matched native CPU PyTorch for all 256 E4M3FN patterns. E8M0 and packed FP4 used the stored-byte reference path because this PyTorch build lacks those native dtypes.

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
fixture_dir=$(mktemp -d)
report_dir=$(mktemp -d)
python3 -m deepseek_fixture "$fixture_dir"
python3 harness.py inspect-checkpoint --checkpoint "$fixture_dir"
python3 harness.py generate --checkpoint "$fixture_dir" --ids 2 11 3 7 9 --new-tokens 16
python3 harness.py verify --checkpoint "$fixture_dir" --oracle full \
  --prompt-lengths 1 6 17 --new-tokens 16 --atol 2e-6 --rtol 2e-6
python3 harness.py verify --tiny --prompt-lengths 1 8 32 128 --new-tokens 12
python3 -m unittest -v test_checkpoint test_deepseek test_deepseek_quant test_deepseek_placement
python3 -m tools.cpu_speculative_bench --checkpoint "$fixture_dir" \
  --new-tokens 16 --seed 29 --output "$report_dir/speculative.json"
python3 -m tools.cpu_parity deepseek --output "$report_dir/parity-dense.json"
python3 -m tools.cpu_parity deepseek --quantized-fixture --output "$report_dir/parity-quantized.json"
python3 -m tools.cpu_parity qwen3-archive --output "$report_dir/qwen3-archive-replay.json"
python3 -m unittest -v
git diff --check
```

Use the pinned config and index to reproduce the published layout and rank plan. The header reader checks HTTP range responses and reads metadata only.

```bash
metadata_dir=$(mktemp -d)
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
source_url="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
curl -fsSL "$source_url/config.json" -o "$metadata_dir/config.json"
curl -fsSL "$source_url/model.safetensors.index.json" -o "$metadata_dir/model.safetensors.index.json"
python3 harness.py inspect-checkpoint --checkpoint "$metadata_dir" --metadata-only
python3 tools/deepseek_headers.py --checkpoint "$metadata_dir" --output "$metadata_dir/headers"
python3 tools/deepseek_placement.py --checkpoint "$metadata_dir" \
  --headers "$metadata_dir/headers" --output "$report_dir/placement.json"
```

The initial managed CPU run recorded 75 tests with 63 passes and 12 skips. The in-process rank tests ran for world sizes 1, 2, 4, and 8. The Gloo test also passed independently in two CPU processes with local loopback and shared-memory access.

```bash
GLOO_SOCKET_IFNAME=lo0 python3 -m unittest -v \
  test_deepseek_placement.DeepSeekPlacementTests.test_two_process_gloo_collectives
```

| Skipped check | Reason |
| --- | --- |
| `test_checkpoint.CheckpointTests.test_native_quantized_storage_loads_from_both_deepseek_shards` | Safetensors is absent and the installed PyTorch lacks native E8M0 |
| `test_deepseek_placement.DeepSeekPlacementTests.test_two_process_gloo_collectives` | Managed sandbox denies loopback. Passed separately with local access |
| `test_cuda_engine.CudaEngineTests.test_bf16_cache_initializes_attention_padding` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_block_commit_and_rollback_match_scalar_cache` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_cached_decode_matches_cpu_and_preserves_independent_requests` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_decode_graph_replays_with_new_tokens_and_reset_cache` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_draft_rejection_preserves_greedy_continuation` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_fused_decode_pointwise_preserves_bf16_rounding` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_fused_rms_norm_preserves_bf16_rounding` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_packed_projections_preserve_prefill_and_decode_logits` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_tiled_block_attention_matches_scalar_across_tile_boundary` | Requires CUDA |
| `test_cuda_engine.CudaEngineTests.test_tiled_decode_masks_padding_and_appends_gqa_cache` | Requires CUDA |

## Prepare prompts and generate

### Inspect a checkpoint

The shared checkpoint reader accepts the published DeepSeek V4.1 Flash config and Safetensors index. Inspect its metadata before downloading the weight shards.

```bash
python3 harness.py inspect-checkpoint --checkpoint checkpoints/DeepSeek-V4.1-Flash --metadata-only
```

`Checkpoint.open(path).read_tensor(name)` reads a local tensor using the native PyTorch dtype. The CPU DeepSeek loader can also interpret stored E8M0 and packed FP4 bytes directly with NumPy.

### Generate from text, messages, or token IDs

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
fixture_dir=$(mktemp -d)
python3 -m deepseek_fixture "$fixture_dir" --with-tokenizer
python3 harness.py generate --checkpoint "$fixture_dir" --text 'Hello, world!' --new-tokens 4
python3 harness.py generate --checkpoint "$fixture_dir" --text-file prompt.txt
python3 harness.py generate --checkpoint "$fixture_dir" --messages messages.json
```

`messages.json` is a JSON array of DeepSeek role and content objects, such as
`[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello, world!"}]`.
Use `--thinking` with an optional `--reasoning-effort` from 1 to 100, or
`low`, `high` or `max`, to use the pinned reasoning format. Text and chat stop at checkpoint EOS and return decoded text,
the tokenizer SHA-256, and the prompt and continuation IDs with their hashes.
The continuation IDs include EOS when emitted. `--ids` replays a fixed number
of steps and works with checkpoints that have no tokenizer.

### Generate from images

Install PyTorch, Pillow, and `tokenizers`, then build the reduced vision
fixture. Use `<image>path.png</image>` in plain text or a structured
`{"type":"image_url","image_url":{"url":"path.png"}}` block in a message.
Both accept local paths and base64 data URLs. Files named inside a prompt
file are resolved relative to that file.

```bash
vision_dir=$(mktemp -d)
python3 -m deepseek_fixture "$vision_dir" --vision
python3 harness.py generate --checkpoint "$vision_dir" \
  --text 'What is shown? <image>fixtures/deepseek-images/signal-red.png</image>' --new-tokens 8
python3 harness.py prepare-prompt --checkpoint "$vision_dir" --messages messages.json
```

`prepare-prompt` also works with just the published `config.json` and
`tokenizer.json`. It emits expanded IDs, per-token image types, image span
positions, grids, source hashes and BF16 patch hashes without loading weights.
Reduced vision generation uses synthetic weights, so it tests pixel input,
embedding replacement, routing, Engram masking and decode state.

## CPU vision checks

The reduced engine accepts local images as actual pixel input. The pinned image
processor forms BF16 patches and expands the image placeholder into start,
patch, newline and end positions. The reduced ViT and aligner replace the text
embeddings for those positions. They use the image router bias, contribute no
Engram vectors, and mark the n-gram history so text after an image starts a new
n-gram. In the same-size image swap test, the IDs and token types match while the
patches, prefill logits and next decode logits differ.

The published checkpoint has a 970,533,624-byte vision-only shard. Its 32-layer
ViT and aligner produced 425 and 176 finite 5,120-wide rows on CPU for the
upstream carrot and corn images at the published resize limit, using both
BF16 tensors and weights expanded to FP32. A separate BF16 check used
a 12-token image cap and produced six finite rows per image. These runs use
published vision weights and stop at the aligner.
[Published resize, BF16](../results/deepseek-cpu/vision-bf16-published-resize.json) · [Published resize, FP32](../results/deepseek-cpu/vision-fp32-published-resize.json) · [Reduced image cap, BF16](../results/deepseek-cpu/vision-bf16-cap12.json)

```bash
vision_dir=$(mktemp -d)
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
source_url="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
for asset in config.json tokenizer.json model-00001-of-00048.safetensors; do
  curl -fsSL "$source_url/$asset" -o "$vision_dir/$asset"
done
for image in carrots.jpeg corn.jpeg; do
  curl -fsSL "$source_url/inference/examples/images/$image" -o "$vision_dir/$image"
done
python3 -m tools.deepseek_vision_probe --checkpoint "$vision_dir" \
  --shard "$vision_dir/model-00001-of-00048.safetensors" \
  --image "$vision_dir/carrots.jpeg" --image "$vision_dir/corn.jpeg" \
  --dtype float32 --output "$vision_dir/vision-report.json"
```

Omit `--dtype float32` for native BF16. Add `--image-token-cap 12` for the
smaller BF16 check.
The probe verifies the config, tokenizer and shard before loading the weights.
The reports include source image, patch and aligned-feature hashes.

## 100-prompt workload

The [manifest](../fixtures/deepseek-prompts.jsonl) contains 100 distinct cases, ten
each for coding, logic and math, multilingual input, multi-turn chat,
extraction, summarization, token boundaries, chat and tool protocol,
long documents, and images. The long documents place facts and plausible
distractors at different points in the repeated passage. Each document's
`facts` entries give the text and its fractional position `at` among the
filler blocks. Edit `target_tokens` to change its length. The compiler measures
the rendered prompt with the tokenizer and leaves space for
`new_tokens`. For an existing document, use `harness.py prepare-prompt
--text-file` with the file path.

| CPU workload | Prompts | Longest input | Generated tokens |
| --- | ---: | ---: | ---: |
| [Published tokenizer and image preparation](../results/deepseek-cpu/prompts-frozen.json.gz) | 100 | 1,048,427 | |
| [Reduced dense model](../results/deepseek-cpu/prompts-smoke-dense.json.gz) | 100 | 512 | 800 |
| [Reduced FP8/FP4 model](../results/deepseek-cpu/prompts-smoke-quantized.json.gz) | 100 | 512 | 800 |
| [Reduced dense model, long continuation](../results/deepseek-cpu/prompt-32743-dense.json.gz) | 1 | 32,743 | 8 |
| [Reduced FP8/FP4 model, long continuation](../results/deepseek-cpu/prompt-32743-quantized.json.gz) | 1 | 32,743 | 8 |

These archived reports predate the revisions to the ten long-document prompts
and two tool prompts. The frozen artifact records the exact prompt IDs, hashes, token budgets,
expected answers, and expanded image spans. It contains 2,102,202 prompt
tokens and 12 image spans. The reduced runs use the published tokenizer
and its real Engram map with synthetic weights. The 100-case runs use a
512-token cap on generated documents. The two long runs use the same
32,743-token text prefix. All reduced runs use an eight-token output cap. The
reports preserve every greedy ID, each evaluated prefix hash, the top five
logits and a digest of the full logit vector. They measure execution and
reproducibility. The manifest's expected answers are for full-checkpoint
evaluation. Swapping two same-size images keeps IDs and token types identical
and changes the first logit vector and greedy sequence in both reduced models.

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
metadata_dir=$(mktemp -d)
report_dir=$(mktemp -d)
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
source_url="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
for asset in config.json tokenizer.json; do
  curl -fsSL "$source_url/$asset" -o "$metadata_dir/$asset"
done
python3 -m tools.deepseek_prompts freeze --checkpoint "$metadata_dir" \
  --output "$report_dir/prompts-frozen.json.gz"
python3 -m deepseek_fixture "$report_dir/reduced" --vision --context 4096 \
  --published-tokenizer "$metadata_dir/tokenizer.json"
python3 -m tools.deepseek_prompts smoke --checkpoint "$report_dir/reduced" \
  --document-cap 512 --output-cap 8 --output "$report_dir/prompts-smoke-quantized.json.gz"
```

Use `--dense` when creating a dense fixture. Use `--case` to select a named
case and `--document-cap` to limit the generated documents for a smaller run.
To reproduce either 32,743-token run, create the fixture with `--context 65536`
and omit `--vision`, then use `--case long-context-million --document-cap 32768
--output-cap 8`.
### Capture and replay reference logits

`replay` checks the prompt IDs, token types, ordered image spans, tokenizer and
EOS IDs. It feeds the reference's greedy choice to the candidate at each step,
even after the candidate chooses a different token or predicts EOS. Step zero
compares logits immediately after prefill. The report gives the first
different full-vector digest, the first different greedy token, and the
competing vocabulary entries and their scores. Margins subtract the competing
token's score from the reference winner's score on each engine. When both
engines choose the same token, the competing token is the reference runner-up.

Use `--capture-logits` to save the complete float32 vectors in a compressed NPZ
alongside the report. Supply that archive as `--reference-logits` to add the
maximum vocabulary error, its token and scores, and the first step beyond
`--atol + --rtol * abs(reference)`. With a report alone, replay can compare
hashes and top-five scores. A reference score outside its saved top five is
unavailable in this mode. Capture writes about 0.49 MiB per generated token
before compression for the published vocabulary.

```bash
python3 -m tools.deepseek_prompts smoke --checkpoint "$report_dir/reduced" \
  --case vision-signal-order-red-blue --output-cap 4 \
  --capture-logits "$report_dir/reference.npz" --output "$report_dir/reference.json.gz"
python3 -m tools.deepseek_prompts replay --checkpoint "$report_dir/reduced" \
  --reference "$report_dir/reference.json.gz" --reference-logits "$report_dir/reference.npz" \
  --output "$report_dir/replay.json.gz"
```

For a published-model reference, install the pinned [upstream inference
requirements](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/inference)
and `jsonschema`, and use its `convert.py` to prepare the tensor-parallel shards.
The capture command verifies the upstream inference files and the published
config and tokenizer artifacts. Give `--checkpoint` the original local
checkpoint or a metadata directory with the published `config.json`,
`tokenizer.json`, and `tokenizer_config.json`.

```bash
torchrun --nproc-per-node "$MP" -m tools.deepseek_prompts upstream \
  --checkpoint "$HF_CKPT_PATH" --upstream-source "$UPSTREAM_ROOT/inference" \
  --converted-checkpoint "$CONVERTED_CKPT_PATH" --case coding-aliasing \
  --capture-logits "$report_dir/upstream.npz" --output "$report_dir/upstream.json.gz"
```

The command reads the logits returned by the pinned
[`Transformer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
after one full prefill and each single-token decode. Rank zero writes the
report. Use `--case` or `--document-cap` to fit the prefill in GPU memory.
For a numerical parity measurement, replay with a candidate using the same
weights and input preprocessing. The CPU loader currently supports reduced
checkpoints up to 512 MiB of stored tensors.

### Score answers and run CPU CI

Upstream reports include per-case answer scores and an aggregate pass count.
The default checker compares exact text after trimming surrounding whitespace.
Structured checks validate JSON against the case schema and compare the
values, including array and image order. Tool checks parse the DeepSeek tool
call and compare its qualified name and typed arguments. Thinking cases score
the answer after the reasoning delimiter. Free-form summaries use exact
reference wording. CPU smoke runs can exercise the same checkers with
`--score`. Replay predictions are conditioned on the reference continuation.

The [CPU workflow](../.github/workflows/cpu.yml) runs on pushes to `main` and
pull requests. It requires native E8M0 reads, two-process Gloo, parity with
the pinned tokenizer and prompt fixtures, and reduced vision generation.
Trigger **CPU inference** manually in GitHub Actions to freeze all 100 prompts
and run dense and quantized smoke tests. Both smoke runs cap generated
documents at 512 tokens and each continuation at eight tokens. The workflow
retains the reports as `deepseek-cpu-reports` for 30 days.

## Compare the pinned upstream CPU paths

The differential checker loaded 18 pinned assets and verified their SHA-256
digests. All five upstream prompt fixtures matched byte for byte, including
their token IDs and media order. Both upstream JPEGs matched exactly in BF16
patch bytes, patch grids, image types and expanded IDs. All 129,280 token
mappings matched the upstream Engram implementation and compressed to 99,092
entries. Primes, offsets and multipliers matched. Across six inputs, all
95,664 hashes matched for both full-prefix and split prefill/decode against
upstream, including masked image positions. [Differential report](../results/deepseek-cpu/upstream-checks.json)

```bash
upstream_dir=$(mktemp -d)
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
source_url="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
python3 -m tools.deepseek_upstream_checks --list-files | while IFS= read -r asset; do
  mkdir -p "$upstream_dir/$(dirname "$asset")"
  curl -fsSL "$source_url/$asset" -o "$upstream_dir/$asset" || exit 1
done
python3 -m tools.deepseek_upstream_checks --upstream "$upstream_dir"
```

With PyTorch 2.15.0, Tokenizers 0.22.2, Safetensors 0.7.0 and the pinned metadata,
83 of 93 unit tests passed with ten CUDA skips. The native E8M0 sharded load and
two-process Gloo tests both passed on CPU. The regenerated dense and quantized
parity reports and the Qwen3 CPU archive report matched the prior reports.

The pinned Qwen3-0.6B weights also ran on CPU through the new text entry point.
The prompt `The capital of France is` produced ` Paris. The capital of Italy
is Rome` in eight greedy tokens. Replaying its five input IDs returned all
eight output IDs and the same hashes. [Qwen3 text smoke and checkpoint hashes](../results/deepseek-cpu/qwen3-text-smoke.json)

## GPU validation pending

- Load and execute the full pinned text checkpoint on the selected topology. Validate gated Engram embeddings, compressed and index state, and greedy logits on the frozen prompts against the pinned DeepSeek runtime. The CPU reference caps materialized storage at 512 MiB. Full multimodal language generation remains a GPU validation case.
- Compare device FP8/FP4 linears and expert outputs with the CPU golden vectors, including activation scaling, saturation and ties, BF16 boundaries, compressed KV with E4M3 scales, and indexed FP4 with E8M0 scales. Establish device tolerances from measured error.
- Run real NCCL collectives and compare single-rank and multi-rank arithmetic on reduced checkpoints before validating full-weight rank layouts. Check routed experts, vocabulary gather, group output sum, Engram padding, and index score reduction.
- Measure peak memory during load, prefill, decoding and graph capture. Validate raw-window rollover and compressed/index-cache growth at longer contexts through the published limit on an appropriate host. Measure allocator, kernel and collective workspace beyond the modeled buffers.
- Capture scalar and block CUDA graphs, then replay across compression boundaries and with rejected suffixes, EOS, padded slots, and consecutive requests. Compare every valid verifier row with scalar target logits on the same prefix.
- Validate all DSpark stages on the full checkpoint, target feature taps, fixed-block behavior, Markov/confidence outputs, accepted-prefix rollback and actual proposal acceptance. Measure TTFT, token gaps, draft and verifier timing, and throughput only on the selected hardware.
- Replay the archived Qwen3 disagreements and keep oracle tolerance distinct from exact sequence agreement. The historical BF16 tolerance is `atol=0.5, rtol=0.025`. The CPU archive report reproduces all six prompt hashes and the frozen step-0 prefix hash. The other six hashes need the original scalar continuation IDs, which earlier artifacts omitted. GPU replay now writes the scalar IDs and their digest. [Archive and replay inputs](../results/deepseek-cpu/qwen3-archive-replay.json)

| Qwen3 case | Zero-based step | GPU replay target |
| --- | ---: | --- |
| Frozen 128 input | 0 | SGLang 13 from its exact tie and CUDA/HF 315 |
| Frozen 128 input | 16 | CUDA 576 versus HF BF16 4710, rank 2 and margin 0.0625 |
| Frozen 128 input | 37 | Packed 2621 versus tiled and HF BF16 13 |
| Held-out seed 711, 1,024 input | 91 | 7 BF16 and 495 FP32 logit violations |
| Held-out seed 711, 1,024 input | 100 | 13 BF16 and zero FP32 logit violations |
| Held-out seed 711, 4,096 input | 15 | 3 BF16 and zero FP32 logit violations |
| Held-out seed 711, 4,096 input | 21 | 1,852 BF16 and 43 FP32 logit violations |

The Qwen3 checkpoint is pinned to `167b8104f88905a951069f5f95f9776908da5f68`. Use the existing Qwen3 CUDA stack and write new output paths. The CPU archive command checks each reconstructed prefix against its original hash whenever the replay reproduces the recorded CUDA scalar continuation. The tiled path is the scalar reference for both replay invocations. Step 37 also needs the packed scalar variant on the same prefix. SGLang's own logits are needed to replay the step-0 tie.

```bash
python3 -m unittest -v test_cuda_engine
python3 -m tools.speculative_parity --checkpoint checkpoints/Qwen3-0.6B \
  --verification tiled --output /tmp/qwen3-replay-tiled.json
python3 -m tools.speculative_parity --checkpoint checkpoints/Qwen3-0.6B \
  --verification flash --output /tmp/qwen3-replay-flash.json
python3 -m tools.cpu_parity qwen3-archive --replay-input /tmp/qwen3-replay-tiled.json \
  --output /tmp/qwen3-archive-replay.json
```

The DeepSeek CUDA backend, full-size checkpoint execution, and the hardware-matched Baseten comparison remain subsequent work.

# Vaidylang handoff for DeepSeek CPU scaffolding

Prepared November 17, 2026, America/Detroit.

## Assignment and completion boundary

Implement the CPU scaffolding for the four remaining work items below. The user is deferring GPU use until this work is done. Carry the work through runnable CPU implementations, regression tests, a reduced model that generates tokens, and a precise report of the later GPU checks.

1. DeepSeek V4.1 Flash model execution and architecture dispatch.
2. Quantized weight execution and distributed placement.
3. Speculative draft integration and acceptance measurement.
4. Numerical parity investigation and broader regression coverage.

Execute this phase on local CPU resources. GPU provisioning, GPU-host SSH sessions, CUDA execution, and live GPU benchmarks are deferred. CPU PyTorch, NumPy, small synthetic checkpoints, and pinned upstream source/config downloads are sufficient for the immediate work. Use small bounded tensor downloads only where they add evidence. The published weight index describes 510,286,023,000 bytes, so full-checkpoint allocation and download belong to a later resource plan.

In this assignment, scaffolding includes working reference computations on reduced dimensions. Every feature covered by the CPU completion checklist needs exercised numerical behavior. A dispatcher, class skeleton, mocked forward pass, or successful metadata inspection alone leaves that feature unfinished. Hardware-specific performance and full-checkpoint validation remain explicitly pending after the CPU work.

## Starting point

- Repository `https://github.com/nikhil-lalgudi/vaidylang.git`.
- Local checkout `/Users/nikhilvaidyanathan/Documents/ChatGPT/inference`.
- PR #3 is merged into `main` at `00c7e17cc2efd33afb9d01b21ef92f02bd460c93`.
- Its feature branch ended at `1713b4eed73a044481bef6357fecc4a9c94a14e4`.
- This handoff is on `codex/deepseek-cpu-handoff`, based on that merged `main`. Inspect the current branch and worktree before beginning. Continue the handoff branch or create another `codex/` branch from its tip.
- The user's standing instruction is to commit as work progresses. Commit each coherent implementation after its appropriate CPU checks pass. Preserve existing benchmark artifacts as historical evidence.

The local baseline suite currently runs 29 tests with 11 dependency/CUDA skips. Those skips identify unavailable checks and contribute no evidence about GPU execution. Earlier H100 testing passed all 29 tests, and Qwen3 BF16 forced/greedy parity passed four output tokens at 128, 1,024, and 4,096 input tokens after the reader change. That evidence predates this assignment.

Existing commands that work without a GPU are below.

```bash
cd /Users/nikhilvaidyanathan/Documents/ChatGPT/inference
git status --short --branch
python3 -m unittest -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 harness.py verify --tiny \
  --prompt-lengths 1 8 32 128 --new-tokens 12
git diff --check
```

## Source pins and model facts

Use DeepSeek's model repository at `314113f8a7dadb388215396ba5fbee2c610a04d6` as the architecture and encoding reference. Record the revision and hashes of the source files used to derive tests. Check source licenses and preserve attribution when adapting code.

- [Published config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json)
- [Safetensors index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/model.safetensors.index.json)
- [Reference model and DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
- [Reference quantization and attention kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py)
- [Reference conversion and rank partitioning](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/convert.py)
- [Engram hashing and state](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/engram.py)
- [Prompt encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding/encoding.py)

The published root `model_type` is `deepseek_v41` and the nested text type is `deepseek_v41_text`. The root architecture is `DeepseekV41ForCausalLM`. The manifest contains 96,085 tensors across 48 shards. The text config specifies vocabulary 129,280, hidden size 5,120, 40 main layers, 64 attention heads, head dimension 512, 384 routed experts, six active experts per token, a 128-token raw attention window, and maximum context 1,048,576. Read the remaining schedules and dimensions directly from the pinned config. Some schedules also cover draft layers, so derive schedule validation from the reference implementation.

Quantization uses FP8 E4M3 weights with E8M0 scale tensors and packed FP4 expert weights stored as `I8`. The nested text config includes compression/indexing schedules, Engram configuration, hyper-connection configuration, and DSpark settings. The checkpoint also contains vision tensors. This handoff implements text generation and identifies the vision tensor namespace during loading. Image execution can be tracked separately.

The upstream reference imports CUDA-specific kernels. Obtain independent CPU expectations by translating the relevant reference formulas into small CPU reference functions, or by using pinned precomputed reference fixtures. Document each derivation. A default CUDA device or an upstream GPU-only import is a dependency to isolate while continuing CPU implementation.

## What already exists

| File or artifact | Existing behavior and relevance |
| --- | --- |
| `checkpoint.py` | `Checkpoint.open(path, metadata_only=True)` accepts the published nested config and shard index. `read_tensor(name, device="cpu")` reads one tensor with its stored dtype. It resolves names and files without interpreting model architecture. |
| `engine.py` | NumPy Qwen3 implementation, `Config`, `weight_shapes`, tiny weights, and CPU checkpoint entry point. `Config.from_mapping` remains Qwen3-specific. |
| `cuda_engine.py` | CUDA Qwen3, scalar `DecodeGraph`, and `BlockDecodeGraph`. Its `from_checkpoint` still calls Qwen3 `Config.from_mapping`. DeepSeek inference currently fails there. |
| `cuda_attention.py`, `cuda_kernels.py` | Qwen3 tiled attention and fused pointwise kernels. Importing these modules requires GPU-oriented dependencies. |
| `speculative.py` | CPU n-gram proposal and greedy acceptance controller. The current runtime expects CUDA graph objects. It commits the verified cache prefix and leaves rejected suffix storage outside the visible prefix. |
| `speculative_bench.py` | Six coding/chat prompts, target-only comparison, block sizes 2/4/8, raw IDs, draft/verify times, emitted tokens per cycle, TTFT, and streaming gaps. It currently requires Qwen3 and CUDA. |
| `speculative_learned_draft.py`, `tools/train_speculative_draft.py` | Small CPU embedding/MLP draft and its Qwen3 GPU data-generation path. Historical weights and metrics are committed. |
| `tools/speculative_parity.py` | Existing identical-prefix Qwen3 CUDA, HF BF16, and HF FP32 diagnostic. It requires CUDA to rerun its current model paths. |
| `test_checkpoint.py` | Config/index recognition, missing shards, path validation, and native I8/E8M0 storage tests. |
| `results/speculative/` | Current evidence for draft acceptance, exact continuation parity, oracle differences, and failed cuBLASLt experiment. |

Build the architecture selection above the architecture-specific engines. Preserve Qwen3's config invariants and route DeepSeek through a DeepSeek config and implementation. The checkpoint reader is reusable for both architectures.

## Work item 1 — DeepSeek execution on CPU

Start with a public engine-loading entry point that inspects `Checkpoint.model_type` and selects the correct engine. Wire it into a normal CLI or harness path so a reduced DeepSeek checkpoint loads and generates through the same user-facing path that selects Qwen3. Keep `prefill(token_ids, capacity)` and `decode(token_id, state)` compatible with the existing reference-engine contract where practical.

Implement a separate DeepSeek configuration that retains the root config, nested text config, quantization metadata, and schedules. Map public checkpoint tensor names using the reference conversion/model code. Validate missing tensors and incompatible shapes with tensor-specific errors. A valid DeepSeek fixture must reach its own engine constructor.

Implement the text forward pass with CPU tensor operations and a deliberately small, deterministic model. Reduce vocabulary, hidden sizes, layer count, experts, memory tables, and context length together. Keep the attention compression/indexing branches, hyper-connections, Engram paths, and draft paths available to tests through coherent reduced configurations. Validate the topology against the published reference instead of replacing these computations with ordinary dense Qwen attention.

Required numerical components are normalization and rotary embeddings, compressed sparse attention and indexing, shared attention caches, expert gating and weighted expert computation, hyper-connection mixing, Engram token mapping/hash state/lookups, and final logits. Establish primitive tests before composing the complete model. Test at compression boundaries, raw-window rollover, cache capacity, and source-layer cache sharing.

**CPU acceptance**

- [ ] The shared loader resolves both a Qwen3 fixture and a DeepSeek fixture through the real user-facing entry point.
- [ ] The reduced DeepSeek model emits finite logits and at least 16 deterministic greedy tokens.
- [ ] Incremental prefill/decode agrees with full-prefix recomputation under explicit, recorded CPU tolerances.
- [ ] Independent numerical tests exercise each architectural component and its state transitions.
- [ ] Interleaved requests keep separate cache/hash state, and prefill reset reproduces a fresh request.
- [ ] CPU import and execution succeed with CUDA hidden and optional CUDA kernels unavailable.

## Work item 2 — Quantized weights and distributed placement

Extend tensor reading into usable CPU quantized computation. Implement FP8 E4M3 and E8M0 interpretation and FP4 decoding from signed storage bytes. Derive nibble order, FP4 value table, scale expansion, and block geometry from the pinned conversion/kernel code. Include activation quantization where the reference forward requires it. Document storage shape and logical shape separately.

Use small CPU buffers and FP32/FP64 reference arithmetic to implement dequantization and matrix multiplication. Preserve quantization rounding, saturation, scale selection, and grouping that affect the reference outputs. Where a CPU PyTorch build lacks native float8 support, provide reference decoding using the stored byte representation so the new correctness tests can still run. Record native dtype support separately from byte-decoder coverage.

Implement rank placement and tensor/expert partitioning as real CPU operations. For small tensors, construct per-rank shards, execute the local computations, and combine outputs using the reference reduction/gather semantics. Cover world sizes 1, 2, 4, and 8 with in-process CPU partition tests. Exercise an actual two-process CPU collective path with Gloo when the local build supports it. Keep unavailable collective coverage explicit while completing the in-process arithmetic tests.

Plan expert ownership, embedding/head partitions, replicated tensors, Engram placement, and shared KV/index state from the reference code. Estimate weight, scale, scratch, and cache bytes from metadata without allocating full published shapes. The output should make later GPU placement and memory requirements calculable.

**CPU acceptance**

- [ ] Golden byte fixtures cover FP4 nibble order and signs, FP8 values, E8M0 scales, zero/saturation behavior, and partial blocks.
- [ ] Quantized linear and expert outputs agree with independently derived small dequantized reference outputs.
- [ ] A reduced sharded checkpoint containing quantized tensors loads through the engine entry point and performs a forward pass.
- [ ] Rank-local slices reconstruct the original tensors, and combined rank computations reproduce the single-rank result within justified reduction tolerances.
- [ ] Placement and memory reports cover the full published config using metadata, with every required tensor assigned.
- [ ] A missing shard, missing scale tensor, invalid scale geometry, or invalid rank assignment fails with an actionable error.

This completes CPU quantization semantics and distributed placement arithmetic. Optimized GPU quantized kernels, real multi-GPU collectives, device memory behavior, and throughput require the later GPU phase.

## Work item 3 — Draft integration and acceptance

Use the published DSpark implementation as the DeepSeek draft reference. Its model file includes draft attention, a Markov head, confidence head, and draft blocks. Inspect how the draft consumes target-layer features and previous tokens. Implement those computations on the reduced CPU model and connect them to speculative verification. A useful small reference preserves the dependency on actual target hidden states.

Generalize the existing acceptance/controller contract enough to support the CPU target engine and its cache. Keep the verified-prefix invariant explicit. All state advanced by a staged draft needs transaction semantics, including compressor/index state, shared KV data, Engram history, and draft state. Logical KV position alone may be insufficient for DeepSeek rollback. Test staged failure recovery as well as ordinary rejection.

Run block sizes 2, 4, and 8 where supported by the reduced reference. Distinguish the verifier input block from the number of proposed tokens. Respect any DSpark fixed-block requirement from the published config and record a mapping or separate supported case rather than silently resizing trained heads. Include zero acceptance, partial acceptance, complete acceptance plus bonus token, short output budgets, EOS, context exhaustion, and consecutive requests.

Retain n-gram and learned-draft CPU comparisons. Separate test/training sequences and measure the actual proposal acceptance. CPU timing can expose controller overhead and establish repeatable behavior. Hardware throughput targets are evaluated later.

**CPU acceptance**

- [ ] The DSpark components produce deterministic draft logits/tokens on a reduced fixture and match independent reference calculations.
- [ ] Target-only and speculative generation emit exactly the same greedy continuation for each deterministic CPU fixture.
- [ ] Every accepted token is target-verified, and rejection restores every affected state component before the next token.
- [ ] Tests cover the supported block sizes, partial commit, full rollback, bonus token, EOS, and request reset.
- [ ] A CPU report records proposal count, accepted count, emitted tokens per cycle, draft time, verification time, raw output IDs, and first mismatch.
- [ ] The report identifies the synthetic/reduced model and CPU device. It identifies any DSpark behavior deferred for GPU validation.

Historical H100 measurements establish why acceptance needs work. On the six-prompt Qwen3 suite, target-only decode reached about 806 tokens/s and tiled n-gram block 8 reached 1,121 tokens/s. Block 8 emitted 1.52 tokens per cycle across all cycles and 2.12 per verification cycle. Median drafting was about 0.004 ms and verification about 1.433 ms. The small learned draft reached roughly 775–781 tokens/s. These remain Qwen3/H100 observations. [Recorded experiment summary](results/speculative/summary.json)

## Work item 4 — Numerical parity and regression coverage

Keep three separate comparisons in the reports: speculative versus scalar target on an identical prefix, target versus an independent numerical reference, and strict agreement with a frozen engine's output sequence. Each comparison must identify the dtype, arithmetic path, prefix IDs/hash, selected IDs, oracle rank, logit error, and tolerance violations.

The existing exact-tie fix accepts a CUDA token only when it attains the oracle's exact maximum and all logits pass the existing tolerance. Preserve exact output hashes and the distinction between an exact maximum and a near tie. New reference tests need independently justified tolerances. Historical results retain the tolerances under which they were collected.

Reproduce the known scenarios first on CPU fixtures, including cache sharing, compression boundaries, position/rotary offsets, quantization rounding, masked padding, and speculative rollback. Add intermediate tensor snapshots at the first observed mismatch. The Qwen3 oracle tolerances are `atol=0.5` and `rtol=0.025` in the archived BF16 reports.

The recorded Qwen3 cases requiring later GPU replay are below. Step numbers are zero-based output-token indices.

| Case | Step | Recorded observation |
| --- | --- | --- |
| Frozen 128-input fixture | 0 | SGLang selected 13 from an exact tie. CUDA selected 315, also selected by HF FP32 and HF BF16 in the diagnostic. |
| Frozen 128-input fixture | 16 | CUDA selected 576. HF BF16 selected 4710, with CUDA's choice ranked second and 0.0625 below the HF maximum. |
| Frozen 128-input fixture | 37 | Packed CUDA selected 2621. Tiled CUDA and HF BF16 selected 13 on the common prefix. |
| Held-out seed 711, 1,024 input tokens | 91 | Existing scalar comparison exceeded tolerance for 7 logits against HF BF16 and 495 against HF FP32. |
| Held-out seed 711, 1,024 input tokens | 100 | Existing scalar comparison exceeded tolerance for 13 logits against HF BF16 and zero against HF FP32. |
| Held-out seed 711, 4,096 input tokens | 15 | Existing scalar comparison exceeded tolerance for 3 logits against HF BF16 and zero against HF FP32. |
| Held-out seed 711, 4,096 input tokens | 21 | Existing scalar comparison exceeded tolerance for 1,852 logits against HF BF16 and 43 against HF FP32. |

The tiled verifier already matched scalar CUDA logits bit for bit on all 84 checked block rows, while FlashAttention verification matched 80 of 84 selected tokens and zero full logit vectors. This establishes agreement between the tiled paths on those rows. Oracle tolerance failures still need diagnosis. [Identical-prefix replay](results/speculative/parity-tiled.json) · [Frozen 128-token diagnostic](results/cuda-tuning/final-tiled/frozen-128-output-diagnostic.json) · [SGLang tie evidence](results/sglang-output-diagnostic.json)

**CPU acceptance**

- [ ] New reduced DeepSeek and quantization tests compare against an independently derived reference.
- [ ] A CPU diagnostic reports the first differing layer/operation and supports full-prefix, scalar-cache, and block-cache execution on the same tokens.
- [ ] Stored Qwen3 report inputs and prefix hashes remain reproducible by the diagnostic tooling without a GPU.
- [ ] Numerical regression fixtures exercise exact ties, near ties, finite/nonfinite values, and all relevant state boundaries.
- [ ] Full-suite CPU results list the names and reasons for skips. The new reduced-model CPU acceptance cases run.
- [ ] The original CUDA/HF discrepancy cases are listed as pending GPU replay unless independently resolved with sufficient numerical evidence.

## Implementation order and repository integration

1. Read the pinned architecture and conversion sources. Write the tensor-name/config map and independent small numerical fixtures first.
2. Add DeepSeek configuration and architecture dispatch. Prove the real loader reaches the CPU engine while Qwen3 regression checks continue to pass.
3. Implement reference primitives and a complete reduced text model. Exercise prefill/decode and complete request-state ownership.
4. Implement byte decoding, quantized operations, rank placement, and CPU partition/collective tests. Connect the quantized reduced checkpoint to the model.
5. Implement reduced DSpark drafting and complete-state speculative transactions, then run the CPU acceptance comparison.
6. Finish the parity diagnostics and CPU regression suite. Record unresolved hardware questions and the later GPU commands supported by the implementation.
7. Commit the verified changes and report the final CPU-ready status with evidence.

Reuse `Checkpoint` for storage, the existing engine `prefill`/`decode` convention for model execution, and the existing benchmark report fields for request timing and output identity. `harness.checkpoint_identity` and the Qwen3 benchmark pin currently assume one `model.safetensors` file. Introduce a deterministic sharded identity for the new path while preserving the existing Qwen3 identity values and archived comparisons.

CPU utilities should import successfully without Triton, TileLang, SGLang kernels, or CUDA availability. Keep GPU-specific imports inside the later GPU backend. Inspect CPU PyTorch and Safetensors capabilities before choosing native float8 tests. The original Mac environment had PyTorch 2.4.1 and lacked Safetensors; use an isolated CPU environment if needed and record versions. Hardware-independent storage/quantization fixtures should still exercise real arithmetic.

The following existing paths provide the main integration points. New module names are implementation choices that should follow the responsibilities below.

| Responsibility | Existing integration points |
| --- | --- |
| Checkpoint metadata and tensor storage | `checkpoint.py`, `test_checkpoint.py` |
| Architecture dispatch and CPU model | `engine.py`, `harness.py`, plus dedicated DeepSeek modules |
| Quantization and placement | New reference modules, `Checkpoint.group_tensors`, model tensor-name mapping |
| CPU target and draft transactions | `speculative.py`, separate reduced-model state tests |
| Draft reference and metrics | `speculative_learned_draft.py`, `speculative_bench.py`, dedicated DSpark reference |
| Numerical replay and comparison | `tools/speculative_parity.py`, CPU diagnostic entry point, deterministic fixtures |

## Definition of CPU scaffolding done

The next agent may report this phase ready when all CPU acceptance items above have concrete evidence and all changes are committed. Its final handoff must include the following.

- Commit SHAs, changed modules, and the pinned source revisions.
- The exact CPU dependency versions and commands for reproducing loader inspection, reduced-model generation, quantized operations, distributed placement, speculative verification, and parity diagnostics.
- Test counts, all skipped checks by name/reason, and reports with actual numerical outputs and mismatch locations.
- A memory/placement plan for the published checkpoint derived from its metadata.
- Every remaining GPU-specific check, including full-size checkpoint execution, FP8/FP4 device arithmetic, real multi-GPU behavior, long-context memory, CUDA graph capture, and the existing Qwen3 oracle mismatches.
- A concise statement of whether the complete reduced CPU model generates correctly, whether CPU speculation preserves its target output, and which features still need full-checkpoint validation.

Finish the CPU work before requesting GPU access. This handoff covers items 1–4. The Baseten hardware-matched head-to-head is the subsequent task and remains pending.

---
name: cpu-validation
description: Validate VaidyLang CPU inference, prompt handling, cache rollback, or reference-logit replay. Use after changes to CPU execution or when reproducing CPU parity failures.
---

# VaidyLang CPU validation

Run commands from the repository root. Use the development image or match the
[CPU workflow's dependency pins](../../.github/workflows/cpu.yml) when checking
native E8M0 storage, Gloo collectives, or reduced vision.

## Choose a check

- Start with the affected test module. For cache rollback use
  `python -m unittest -v test_deepseek_draft test_speculative_cpu`. For the
  prompt compiler, answer scoring, or reference logits use
  `python -m unittest -v test_deepseek_prompts test_deepseek_reference`.
- For a quick forward-pass check, run the deterministic Qwen3 fixture. It
  compares cached decode with full-prefix recomputation.

  ```bash
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python harness.py verify --tiny \
    --prompt-lengths 1 8 32 128 --new-tokens 12
  ```

- For cross-cutting changes, run `python -m unittest -v` with
  `CUDA_VISIBLE_DEVICES=''`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.
  Check skipped tests as well as failures. The published tokenizer test
  requires `VAIDYLANG_DEEPSEEK_METADATA` to point at the pinned assets.
- For DeepSeek tokenizer, vision, or corpus changes, use the
  [CPU reference guide](../../docs/deepseek-cpu-reference.md) to fetch pinned
  assets or regenerate workload reports. Write new reports outside the
  committed archives.

## Interpret the evidence

In logit replay, check the prompt and image hashes first. Compare both engines
on the reference continuation even after they choose different tokens. Report
the first logit and token divergences separately. Use a full-logit capture
when the tolerance or a score outside the saved top five matters. A reduced
checkpoint run verifies execution with synthetic weights. Published-weight
parity and answer quality require evidence from the published checkpoint.

# DeepSeek V4.1 Flash rank placement

The [placement report](placement.json) reads 10,684,928 JSON header bytes from 48 shards at the pinned release. Exact HTTP ranges fetch the Safetensors length prefixes and headers. The report records their SHA-256 hashes and accounts for all 96,085 tensors and 510,286,023,000 tensor bytes. The loader's 48,187 logical text tensor shapes expand to 95,776 storage entries with scales. The other 309 entries belong to vision, including alternate router biases. [Pinned model files](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)

| Ranks | Text runtime weights per rank | E8M0 scales per rank | Persistent state per rank | Text weights, scales and state per rank |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 488,616,735,048 B | 23,561,606,160 B | 3,907,398,464 B | 516,085,739,672 B |
| 2 | 245,663,267,528 B | 11,781,930,760 B | 3,907,398,464 B | 261,352,596,752 B |
| 4 | 124,186,533,896 B | 5,892,093,064 B | 3,907,398,464 B | 133,986,025,424 B |
| 8 | 63,448,167,080 B | 2,947,174,216 B | 3,907,398,464 B | 70,302,739,760 B |

The table uses batch size 1, context 1,048,576, and the reference runtime parameter dtypes. Conversion promotes grouped output weights to BF16, removes their scales, and uses FP32 for the output and Markov heads, confidence projection, and compressor projections with compression ratios above one. Optional replicated vision adds 970,599,936 B per rank. [Reference conversion](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/convert.py)

Vocab embeddings, output heads and per-head projections split along output rows. The grouped output down-projection splits along its input columns. Contiguous expert ranges have a single owner per layer. Each rank uses a ceil-sized Engram row shard and pads any tail with zero weight and unit scale. Raw window KV is separate for each layer and replicated across ranks. Compressed KV and index keys are replicated across ranks only at their source layers, and subsequent layers share the source caches. Partial compressor groups, rotary tables and Engram history contribute to the persistent-state estimate. [Reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)

The report estimates selected scratch buffers for a decode query, including the raw/latent KV concatenation and index score buffers. Their sizes depend on the query chunk, batch size and context. Use `--query-tokens` to model a prefill chunk. GPU measurement is needed for peak allocator use, GEMM and collective workspace, loading and graph capture.

## CPU arithmetic

- FP8 weights store one byte per value with one E8M0 byte per 32×32 block. Engram uses one scale per 32 values in each row. Packed FP4 expert rows store two E2M1 values per signed byte, low nibble first, and one E8M0 scale per 32 elements of the logical reduction dimension.
- Both quantized GEMMs round blockwise input to E4M3FN with power-of-two E8M0 scales. Raw KV and the grouped output input also use FP8 groups of 32. Index keys and queries use E2M1 groups of 32 with E8M0 scales. Compressed KV uses E2M1 groups of 16 with E4M3FN scales. The CPU helpers preserve saturation and nearest-even rounding, and apply each final partial block's scale to its present elements. [Reference kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py)
- Independent golden tests cover all E2M1 nibbles, E4M3FN subnormal and maximum values, scale endpoints, midpoint rounding, saturation, and partial blocks. Weighted expert tests use routing weights before the down-projection. In-process rank tests use world sizes 1, 2, 4 and 8. Storage reconstruction, vocab/Engram lookup and the index-score fixture match exactly. Quantized linear reduction uses `atol=3e-6, rtol=2e-6` for FP32 reassociation of up to 32 products. The weighted-expert test uses `atol=2e-6, rtol=2e-6` against a float64 scalar SiLU expectation.
- The NumPy FP8 decoder matched native PyTorch CPU for all 256 E4M3FN codes. PyTorch 2.4.1 on this arm64 host lacks native E8M0 and packed FP4 dtypes. The NumPy byte fixtures exercise both formats.
- The two-process PyTorch 2.4.1 Gloo test passed with local loopback and shared-memory access. It covers the input-column sum, output-head gather and masked embedding sum. The managed sandbox denies loopback and records an explicit skip. [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)

## Reproduce

```bash
python3 -m unittest -v test_deepseek_quant test_deepseek_placement
GLOO_SOCKET_IFNAME=lo0 python3 -m unittest -v test_deepseek_placement.DeepSeekPlacementTests.test_two_process_gloo_collectives
python3 tools/deepseek_headers.py --checkpoint /private/tmp/vaidylang-v41 --output /private/tmp/vaidylang-v41/headers
python3 tools/deepseek_placement.py --checkpoint /private/tmp/vaidylang-v41 --headers /private/tmp/vaidylang-v41/headers --output results/deepseek-cpu/placement.json
```

The metadata directory contains `config.json` and `model.safetensors.index.json` from release revision `314113f8a7dadb388215396ba5fbee2c610a04d6`. The Gloo interface name `lo0` is specific to the Mac host.

# DeepSeek V4.1 Flash CPU reference

The text topology and storage encodings are derived from the DeepSeek V4.1 Flash [release repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6), revision `314113f8a7dadb388215396ba5fbee2c610a04d6`. DeepSeek's MIT license is retained in `LICENSE.DeepSeek`. The CPU implementation uses NumPy reference arithmetic on reduced buffers.

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

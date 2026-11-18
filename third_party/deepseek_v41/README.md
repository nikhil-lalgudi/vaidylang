# DeepSeek V4.1 Flash upstream source

`encoding.py` is the unmodified [prompt encoder from revision 314113f8a7dadb388215396ba5fbee2c610a04d6](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding/encoding.py).
Its SHA-256 is `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`.
See [LICENSE.DeepSeek](../../LICENSE.DeepSeek).

`vision.py` is the unmodified [ViT and aligner from the same revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/vision.py).
Its SHA-256 is `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c`.

`image_processor.py` is the [upstream image processor](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/image_processor.py)
with a package-relative import and a one-pixel minimum for extreme
aspect ratios that would otherwise round to an empty axis during padding.
The unmodified upstream SHA-256 is
`482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272`.

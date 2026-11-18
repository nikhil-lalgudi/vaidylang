"""Generate deterministic dense or natively byte-encoded reduced DeepSeek shards."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from checkpoint import Checkpoint
from deepseek_config import DeepSeekConfig, weight_shapes
from deepseek_engram import bucket_layout
from deepseek_quant import quantized_shapes
from deepseek_weights import quant_layout, scale_name


def fixture_config(quantized=True, *, vision=False):
    text = {
        "model_type": "deepseek_v41_text", "hidden_act": "silu",
        "vocab_size": 96, "hidden_size": 32, "moe_intermediate_size": 32,
        "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 1,
        "head_dim": 16, "qk_rope_head_dim": 8, "q_lora_rank": 16,
        "o_lora_rank": 8, "o_groups": 8, "n_routed_experts": 8, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "max_position_embeddings": 128,
        "sliding_window": 4, "compress_ratios": [0, 2, 2, 1, 1, 1, 0, 0, 0],
        "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 4],
        "index_n_heads": 8, "index_head_dim": 16, "index_topk": 3,
        "candidate_source_layer_id": 3, "candidate_topk_blocks": 2, "candidate_block_size": 2,
        "compress_rope_theta": 160000.0, "rope_theta": 10000.0,
        "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 4, "beta_slow": 1,
                         "original_max_position_embeddings": 20},
        "rms_norm_eps": 1e-20, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
        "hc_mult": 4, "hc_sinkhorn_iters": 5, "hc_eps": 1e-6,
        "engram_layer_ids": [0, 3], "engram_num_embeddings": [1, 1],
        "engram_max_ngram_size": 3, "engram_vocab_size": 11, "engram_n_heads": 2,
        "engram_head_dim": 16, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 72,
        "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 95,
        "dspark_target_layer_ids": [3, 4, 5], "dspark_markov_rank": 8,
        "dspark_n_routed_experts": 8, "dspark_num_experts_per_tok": 2,
    }
    root = {
        "model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
        "eos_token_id": 1, "pad_token_id": 2, "bos_token_id": 0, "text_config": text,
    }
    if quantized:
        root["quantization_config"] = {
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4",
        }
    cfg = DeepSeekConfig.from_mapping(root)
    text["engram_num_embeddings"] = [int(primes.sum()) for primes in bucket_layout(cfg).values()]
    if vision:
        root["image_token_id"] = 94
        root["vision_config"] = {
            "model_type": "deepseek_v41_vision", "num_hidden_layers": 2, "hidden_size": 16,
            "num_attention_heads": 2, "intermediate_size": 24, "patch_size": 8,
            "rope_theta": 10000, "downsample_ratio": 2, "max_image_tokens": 12,
            "min_pixels": 64, "max_wh_ratio": None,
        }
    return root


def write_safetensors(path, tensors):
    """Write a reduced fixture directly so float8 dtype support is unnecessary."""
    header, data = {}, bytearray()
    for name, (code, tensor) in tensors.items():
        start = len(data)
        data.extend(tensor.tobytes(order="C"))
        header[name] = {"dtype": code, "shape": tensor.shape, "data_offsets": [start, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
    return len(data)


def _float_weight(rng, name, shape):
    if name.endswith("_scale"):
        return np.asarray([0.3, 0.3, 0.1], dtype="<f4")
    if name.endswith(("norm.weight", ".q_weight", ".k_weight")):
        values = 1 + 0.05 * rng.standard_normal(shape)
    elif len(shape) == 1:
        values = 0.08 * rng.standard_normal(shape)
    else:
        amplitude = 0.8 if name in ("embed.weight", "head.weight") else 0.45
        values = rng.standard_normal(shape) * (amplitude / math.sqrt(shape[-1]))
    return values.astype("<f4")


def _quant_weight(rng, shape, encoding, block):
    physical, scales = quantized_shapes(shape, block, encoding)
    if encoding == "fp4":
        raw = rng.integers(0, 256, physical, dtype=np.uint8).view(np.int8)
        code, typical = "I8", 3.0
    else:
        raw = rng.integers(40, 76, physical, dtype=np.uint8)
        raw |= rng.integers(0, 2, physical, dtype=np.uint8) << 7
        code, typical = "F8_E4M3", 1.25
    exponent = round(math.log2(0.45 / (typical * math.sqrt(shape[-1]))))
    return (code, raw), ("F8_E8M0", np.full(scales, exponent + 127, np.uint8))


def write_fixture(path, *, quantized=True, seed=27, vision=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    root = fixture_config(quantized, vision=vision)
    cfg = DeepSeekConfig.from_mapping(root)
    (path / "config.json").write_text(json.dumps(root, indent=2) + "\n")
    tokens = [i % cfg.engram_compressed_vocab_size for i in range(cfg.vocab_size)]
    (path / "engram_token_map.json").write_text(json.dumps(tokens) + "\n")
    rng = np.random.default_rng(seed)
    shards = [{}, {}]
    shapes = weight_shapes(cfg)
    if vision:
        from deepseek_vision import vision_weight_shapes

        shapes.update(vision_weight_shapes(Checkpoint.open(path, metadata_only=True)))
        write_fixture_tokenizer(path / "tokenizer.json")
    for i, (name, shape) in enumerate(shapes.items()):
        layout = quant_layout(name) if quantized else None
        if layout:
            shards[i % 2][name], shards[(i + 1) % 2][scale_name(name)] = _quant_weight(rng, shape, *layout)
        else:
            shards[i % 2][name] = ("F32", _float_weight(rng, name, shape))
    mapping, total = {}, 0
    for i, tensors in enumerate(shards, 1):
        filename = f"model-{i:05d}-of-00002.safetensors"
        total += write_safetensors(path / filename, tensors)
        mapping.update(dict.fromkeys(tensors, filename))
    index = {"metadata": {"total_size": total}, "weight_map": mapping}
    (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return path


def write_fixture_tokenizer(path, *, size=96):
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers
    except ImportError as error:
        raise ValueError("Install tokenizers to generate the fixture tokenizer") from error

    tokens = [
        "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>", "[UNK]",
        "<｜User｜>", "<｜Assistant｜>", "</think>", "<think>", "<｜System｜>",
        "Hello,", "world!", "Blue.", "Again?", "Be", "brief.",
    ]
    tokens += [f"sample-{i}" for i in range(size - len(tokens))]
    image_id = size - 2
    tokens[image_id] = "<｜deepseek_image｜>"
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(size))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_special_tokens([*tokens[:9], tokens[image_id]])
    backend.save(str(path))
    return backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--with-tokenizer", action="store_true", help="Write the matching synthetic tokenizer")
    parser.add_argument("--vision", action="store_true", help="Include a reduced vision tower and tokenizer")
    args = parser.parse_args()
    write_fixture(args.output, quantized=not args.dense, seed=args.seed, vision=args.vision)
    if args.with_tokenizer and not args.vision:
        write_fixture_tokenizer(args.output / "tokenizer.json")

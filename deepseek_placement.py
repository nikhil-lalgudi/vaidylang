"""CPU rank arithmetic and metadata planning for DeepSeek V4.1 Flash.

Partition rules follow DeepSeek's MIT-licensed inference/convert.py and
inference/model.py at 314113f8a7dadb388215396ba5fbee2c610a04d6.
"""

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np

from deepseek_ops import swiglu
from deepseek_quant import QuantizedWeight, quantized_shapes


_EXPERT = re.compile(r"^(?:layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.")
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1}
_OUTPUT_PARTS = {"embed", "head", "wq_b", "wo_a", "weights_proj"}


@dataclass(frozen=True)
class Placement:
    mode: str
    axis: int | None = None
    owner: int | None = None


def _validate_sizes(world_size, experts, draft_experts):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world size must be a positive integer")
    for label, count in (("backbone", experts), ("draft", draft_experts)):
        if type(count) is not int or count < world_size or count % world_size:
            raise ValueError(f"{label} experts ({count}) must divide evenly over world size {world_size}")


def _placement(name, world_size, experts, draft_experts):
    match = _EXPERT.match(name)
    if match:
        count = draft_experts if name.startswith("mtp.") else experts
        index = int(match[1])
        if index >= count:
            raise ValueError(f"{name} uses expert {index} but has only {count} experts")
        return Placement("expert_owner", owner=index // (count // world_size))
    if ".engram.embed." in name:
        return Placement("engram_rows_padded", axis=0)
    if name.endswith(".attn_sink"):
        return Placement("output_rows", axis=0)
    leaf = name.rsplit(".", 2)[-2] if "." in name else name
    if leaf in _OUTPUT_PARTS:
        return Placement("output_rows", axis=0)
    if leaf == "wo_b":
        return Placement("input_columns", axis=1)
    return Placement("replicated")


def _local_shape(name, shape, location, world_size):
    if location.axis is None:
        return tuple(shape)
    axis = location.axis
    if len(shape) <= axis:
        raise ValueError(f"{name} has no partition axis {axis} in shape {shape}")
    width = shape[axis]
    if width % world_size and location.mode != "engram_rows_padded":
        raise ValueError(f"{name} axis {axis} width {width} must divide evenly over world size {world_size}")
    local_width = (width + world_size - 1) // world_size
    return (*shape[:axis], local_width, *shape[axis + 1:])


def partition_tensor(name, tensor, world_size, *, experts, draft_experts=None):
    """Materialize independent rank-local copies of a public weight or scale tensor."""
    draft_experts = experts if draft_experts is None else draft_experts
    _validate_sizes(world_size, experts, draft_experts)
    location = _placement(name, world_size, experts, draft_experts)
    tensor = np.asarray(tensor)
    shape = _local_shape(name, tensor.shape, location, world_size)
    if location.axis is None:
        return tuple(
            tensor.copy() if location.owner is None or rank == location.owner else None
            for rank in range(world_size)
        )
    if location.mode == "engram_rows_padded":
        padding = shape[0] * world_size - tensor.shape[0]
        # E8M0 stores one as byte 127. Non-byte fixtures can represent one directly.
        fill = 0
        if name.endswith(".scale"):
            fill = 127 if tensor.dtype.kind in "iu" and tensor.dtype.itemsize == 1 else 1
        tensor = np.pad(tensor, ((0, padding), *((0, 0) for _ in tensor.shape[1:])), constant_values=fill)
    return tuple(part.copy() for part in np.split(tensor, world_size, axis=location.axis))


def partition_quantized(
    name, raw, scales, logical_shape, world_size, *,
    experts, draft_experts=None, block_shape, encoding, activation_group_size=None,
):
    """Shard encoded values and scales together, requiring whole format blocks per split."""
    draft_experts = experts if draft_experts is None else draft_experts
    _validate_sizes(world_size, experts, draft_experts)
    location = _placement(name, world_size, experts, draft_experts)
    local_shape = _local_shape(name, logical_shape, location, world_size)
    storage, scale_storage = quantized_shapes(logical_shape, block_shape, encoding)
    if np.shape(raw) != storage or np.shape(scales) != scale_storage:
        raise ValueError(f"{name} requires storage shape {storage} and scale shape {scale_storage}")
    if world_size > 1 and location.axis is not None:
        axis = location.axis
        width = local_shape[axis]
        if width % block_shape[axis] or (axis == 1 and encoding == "fp4" and width % 2):
            raise ValueError(f"{name} rank boundary cuts a scale block or packed weight byte")
        if axis == 1 and activation_group_size and width % activation_group_size:
            raise ValueError(f"{name} rank boundary cuts a dynamic activation group")

    options = {"experts": experts, "draft_experts": draft_experts}
    weights = partition_tensor(name, raw, world_size, **options)
    scale_parts = partition_tensor(name.removesuffix(".weight") + ".scale", scales, world_size, **options)
    return tuple(
        None if weight is None else QuantizedWeight(
            weight, scale, local_shape, block_shape=block_shape, encoding=encoding,
            activation_group_size=activation_group_size,
        )
        for weight, scale in zip(weights, scale_parts)
    )


def embedding_contribution(ids, weight, rank):
    """Return this rank's part of a vocabulary/Engram lookup before sum reduction."""
    if type(rank) is not int or rank < 0 or weight.ndim != 2 or weight.shape[0] == 0:
        raise ValueError("Expected a nonnegative rank and a nonempty rank-local embedding matrix")
    local_ids = np.asarray(ids) - rank * weight.shape[0]
    mask = (local_ids >= 0) & (local_ids < weight.shape[0])
    return np.where(mask[..., None], weight[np.where(mask, local_ids, 0)], 0)


def index_score_contribution(query, weights, keys, *, total_heads):
    """Weighted, rectified query scores from this rank's heads and replicated keys."""
    scores = np.maximum(query @ keys.T, 0) * weights[..., None]
    return np.sum(scores, axis=-2, dtype=np.float32) / math.sqrt(total_heads * query.shape[-1])


def _linear(x, weight):
    return weight.linear(x) if isinstance(weight, QuantizedWeight) else x @ np.asarray(weight).T


def parallel_linear(x, shards, *, axis):
    """Execute each rank's linear, then apply the reference gather or sum."""
    x = np.asarray(x, dtype=np.float32)
    shards = tuple(shards)
    if not shards or axis not in (0, 1):
        raise ValueError("Expected rank-local linear shards and weight partition axis 0 or 1")
    if axis == 0:
        return np.concatenate([_linear(x, shard) for shard in shards], axis=-1)
    if len(shards) > 1:
        for shard in shards:
            if isinstance(shard, QuantizedWeight):
                group = shard.activation_group_size
                if group and shard.shape[1] % group:
                    raise ValueError("Rank input width must align with dynamic activation groups")
    return np.sum(
        [_linear(v, shard) for v, shard in zip(np.split(x, len(shards), axis=-1), shards)],
        axis=0, dtype=np.float32,
    )


def expert_contribution(x, selected, routing, local_experts, swiglu_limit):
    """Execute owned SwiGLU experts, weighting the hidden vector before w2."""
    x = np.asarray(x, dtype=np.float32)
    result = np.zeros_like(x)
    for expert, (w1, w2, w3) in local_experts.items():
        rows, slots = np.where(selected == expert)
        if not rows.size:
            continue
        gate, up = _linear(x[rows], w1), _linear(x[rows], w3)
        hidden = swiglu(gate, up, swiglu_limit)
        np.add.at(result, rows, _linear(hidden * routing[rows, slots, None], w2))
    return result


def _validate_header(shard, header, indexed):
    if not isinstance(header, dict):
        raise ValueError(f"{shard} metadata must be a Safetensors header JSON object")
    entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
    if entries.keys() != indexed:
        missing = sorted(indexed - entries.keys())
        extra = sorted(entries.keys() - indexed)
        raise ValueError(f"{shard} header differs from the index: missing {missing[:3]}, unexpected {extra[:3]}")
    offset = 0
    for name, entry in sorted(entries.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        dtype, shape, offsets = entry["dtype"], tuple(entry["shape"]), entry["data_offsets"]
        if dtype not in _DTYPE_BYTES or not all(type(dimension) is int and dimension >= 0 for dimension in shape):
            raise ValueError(f"{shard}: invalid stored dtype or shape for {name}")
        size = _DTYPE_BYTES[dtype] * math.prod(shape)
        if list(offsets) != [offset, offset + size]:
            raise ValueError(f"{shard}: invalid data offsets for {name}")
        offset += size
        yield name, dtype, shape, size


def _state_and_scratch(config, world_size, capacity, batch_size, query_tokens):
    n_layers = config["num_hidden_layers"]
    draft_layers = config.get("num_nextn_predict_layers", 0)
    head_dim, hidden = config["head_dim"], config["hidden_size"]
    window = config["sliding_window"]
    ratios = config["compress_ratios"]
    sources = config["kv_source_layer_ids"]
    index_sources = set(config["index_source_layer_ids"])
    index_dim, index_heads, index_topk = (config.get(key, 0) for key in ("index_head_dim", "index_n_heads", "index_topk"))
    rope_dim = config.get("qk_rope_head_dim", head_dim)
    engram_layers = len(config.get("engram_layer_ids", ()))
    ngram = config.get("engram_max_ngram_size", 1)
    hash_cols = (ngram - 1) * config.get("engram_n_heads", 0)
    compressed_slots = sum(capacity // ratios[layer] for layer in sources)
    max_compressed = max((capacity // ratios[layer] for layer in sources), default=0)
    num_rotary_tables = int(any(ratios)) + int(any(r == 0 for r in ratios))
    state = {
        "window_kv": batch_size * (n_layers + draft_layers) * window * head_dim * 2,
        "compressed_kv": batch_size * compressed_slots * head_dim * 2,
        "index_key": batch_size * sum(capacity // ratios[layer] for layer in sources if layer in index_sources) * index_dim * 2,
        "compressor_tail": batch_size * sum(ratios[layer] for layer in sources if ratios[layer] > 1) * head_dim * 2 * 4,
        "rotary_tables": num_rotary_tables * capacity * (rope_dim // 2) * 8,
        "engram_history": batch_size * capacity * 8 if engram_layers else 0,
        "engram_tables": (config["vocab_size"] + engram_layers * (2 * hash_cols + ngram)) * 8 if engram_layers else 0,
    }
    tokens = batch_size * query_tokens
    scratch = {
        "index_dot_scores_fp32": tokens * max_compressed * (index_heads // world_size) * 4,
        "index_reduced_scores_fp32": tokens * max_compressed * 4,
        "index_candidate_mask": tokens * max_compressed if config.get("candidate_topk_blocks", 0) else 0,
        "attention_indices_int32": tokens * (window + min(index_topk, max_compressed)) * 4,
        "concat_raw_and_compressed_kv_bf16": batch_size * (max(window, query_tokens) + max_compressed) * head_dim * 2,
        "hc_residual_and_output_bf16": tokens * config["hc_mult"] * hidden * 2 * 2,
        "engram_selected_rows_bf16": tokens * hash_cols * config.get("engram_head_dim", 0) * 2,
        "dspark_feature_rows_bf16": tokens * len(config.get("dspark_target_layer_ids", ())) * hidden * 2,
        "replicated_logits_fp32": batch_size * max(1, config.get("dspark_block_size", 0)) * config["vocab_size"] * 4,
    }
    return state, scratch


def _runtime_item_bytes(name, source_dtype, ratios):
    if name.endswith(".wo_a.scale"):
        return 0  # converted grouped output projection drops the stored scale
    if name.endswith(".wo_a.weight"):
        return 2  # FP8 blocks are converted to BF16
    if name.endswith(("head.weight", ".confidence_head.proj.weight")):
        return 4
    if ".attn.compressor.w" in name and ratios[int(name.split(".")[1])] > 1:
        return 4
    return _DTYPE_BYTES[source_dtype]


def placement_report(checkpoint, headers, world_size, *, capacity=None, batch_size=1, query_tokens=1):
    """Assign every indexed tensor and estimate allocation from sizes alone."""
    if checkpoint.model_type != "deepseek_v41" or checkpoint.weight_map is None:
        raise ValueError("Placement requires a sharded DeepSeek V4.1 text config and weight index")
    config = checkpoint.text_config
    experts = config["n_routed_experts"]
    draft_experts = config.get("dspark_n_routed_experts") or experts
    _validate_sizes(world_size, experts, draft_experts)
    for key in ("vocab_size", "num_attention_heads", "o_groups", "index_n_heads"):
        dimension = config.get(key, 0)
        if dimension % world_size:
            raise ValueError(f"{key} ({dimension}) must divide evenly over world size {world_size}")
    capacity = config["max_position_embeddings"] if capacity is None else capacity
    if batch_size < 1 or not 1 <= query_tokens <= capacity <= config["max_position_embeddings"]:
        raise ValueError("Use positive batch/query sizes and capacity within the model context limit")
    if headers.keys() != set(checkpoint.shards):
        missing = sorted(set(checkpoint.shards) - headers.keys())
        extra = sorted(headers.keys() - set(checkpoint.shards))
        raise ValueError(f"Shard headers differ from the index: missing {missing[:3]}, unexpected {extra[:3]}")

    by_shard = {shard: set() for shard in checkpoint.shards}
    for name, shard in checkpoint.weight_map.items():
        by_shard[shard].add(name)
    ranks = [Counter() for _ in range(world_size)]
    source_by_dtype = Counter()
    source_by_placement = Counter()
    source_by_namespace = Counter()
    examples = {}
    for shard in checkpoint.shards:
        for name, dtype, shape, size in _validate_header(shard, headers[shard], by_shard[shard]):
            location = _placement(name, world_size, experts, draft_experts)
            local_shape = _local_shape(name, shape, location, world_size)
            source_by_dtype[dtype] += size
            source_by_placement[location.mode] += 1
            namespace = (
                "vision" if name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".gate.bias_vl")
                else "draft" if name.startswith("mtp.") else "backbone"
            )
            source_by_namespace[namespace] += size
            category = "scale" if dtype == "F8_E8M0" else "weight"
            stored = math.prod(local_shape) * _DTYPE_BYTES[dtype]
            runtime = math.prod(local_shape) * _runtime_item_bytes(name, dtype, config["compress_ratios"])
            targets = range(world_size) if location.owner is None else (location.owner,)
            for rank in targets:
                counts = ranks[rank]
                counts["tensor_count"] += 1
                if namespace == "vision":
                    counts["vision_bytes"] += runtime
                else:
                    counts[f"text_stored_{category}_bytes"] += stored
                    counts[f"text_runtime_{category}_bytes"] += runtime
                    counts[f"{namespace}_runtime_bytes"] += runtime
            examples.setdefault(location.mode, {
                "tensor": name, "dtype": dtype, "stored_shape": list(shape),
                "rank_shape": list(local_shape), "axis": location.axis, "owner_rank": location.owner,
            })
    source_bytes = sum(source_by_dtype.values())
    if checkpoint.total_tensor_bytes is not None and checkpoint.total_tensor_bytes != source_bytes:
        raise ValueError(f"Index size {checkpoint.total_tensor_bytes} does not match headers ({source_bytes} bytes)")
    state, scratch = _state_and_scratch(config, world_size, capacity, batch_size, query_tokens)
    return {
        "world_size": world_size,
        "batch_size": batch_size,
        "capacity": capacity,
        "scratch_query_tokens": query_tokens,
        "source_tensor_count": len(checkpoint.weight_map),
        "source_tensor_bytes": source_bytes,
        "source_bytes_by_dtype": dict(source_by_dtype),
        "source_bytes_by_namespace": dict(source_by_namespace),
        "source_count_by_placement": dict(source_by_placement),
        "example_placements": examples,
        "attention_cache_ownership": _cache_ownership(config),
        "per_rank_state_bytes": state,
        "per_rank_scratch_bytes": scratch,
        "per_rank_total_state_bytes": sum(state.values()),
        "ranks": [
            {
                "rank": rank,
                **counts,
                "text_runtime_weights_and_state_bytes": counts["text_runtime_weight_bytes"]
                + counts["text_runtime_scale_bytes"] + sum(state.values()),
            }
            for rank, counts in enumerate(ranks)
        ],
    }


def _cache_ownership(config):
    kv_sources = set(config["kv_source_layer_ids"])
    index_sources = set(config["index_source_layer_ids"])
    candidate_source = config.get("candidate_source_layer_id", -1)
    kv_owner = index_owner = None
    layers = []
    for layer, ratio in enumerate(config["compress_ratios"]):
        if layer in kv_sources:
            kv_owner = layer
        if layer in index_sources:
            index_owner = layer
        layers.append({
            "layer": layer,
            "replication": "every rank",
            "owns_window": True,
            "compression_ratio": ratio,
            "compressed_kv_source": kv_owner if ratio else None,
            "index_key_source": kv_owner if ratio else None,
            "index_selection_source": index_owner if ratio else None,
            "candidate_source": candidate_source if ratio and 0 <= candidate_source <= layer else None,
        })
    return layers

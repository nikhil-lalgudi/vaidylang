"""Cache-free FP64 oracles for reduced DeepSeek V4.1 text checkpoints.

Equations are derived from the DeepSeek V4.1 Flash model.py, kernel.py and
engram.py at revision 314113f8a7dadb388215396ba5fbee2c610a04d6 (MIT).
Each layer works on the whole prefix. In particular, compression never uses a
pending group, raw attention never wraps a ring, and Engram reads token IDs
directly from the prefix. It shares no model operations with the cache engine.
"""

import math

import numpy as np


def _e4m3(code):
    exponent, fraction = divmod(code, 8)
    if exponent == 0:
        return fraction / 512
    if code == 127:
        return math.nan
    return (8 + fraction) * 2.0 ** (exponent - 10)


_POSITIVE_FP8 = np.asarray([_e4m3(n) for n in range(127)])
_POSITIVE_FP4 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _nearest(values, codebook):
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    magnitude = np.abs(values)
    nearest = np.searchsorted(boundaries, magnitude)
    on_boundary = magnitude == boundaries[np.minimum(nearest, len(boundaries) - 1)]
    nearest += on_boundary & (nearest % 2 == 1)
    return np.copysign(codebook[nearest], values)


def _microscale(values, kind="fp8", size=32, scale_kind="e8m0"):
    # Scaling and casting follow the pinned kernel's FP32 input arithmetic.
    values = values.astype(np.float32)
    codebook = _POSITIVE_FP8 if kind == "fp8" else _POSITIVE_FP4
    groups = []
    for first in range(0, values.shape[-1], size):
        part = values[..., first:first + size]
        peak = np.max(np.abs(part), axis=-1, keepdims=True)
        if scale_kind == "fp8":
            scale = _nearest(np.maximum(peak, np.float32(6 / 512)) / np.float32(6), _POSITIVE_FP8)
        else:
            minimum = 0.0001 if kind == "fp8" else 6 * 2.0**-126
            fraction = np.maximum(peak, np.float32(minimum)) * np.float32(1 / codebook[-1])
            scale = np.exp2(np.ceil(np.log2(fraction.astype(np.float64))))
        groups.append(_nearest(part / scale.astype(np.float32), codebook) * scale)
    return np.concatenate(groups, axis=-1)


class _QuantizedMatrix:
    def __init__(self, data):
        self.data = data


def _project(x, weight):
    if isinstance(weight, _QuantizedMatrix):
        return _microscale(x) @ weight.data.T
    return x @ weight.T


def _norm(x, scale, eps):
    return scale * x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) / x.shape[-1] + eps)


def _sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def _normalized_exp(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _rotate(x, positions, cfg, compressed, inverse=False):
    """Complex multiplication independently computes the adjacent-pair rotation."""
    dim = cfg.qk_rope_head_dim
    theta = cfg.compress_rope_theta if compressed else cfg.rope_theta
    freqs = np.geomspace(1, theta ** (-1 + 2 / dim), num=dim // 2)
    yarn = cfg.rope_scaling if compressed else None
    if yarn and yarn["original_max_position_embeddings"] > 0:
        orig = yarn["original_max_position_embeddings"]

        def correction(beta):
            return dim / (2 * math.log(theta)) * math.log(orig / (2 * math.pi * beta))

        start = max(0, math.floor(correction(yarn["beta_fast"])))
        stop = min(dim - 1, math.ceil(correction(yarn["beta_slow"])))
        fade = np.clip((np.arange(dim // 2) - start) / max(stop - start, 0.001), 0, 1)
        freqs *= (1 - fade) + fade / yarn["factor"]
    phase = np.exp(((-1j if inverse else 1j) * np.asarray(positions)[:, None]) * freqs)
    phase = phase.reshape((len(positions),) + (1,) * (x.ndim - 2) + (dim // 2,))
    tail = x[..., -dim:]
    rotated = (tail[..., ::2] + 1j * tail[..., 1::2]) * phase
    y = x.copy()
    y[..., -dim::2] = rotated.real
    y[..., -dim + 1::2] = rotated.imag
    return y


def _hashes(ids, config, token_map):
    """Use trial division and Python integer XOR to derive all lookup indices."""
    assigned = set()
    mapped = [int(token_map[token]) for token in ids]
    pad = int(token_map[config.engram_pad_token_id])
    orders = config.engram_max_ngram_size
    limit = (2**63 - 1) // config.engram_compressed_vocab_size // 2
    result = {}
    for layer in config.engram_layer_ids:
        seed = np.random.default_rng(10007 * layer)
        multipliers = [2 * int(v) + 1 for v in seed.integers(limit, size=orders)]
        ranges = []
        offset = 0
        for _ in range(1, orders):
            row = []
            n = config.engram_vocab_size
            while len(row) < config.engram_n_heads:
                if n not in assigned and n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1)):
                    assigned.add(n)
                    row.append((n, offset))
                    offset += n
                n += 1
            ranges.append(row)
        layer_indices = []
        for pos in range(len(mapped)):
            rolling = mapped[pos] * multipliers[0]
            indices = []
            for shift, row in enumerate(ranges, start=1):
                rolling ^= (mapped[pos - shift] if pos >= shift else pad) * multipliers[shift]
                indices.extend(rolling % prime + base for prime, base in row)
            layer_indices.append(indices)
        result[layer] = np.asarray(layer_indices)
    return result


def _engram(stream, w, indices, cfg):
    rows = w["engram.embed.weight"][indices].reshape(len(indices), -1)
    kv = _project(rows, w["engram.wkv.weight"])
    key = kv[:, :-cfg.hidden_size].reshape(stream.shape)
    value = kv[:, -cfg.hidden_size:]
    query = _norm(stream, w["engram.q_weight"], cfg.rms_norm_eps)
    key = _norm(key, w["engram.k_weight"], cfg.rms_norm_eps)
    dot = np.sum(key * query, axis=-1) / math.sqrt(cfg.hidden_size)
    scores = _sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot))
    return stream + scores[..., None] * value[:, None, :]


def _connection(stream, w, which, cfg):
    h = cfg.hc_mult
    normalized = _norm(stream.reshape(len(stream), -1), 1, cfg.rms_norm_eps)
    dynamic = normalized @ w[f"hc_{which}_fn"].T
    scales = np.repeat(w[f"hc_{which}_scale"], [h, h, h * h])
    a, b, blend = np.split(dynamic * scales + w[f"hc_{which}_base"], [h, 2 * h], axis=1)
    blend = _normalized_exp(blend.reshape(len(stream), h, h)) + cfg.hc_eps
    for i in range(cfg.hc_sinkhorn_iters):
        if i:
            blend /= blend.sum(axis=2, keepdims=True) + cfg.hc_eps
        blend /= blend.sum(axis=1, keepdims=True) + cfg.hc_eps
    return _sigmoid(a) + cfg.hc_eps, 2 * _sigmoid(b), blend


def _index(q, keys, weights, cfg, ratio, prior_candidates, candidate_source):
    selected = []
    candidates = []
    for p, head_queries in enumerate(q):
        visible = (p + 1) // ratio
        scores = [
            sum(float(weight) * max(0.0, float(head @ key)) for head, weight in zip(head_queries, weights[p]))
            for key in keys[:visible]
        ]
        allowed = list(range(visible))
        if candidate_source and visible:
            size = cfg.candidate_block_size
            blocks = list(range((visible - 1) // size + 1))
            latest = blocks.pop()
            blocks.sort(key=lambda block: (-max(scores[block * size:(block + 1) * size]), block))
            chosen = {latest, *blocks[:cfg.candidate_topk_blocks - 1]}
            candidates.append([i for i in allowed if i // size in chosen])
        else:
            candidates.append([])
        if prior_candidates is not None:
            allowed = prior_candidates[p]
        best = sorted(allowed, key=lambda ix: (-scores[ix], ix))[:cfg.index_topk]
        selected.append(np.asarray(sorted(best), dtype=np.int64))
    return selected, candidates


def _attention(x, w, layer, cfg, shared, record, activated):
    positions = np.arange(len(x))
    ratio = cfg.compress_ratios[layer]
    latent_q = _norm(_project(x, w["attn.wq_a.weight"]), w["attn.q_norm.weight"], cfg.rms_norm_eps)
    query = _rotate(
        _project(latent_q, w["attn.wq_b.weight"]).reshape(len(x), cfg.num_attention_heads, cfg.head_dim),
        positions, cfg, ratio,
    )
    raw = activated(_rotate(
        _norm(_project(x, w["attn.wkv.weight"]), w["attn.kv_norm.weight"], cfg.rms_norm_eps),
        positions, cfg, ratio,
    ))
    record("attention_query", query)
    record("raw_kv", raw)
    starts = positions[:len(x) - len(x) % ratio:ratio] if ratio else []
    latent = None
    if layer in cfg.kv_source_layer_ids:
        values = x @ w["attn.compressor.wkv.weight"].T
        if ratio > 1:
            scores = x @ w["attn.compressor.wgate.weight"].T
            values = np.asarray([
                (values[s:s + ratio] * _normalized_exp(scores[s:s + ratio], axis=0)).sum(axis=0)
                for s in starts
            ], dtype=np.float64).reshape(len(starts), cfg.head_dim)
        latent = _norm(values, w["attn.compressor.norm.weight"], cfg.rms_norm_eps)
        for group, item in enumerate(latent):
            record("compress_latent", item, position=(group + 1) * ratio - 1)
        compressed = _rotate(latent, starts, cfg, ratio)
        shared["compressed"] = activated(compressed, "fp4", 16, "fp8")
        keys = _rotate(
            _norm(_project(latent, w["attn.indexer.wk.weight"]), w["attn.indexer.k_norm.weight"], cfg.rms_norm_eps),
            starts, cfg, ratio,
        )
        shared["keys"] = activated(keys, "fp4")
    if layer in cfg.index_source_layer_ids:
        q = _project(latent_q, w["attn.indexer.wq_b.weight"]).reshape(len(x), cfg.index_n_heads, cfg.index_head_dim)
        q = activated(_rotate(q, positions, cfg, ratio), "fp4")
        weights = _project(x, w["attn.indexer.weights_proj.weight"]) / math.sqrt(cfg.index_n_heads * cfg.index_head_dim)
        candidate_source = layer == cfg.candidate_source_layer_id
        prior = shared.get("candidates") if 0 <= cfg.candidate_source_layer_id < layer else None
        keys = shared["keys"]
        shared["selected"], candidates = _index(q, keys, weights, cfg, ratio, prior, candidate_source)
        if candidate_source:
            shared["candidates"] = candidates
        for p in positions:
            record("index_keys", keys[:(p + 1) // ratio], position=p)
    head_outputs = []
    for p in positions:
        kv = raw[max(0, p + 1 - cfg.sliding_window):p + 1]
        if ratio:
            compressed_indices = shared["selected"][p]
            record("index_selected", compressed_indices, position=p)
            kv = np.concatenate((kv, shared["compressed"][compressed_indices]), axis=0)
        scores = query[p] @ kv.T / math.sqrt(cfg.head_dim)
        logits = np.column_stack((scores, w["attn.attn_sink"]))
        head_outputs.append(_normalized_exp(logits)[:, :-1] @ kv)
    outputs = np.asarray(head_outputs)
    record("sparse_attention", outputs)
    outputs = _rotate(outputs, positions, cfg, ratio, inverse=True)
    groups = activated(outputs.reshape(len(x), cfg.o_groups, -1))
    projector = w["attn.wo_a.weight"].reshape(cfg.o_groups, cfg.o_lora_rank, groups.shape[-1])
    joined = np.concatenate([groups[:, g] @ projector[g].T for g in range(cfg.o_groups)], axis=-1)
    answer = _project(joined, w["attn.wo_b.weight"])
    record("attention", answer)
    return answer


def _moe(x, w, cfg, record):
    scores = (x @ w["ffn.gate.weight"].T) / cfg.gate_temperature
    if cfg.scoring_func == "sqrtsoftplus":
        scores = np.sqrt(np.logaddexp(0, scores))
    elif cfg.scoring_func == "sigmoid":
        scores = _sigmoid(scores)
    else:
        scores = _normalized_exp(scores)
    outputs = []
    for p, value in enumerate(x):
        chosen = sorted(range(cfg.n_routed_experts), key=lambda n: (-scores[p, n] - w["ffn.gate.bias"][n], n))
        chosen = np.asarray(chosen[:cfg.num_experts_per_tok])
        weights = scores[p, chosen]
        if cfg.norm_topk_prob and len(chosen) > 1:
            weights = weights / (sum(weights) + 1e-20)
        weights = weights * cfg.routed_scaling_factor
        record("expert_ids", chosen, position=p)
        record("expert_weights", weights, position=p)
        answer = np.zeros(cfg.hidden_size)
        ranked = sorted(zip(chosen, weights))
        for kind, weight in [*((f"experts.{i}", weight) for i, weight in ranked), ("shared_experts", 1.0)]:
            gate = _project(value, w[f"ffn.{kind}.w1.weight"])
            up = _project(value, w[f"ffn.{kind}.w3.weight"])
            if cfg.swiglu_limit > 0:
                gate = np.minimum(gate, cfg.swiglu_limit)
                up = np.clip(up, -cfg.swiglu_limit, cfg.swiglu_limit)
            intermediate = (gate * _sigmoid(gate)) * up * weight
            answer += _project(intermediate, w[f"ffn.{kind}.w2.weight"])
        outputs.append(answer)
    answer = np.asarray(outputs)
    record("moe", answer)
    return answer


class DenseOracle:
    """Independent FP64 full-prefix evaluator. The fixture supplies only model data."""

    arithmetic_path = "FP32 cached target versus independent FP64 cache-free dense oracle"

    def __init__(self, config, weights, token_map):
        self.config = config
        self.token_map = token_map
        weights = {name: value for name, value in weights.items() if not name.startswith("mtp.")}
        if any(not isinstance(value, (np.ndarray, _QuantizedMatrix)) for value in weights.values()):
            raise ValueError("Independent FP64 oracle requires dense reduced weights")
        self.weights = {
            name: value.astype(np.float64) if isinstance(value, np.ndarray) else value
            for name, value in weights.items()
        }
        self.layers = []
        for i in range(config.num_hidden_layers):
            prefix = f"layers.{i}."
            self.layers.append({key[len(prefix):]: value for key, value in self.weights.items() if key.startswith(prefix)})

    def activation(self, values, kind="fp8", size=32, scale_kind="e8m0"):
        return values

    def forward(self, ids, trace=None):
        cfg, weights = self.config, self.weights
        stream = np.repeat(weights["embed.weight"][ids, None, :], cfg.hc_mult, axis=1)
        pre = np.zeros((len(ids), cfg.hc_mult))
        pre[:, 0] = 1.0
        hashes = _hashes(ids, cfg, self.token_map) if cfg.engram_layer_ids else {}
        shared = {}

        def record(operation, value, *, position=None):
            if trace is not None:
                if position is None:
                    for p, item in enumerate(value):
                        trace(p, layer, operation, item)
                else:
                    trace(int(position), layer, operation, value)

        for layer, w in enumerate(self.layers):
            if layer in hashes:
                record("engram_hashes", hashes[layer])
                stream = _engram(stream, w, hashes[layer], cfg)
                record("engram", stream)
            after_attn, post, combination = _connection(stream, w, "attn", cfg)
            collapsed = np.einsum("pi,pid->pd", pre, stream)
            attn_input = _norm(collapsed, w["attn_norm.weight"], cfg.rms_norm_eps)
            attn = _attention(attn_input, w, layer, cfg, shared, record, self.activation)
            stream = np.matmul(combination.transpose(0, 2, 1), stream) + post[..., None] * attn[:, None, :]
            pre, post, combination = _connection(stream, w, "ffn", cfg)
            collapsed = np.einsum("pi,pid->pd", after_attn, stream)
            ffn = _moe(_norm(collapsed, w["ffn_norm.weight"], cfg.rms_norm_eps), w, cfg, record)
            stream = np.matmul(combination.transpose(0, 2, 1), stream) + post[..., None] * ffn[:, None, :]
            record("residual", stream)
        layer = -1
        final = _norm(np.einsum("pi,pid->pd", pre, stream), weights["norm.weight"], cfg.rms_norm_eps)
        record("final_hidden", final)
        logits = final @ weights["head.weight"].T
        record("logits", logits)
        return logits


def _decode_matrix(raw, dtype, scales, width, row_size, column_size=32):
    codes = raw.view(np.uint8)
    if dtype == "I8":
        columns = np.arange(width)
        codes = (codes[:, columns // 2] // (16 ** (columns % 2))) % 16
        values = np.where(codes >= 8, -1, 1) * _POSITIVE_FP4[codes % 8]
    else:
        table = np.append(_POSITIVE_FP8, np.nan)
        values = np.where(codes >= 128, -1, 1) * table[codes % 128]
    scale_values = np.exp2(scales.astype(np.float64) - 127)
    tiled_rows = np.repeat(scale_values, row_size, axis=0)
    return values * np.repeat(tiled_rows, column_size, axis=1)[:raw.shape[0], :width]


def _round_to_bfloat(x):
    fraction, exponent = np.frexp(x)
    return np.ldexp(np.rint(fraction * 256) / 256, exponent)


class QuantizedOracle(DenseOracle):
    """Read storage bytes independently and enumerate reference rounding codebooks."""

    arithmetic_path = "FP32 cached target versus independent FP64 cache-free codebook oracle"

    @classmethod
    def from_checkpoint(cls, path, config, token_map, weights):
        from checkpoint import Checkpoint

        checkpoint = Checkpoint.open(path)
        live = {name: weight for name, weight in weights.items() if not name.startswith("mtp.")}
        stored = {name: (dtype, raw) for name, dtype, raw in checkpoint.read_storage(live)}
        quant_names = [name for name, (dtype, _) in stored.items() if dtype in ("I8", "F8_E4M3")]
        scale_names = {name.removesuffix(".weight") + ".scale": name for name in quant_names}
        scales = {scale_names[name]: raw for name, _, raw in checkpoint.read_storage(scale_names)}
        arrays = {}
        for name, (dtype, raw) in stored.items():
            if name in scales:
                embed = name.endswith(".engram.embed.weight")
                wo_a = name.endswith(".attn.wo_a.weight")
                row_block = 1 if embed or dtype == "I8" else 32
                weight = _decode_matrix(raw, dtype, scales[name], live[name].shape[1], row_block)
                arrays[name] = _round_to_bfloat(weight) if embed or wo_a else _QuantizedMatrix(weight)
            elif dtype == "BF16":
                arrays[name] = (raw.astype(np.uint32) * 65536).view(np.float32)
            else:
                arrays[name] = raw
        return cls(config, arrays, token_map)

    def activation(self, values, kind="fp8", size=32, scale_kind="e8m0"):
        return _microscale(values, kind, size, scale_kind)

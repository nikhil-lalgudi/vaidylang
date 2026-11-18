"""Reduced CPU text execution for DeepSeek V4.1 Flash.

The backbone follows the pinned DeepSeek model, including separate raw,
compressed, and index caches. All mutable request data belongs to DeepSeekState.
See deepseek-cpu-reference.md and LICENSE.DeepSeek.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from checkpoint import Checkpoint
from deepseek_config import DeepSeekConfig, cache_owner, weight_shapes
from deepseek_engram import EngramHash, compressed_token_map, gated_lookup
from deepseek_ops import (
    candidate_mask, hc_post, hc_pre, index_scores, pool_latents, rms_norm,
    rope_frequencies, rotary, route, sparse_attention, split_hc, swiglu, top_indices,
)

# The CPU reference materializes reduced weights. Metadata inspection and
# placement estimation are the entry points for the published checkpoint.
_MAX_CPU_STORAGE_BYTES = 512 * 1024**2


@dataclass
class DeepSeekState:
    capacity: int
    raw: list
    compressed: dict
    index_keys: dict
    pending: dict
    mapped_tokens: list
    target_features: list
    length: int = 0

    @property
    def remaining(self):
        return self.capacity - self.length


class DeepSeekEngine:
    def __init__(self, config, weights, token_map=None):
        self.config = config
        self.weights = weights
        self.token_map = token_map
        if config.engram_layer_ids and token_map is None:
            raise ValueError("Engram requires a compressed token map")
        self.engram_hash = EngramHash(config, token_map) if config.engram_layer_ids else None
        self.freqs = rope_frequencies(config)
        self.compress_freqs = rope_frequencies(config, compressed=True)
        self.kv_sources = [cache_owner(config.kv_source_layer_ids, i) for i in range(config.num_hidden_layers)]
        self.index_sources = [cache_owner(config.index_source_layer_ids, i) for i in range(config.num_hidden_layers)]

    @classmethod
    def from_checkpoint(cls, directory):
        checkpoint = directory if isinstance(directory, Checkpoint) else Checkpoint.open(directory)
        config = DeepSeekConfig.from_mapping(checkpoint.config)
        file_bytes = sum(
            (checkpoint.directory / shard).stat().st_size for shard in checkpoint.shards
        )
        if max(file_bytes, checkpoint.total_tensor_bytes or 0) > _MAX_CPU_STORAGE_BYTES:
            raise ValueError("DeepSeek CPU reference loads reduced checkpoints up to 512 MiB of stored tensors")
        from deepseek_weights import read_deepseek_weights

        weights = read_deepseek_weights(checkpoint, config, weight_shapes(config))
        return cls(config, weights, compressed_token_map(config, checkpoint.directory))

    def dense_weight(self, name):
        weight = self.weights[name]
        return weight if isinstance(weight, np.ndarray) else weight.weight

    def linear(self, name, x):
        weight = self.weights[name]
        return x @ weight.T if isinstance(weight, np.ndarray) else weight.linear(x)

    def activation(self, x, encoding="fp8", group_size=32, scale_encoding="e8m0"):
        if self.config.quantization_config:
            from deepseek_quant import quantize_activation

            return quantize_activation(x, encoding=encoding, group_size=group_size, scale_encoding=scale_encoding)
        return x

    def new_state(self, capacity):
        c = self.config
        if type(capacity) is not int or not 1 <= capacity <= c.max_position_embeddings:
            raise ValueError("KV capacity is outside the model context limit")
        sources = c.kv_source_layer_ids
        return DeepSeekState(
            capacity=capacity,
            raw=[deque(maxlen=c.sliding_window) for _ in range(c.num_hidden_layers)],
            compressed={i: [] for i in sources},
            index_keys={i: [] for i in sources},
            pending={i: [] for i in sources},
            mapped_tokens=[],
            target_features=[],
        )

    def prefill(self, token_ids, capacity):
        state = self.new_state(capacity)
        return self.forward(token_ids, state)[-1], state

    def decode(self, token_id, state):
        return self.forward([token_id], state)[-1]

    def forward(self, token_ids, state, trace=None):
        """Append live tokens and return one next-token logit vector per position."""
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty one-dimensional sequence of token IDs")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("Token ID is outside the vocabulary")
        if ids.size > state.remaining:
            raise ValueError("KV cache capacity exceeded")
        return np.stack([self._token(int(token), state, trace) for token in ids])

    def _token(self, token, state, trace):
        c, pos = self.config, state.length
        emit = (lambda layer, op, value: trace(pos, layer, op, value)) if trace else None
        x = np.repeat(self.weights["embed.weight"][token][None, :], c.hc_mult, axis=0)
        pre = np.zeros(c.hc_mult, np.float32)
        pre[0] = 1.0
        shared = {}
        features = []
        if self.engram_hash is not None:
            state.mapped_tokens.append(int(self.token_map[token]))
        for layer in range(c.num_hidden_layers):
            prefix = f"layers.{layer}"
            if layer in c.engram_layer_ids:
                x = self._engram(prefix, layer, x, state, emit)
            if layer in c.dspark_target_layer_ids:
                features.append(x.mean(axis=0))

            attn_pre, post, comb = self.hc_mixes(prefix, "attn", x)
            sub = rms_norm(hc_pre(x, pre), self.weights[prefix + ".attn_norm.weight"], c.rms_norm_eps)
            attn = self._attention(prefix + ".attn", layer, sub, state, shared, emit)
            x = hc_post(attn, x, post, comb)
            pre, post, comb = self.hc_mixes(prefix, "ffn", x)
            sub = rms_norm(hc_pre(x, attn_pre), self.weights[prefix + ".ffn_norm.weight"], c.rms_norm_eps)
            x = hc_post(self.moe(prefix, sub, layer, emit), x, post, comb)
            if emit:
                emit(layer, "residual", x)
        if features:
            state.target_features.append(np.concatenate(features))
        hidden = rms_norm(hc_pre(x, pre), self.weights["norm.weight"], c.rms_norm_eps)
        logits = self.linear("head.weight", hidden)
        if emit:
            emit(-1, "final_hidden", hidden)
            emit(-1, "logits", logits)
        state.length += 1
        return logits

    def hc_mixes(self, prefix, branch, x):
        c = self.config
        prefix += f".hc_{branch}_"
        flat = x.reshape(*x.shape[:-2], -1)
        mixes = self.linear(prefix + "fn", rms_norm(flat, 1.0, c.rms_norm_eps))
        return split_hc(
            mixes, self.weights[prefix + "scale"], self.weights[prefix + "base"],
            c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps,
        )

    def moe(self, prefix, x, layer_id, emit=None):
        c = self.config
        prefix += ".ffn."
        logits = self.linear(prefix + "gate.weight", x) / c.gate_temperature
        _, count = c.expert_counts(layer_id)
        ids, weights = route(
            logits, self.weights[prefix + "gate.bias"], count,
            c.scoring_func, c.norm_topk_prob, c.routed_scaling_factor,
        )
        y = np.zeros_like(x)
        for i in np.argsort(ids):
            y += self._expert(f"{prefix}experts.{ids[i]}", x, weights[i])
        y += self._expert(prefix + "shared_experts", x, 1.0)
        if emit:
            emit(layer_id, "expert_ids", ids)
            emit(layer_id, "expert_weights", weights)
            emit(layer_id, "moe", y)
        return y

    def _expert(self, prefix, x, weight):
        gate = self.linear(prefix + ".w1.weight", x)
        up = self.linear(prefix + ".w3.weight", x)
        # Weight before w2, which quantizes its own input in the FP4 path.
        return self.linear(prefix + ".w2.weight", swiglu(gate, up, self.config.swiglu_limit) * weight)

    def _engram(self, prefix, layer, x, state, emit):
        prefix += ".engram."
        hashes = self.engram_hash.hashes(layer, state.mapped_tokens)
        rows = self.dense_weight(prefix + "embed.weight")[hashes]
        y = gated_lookup(
            x, self.linear(prefix + "wkv.weight", rows.ravel()),
            self.weights[prefix + "q_weight"], self.weights[prefix + "k_weight"], self.config.rms_norm_eps,
        )
        if emit:
            emit(layer, "engram_hashes", hashes)
            emit(layer, "engram", y)
        return y

    def _attention(self, prefix, layer, x, state, shared, emit):
        c = self.config
        ratio = c.compress_ratios[layer]
        width, pos = c.qk_rope_head_dim, state.length
        angle = pos * (self.compress_freqs if ratio else self.freqs)
        qr = rms_norm(self.linear(prefix + ".wq_a.weight", x), self.weights[prefix + ".q_norm.weight"], c.rms_norm_eps)
        q = self.linear(prefix + ".wq_b.weight", qr).reshape(c.num_attention_heads, c.head_dim)
        q = rotary(q, width, angle)
        kv = rms_norm(self.linear(prefix + ".wkv.weight", x), self.weights[prefix + ".kv_norm.weight"], c.rms_norm_eps)
        raw = self.activation(rotary(kv, width, angle))
        state.raw[layer].append(raw)
        values = list(state.raw[layer])
        if emit:
            emit(layer, "attention_query", q)
            emit(layer, "raw_kv", raw)
        if ratio:
            values.extend(self._compressed(prefix, layer, x, qr, state, shared, angle, emit))
        attended = sparse_attention(q, np.asarray(values), self.weights[prefix + ".attn_sink"])
        grouped = rotary(attended, width, angle, inverse=True).reshape(c.o_groups, -1)
        grouped = self.activation(grouped)
        out_a = self.dense_weight(prefix + ".wo_a.weight").reshape(c.o_groups, c.o_lora_rank, -1)
        result = self.linear(prefix + ".wo_b.weight", np.einsum("gd,grd->gr", grouped, out_a).ravel())
        if emit:
            emit(layer, "sparse_attention", attended)
            emit(layer, "attention", result)
        return result

    def _compressed(self, prefix, layer, x, qr, state, shared, angle, emit):
        c = self.config
        pos, ratio = state.length, c.compress_ratios[layer]
        width = c.qk_rope_head_dim
        source, index = self.kv_sources[layer], self.index_sources[layer]
        if layer == source:
            name = prefix + ".compressor."
            latent = self.linear(name + "wkv.weight", x)
            if ratio > 1:
                pending = state.pending[layer]
                pending.append((latent, self.linear(name + "wgate.weight", x)))
                if len(pending) == ratio:
                    values, scores = map(np.asarray, zip(*pending))
                    latent = pool_latents(values, scores)
                    pending.clear()
                else:
                    latent = None
            if latent is not None:
                latent = rms_norm(latent, self.weights[name + "norm.weight"], c.rms_norm_eps)
                if emit:
                    emit(layer, "compress_latent", latent)
                group_angle = (pos + 1 - ratio) * self.compress_freqs
                name = prefix + ".indexer."
                key = rms_norm(self.linear(name + "wk.weight", latent), self.weights[name + "k_norm.weight"], c.rms_norm_eps)
                state.index_keys[layer].append(self.activation(rotary(key, width, group_angle), encoding="fp4"))
                state.compressed[layer].append(self.activation(
                    rotary(latent, width, group_angle), encoding="fp4", group_size=16, scale_encoding="fp8",
                ))
        if layer == index:
            name = prefix + ".indexer."
            keys = np.asarray(state.index_keys[source], dtype=np.float32).reshape(-1, c.index_head_dim)
            selected = np.empty(0, np.int64)
            if len(keys):
                query = self.linear(name + "wq_b.weight", qr).reshape(c.index_n_heads, c.index_head_dim)
                query = self.activation(rotary(query, width, angle), encoding="fp4")
                scores = index_scores(query, keys, self.linear(name + "weights_proj.weight", x))
                candidate = c.candidate_source_layer_id
                if layer == candidate:
                    shared["candidates"] = candidate_mask(scores, c.candidate_topk_blocks, c.candidate_block_size)
                elif 0 <= candidate < layer:
                    scores = np.where(shared["candidates"], scores, -np.inf)
                selected = np.sort(top_indices(scores, c.index_topk))
            shared[index] = selected
            if emit:
                emit(layer, "index_keys", keys)
        selected = shared[index]
        if emit:
            emit(layer, "index_selected", selected)
        return [state.compressed[source][i] for i in selected]

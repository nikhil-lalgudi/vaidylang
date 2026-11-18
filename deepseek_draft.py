"""Reduced CPU DSpark reference.

Equations follow DeepSeek's MIT-licensed inference/model.py at
314113f8a7dadb388215396ba5fbee2c610a04d6 (Copyright (c) 2023 DeepSeek).
See LICENSE.DeepSeek.
"""

from dataclasses import dataclass

import numpy as np

from deepseek_ops import hc_post, hc_pre, rms_norm, rope_frequencies, rotary, sparse_attention


@dataclass(frozen=True)
class DraftOutput:
    ids: np.ndarray
    logits: np.ndarray
    confidence: np.ndarray


class DSparkDraft:
    """Greedy draft over the real target feature window and one verified seed ID."""

    def __init__(self, engine):
        self.engine = engine
        self.config = config = engine.config
        if not config.num_nextn_predict_layers or not config.dspark_block_size:
            raise ValueError("This checkpoint has no DSpark stages")
        self.inv_freq = rope_frequencies(config)

    def propose(self, history, limit, state):
        # Upstream start_pos=0 only initializes the draft cache. Near the context
        # limit the complete fixed draft block may no longer fit in the RoPE table.
        if not limit or state.length < 2:
            return []
        if state.length + self.config.dspark_block_size > self.config.max_position_embeddings:
            return []
        if len(history) != state.length + 1:
            raise ValueError("DSpark needs one verified seed token beyond the cached target prefix")
        return self.forward(state, history[-1]).ids[:limit].tolist()

    def forward(self, state, previous_token):
        config, engine = self.config, self.engine
        width = config.dspark_block_size
        if not 2 <= state.length <= config.max_position_embeddings - width:
            raise ValueError("DSpark forward requires a seeded prefix and space for its fixed block")
        if not 0 <= previous_token < config.vocab_size:
            raise ValueError("DSpark seed token is outside the vocabulary")
        start = max(0, state.length - config.sliding_window)
        if len(state.target_features) != state.length:
            raise ValueError("DSpark requires the target features of the verified inputs")
        # DSpark's persistent cache in the published model holds projections of
        # target features only. Rebuild the visible rows from the committed prefix
        # so a rejected or failed proposal cannot leave stale draft projections.
        main = self._norm("mtp.0.main_norm", engine.linear(
            "mtp.0.main_proj.weight", np.stack(state.target_features[start:])
        ))
        main_positions = np.arange(start, state.length)
        draft_positions = np.arange(state.length, state.length + width)
        draft_ids = np.full(width, config.dspark_noise_token_id, dtype=np.int64)
        draft_ids[0] = previous_token
        x = np.repeat(engine.dense_weight("embed.weight")[draft_ids, None, :], config.hc_mult, axis=1)
        pre = np.zeros((width, config.hc_mult), dtype=np.float32)
        pre[:, 0] = 1.0
        for stage in range(config.num_nextn_predict_layers):
            prefix = f"mtp.{stage}"
            attn_pre, post, comb = engine.hc_mixes(prefix, "attn", x)
            query = self._norm(f"{prefix}.attn_norm", hc_pre(x, pre))
            attended = self._attention(prefix, query, main, main_positions, draft_positions)
            x = hc_post(attended, x, post, comb)

            pre, post, comb = engine.hc_mixes(prefix, "ffn", x)
            hidden = self._norm(f"{prefix}.ffn_norm", hc_pre(x, attn_pre))
            expert_id = config.num_hidden_layers + stage
            values = np.stack([engine.moe(prefix, row, expert_id) for row in hidden])
            x = hc_post(values, x, post, comb)
        return self._head(hc_pre(x, pre), previous_token)

    def _norm(self, name, x):
        return rms_norm(x, self.engine.dense_weight(f"{name}.weight"), self.config.rms_norm_eps)

    def _attention(self, prefix, x, main, main_positions, draft_positions):
        config, engine = self.config, self.engine
        prefix = f"{prefix}.attn"
        rd = config.qk_rope_head_dim
        main_angles = main_positions[:, None] * self.inv_freq
        draft_angles = draft_positions[:, None] * self.inv_freq

        q = self._norm(f"{prefix}.q_norm", engine.linear(f"{prefix}.wq_a.weight", x))
        q = engine.linear(f"{prefix}.wq_b.weight", q).reshape(
            len(x), config.num_attention_heads, config.head_dim
        )
        q = rotary(q, rd, draft_angles[:, None, :])

        def kv(inputs, angles):
            values = self._norm(f"{prefix}.kv_norm", engine.linear(f"{prefix}.wkv.weight", inputs))
            return engine.activation(rotary(values, rd, angles))

        keys = np.concatenate((kv(main, main_angles), kv(x, draft_angles)))
        sink = engine.dense_weight(f"{prefix}.attn_sink")
        # The draft attends to every noise slot and its target prefix window.
        output = np.stack([sparse_attention(row, keys, sink) for row in q])
        output = rotary(output, rd, draft_angles[:, None, :], inverse=True)
        grouped = engine.activation(output.reshape(len(x), config.o_groups, -1))
        wo_a = engine.dense_weight(f"{prefix}.wo_a.weight").reshape(
            config.o_groups, config.o_lora_rank, -1
        )
        projected = np.einsum("sgd,grd->sgr", grouped, wo_a).reshape(len(x), -1)
        return engine.linear(f"{prefix}.wo_b.weight", projected)

    def _head(self, x, previous_token):
        engine = self.engine
        prefix = f"mtp.{self.config.num_nextn_predict_layers - 1}"
        logits = engine.linear("head.weight", self._norm(f"{prefix}.norm", x))
        markov_embeddings = engine.dense_weight(f"{prefix}.markov_head.embed.weight")
        ids, embeddings = [], []
        for row in logits:
            embedding = markov_embeddings[previous_token]
            row += engine.linear(f"{prefix}.markov_head.head.weight", embedding)
            previous_token = int(row.argmax())
            ids.append(previous_token)
            embeddings.append(embedding)
        confidence_inputs = np.concatenate((x, np.stack(embeddings)), axis=-1)
        confidence = engine.linear(f"{prefix}.confidence_head.proj.weight", confidence_inputs)[:, 0]
        return DraftOutput(np.asarray(ids), logits, confidence)

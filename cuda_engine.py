"""Single-sequence Qwen3 inference on CUDA."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from checkpoint import Checkpoint
from engine import Config, weight_shapes


@dataclass
class CudaState:
    keys: torch.Tensor
    values: torch.Tensor
    position: torch.Tensor
    remaining: int
    staged_by: object | None = None

    @property
    def capacity(self):
        return self.keys.shape[2]


def rms_norm(x, weight, eps):
    fp32 = x.float()
    normalized = fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


def rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def sdpa(q, k, v, *, causal, mask=None):
    attention = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        is_causal=causal,
        attn_mask=mask,
        enable_gqa=True,
    )
    return attention.squeeze(0).transpose(0, 1)


class CudaQwen3Engine:
    @torch.inference_mode()
    def __init__(
        self, config: Config, weights: dict, *, dtype, device="cuda", fused_rms=False,
        packed_projections=False, unpacked_prefill=False, dense_prefill=0, decode_splits=0,
        fused_decode_pointwise=False, tiled_decode=False,
    ):
        if unpacked_prefill and not packed_projections:
            raise ValueError("Unpacked prefill requires packed decode projections")
        if dense_prefill < 0 or decode_splits < 0:
            raise ValueError("Dense prefill threshold and decode split count must be nonnegative")
        if (dense_prefill or decode_splits) and dtype != torch.bfloat16:
            raise ValueError("FlashAttention tuning requires BF16")
        if tiled_decode and (dtype != torch.bfloat16 or decode_splits):
            raise ValueError("Tiled decode attention requires BF16 and does not use FlashAttention splits")
        if tiled_decode and (config.head_dim > 256 or config.head_dim & (config.head_dim - 1)):
            raise ValueError("Tiled decode attention requires a power-of-two head dimension up to 256")
        if fused_decode_pointwise and not fused_rms:
            raise ValueError("Fused decode pointwise operations require fused RMSNorm")
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.packed_projections = packed_projections
        self.unpacked_prefill = unpacked_prefill
        self.dense_prefill = dense_prefill
        self.decode_splits = decode_splits
        self.fused_decode_pointwise = fused_decode_pointwise
        self.decode_attention = None
        if tiled_decode:
            from cuda_attention import tiled_decode_attention

            self.decode_attention = tiled_decode_attention
        self.qkv_widths = (
            config.num_attention_heads * config.head_dim,
            config.num_key_value_heads * config.head_dim,
            config.num_key_value_heads * config.head_dim,
        )
        self.norm = rms_norm
        if fused_rms:
            if dtype != torch.bfloat16:
                raise ValueError("Fused RMSNorm requires BF16")
            from cuda_kernels import fused_rms_norm

            self.norm = fused_rms_norm
        if fused_decode_pointwise:
            from cuda_kernels import fused_qk_rms_norm, fused_residual_rms_norm, fused_swiglu

            self.norm_qk = fused_qk_rms_norm
            self.norm_residual = fused_residual_rms_norm
            self.swiglu = fused_swiglu
        self.flash_attention = None
        if dtype == torch.bfloat16:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache

            self.flash_attention = flash_attn_with_kvcache
        elif dtype != torch.float32:
            raise ValueError("Use float32 or bfloat16 for CUDA inference")

        weights = {name: torch.as_tensor(value, device=device, dtype=dtype) for name, value in weights.items()}
        self.embeddings = weights["model.embed_tokens.weight"]
        self.lm_head = self.embeddings if config.tie_word_embeddings else weights["lm_head.weight"]
        self.final_norm = weights["model.norm.weight"]
        self.layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}."
            layer = {
                name.removeprefix(prefix): value
                for name, value in weights.items() if name.startswith(prefix)
            }
            if packed_projections:
                layer["self_attn.qkv_proj.weight"] = torch.cat([
                    layer.pop(f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")
                ])
                layer["mlp.gate_up_proj.weight"] = torch.cat([
                    layer.pop(f"mlp.{name}_proj.weight") for name in ("gate", "up")
                ])
                if unpacked_prefill:
                    q, k, v = layer["self_attn.qkv_proj.weight"].split(self.qkv_widths)
                    gate, up = layer["mlp.gate_up_proj.weight"].chunk(2)
                    layer.update({
                        "self_attn.q_proj.weight": q,
                        "self_attn.k_proj.weight": k,
                        "self_attn.v_proj.weight": v,
                        "mlp.gate_proj.weight": gate,
                        "mlp.up_proj.weight": up,
                    })
            self.layers.append(layer)

        self.positions = torch.arange(config.max_position_embeddings, device=device)
        frequency = config.rope_theta ** (-torch.arange(
            0, config.head_dim, 2, dtype=torch.float32, device=device,
        ) / config.head_dim)
        angles = self.positions[:, None].float() * frequency
        self.cos = angles.cos().to(dtype)
        self.sin = angles.sin().to(dtype)
        if dense_prefill:
            from sgl_kernel import rotary_embedding

            self.prefill_rotary = rotary_embedding
            self.cos_sin = torch.cat((self.cos, self.sin), dim=-1)

    @classmethod
    def from_checkpoint(
        cls, path: Path, *, dtype, device="cuda", fused_rms=False,
        packed_projections=False, unpacked_prefill=False, dense_prefill=0, decode_splits=0,
        fused_decode_pointwise=False, tiled_decode=False,
    ):
        checkpoint = Checkpoint.open(path)
        config = Config.from_mapping(checkpoint.config)
        shapes = weight_shapes(config)
        weights = {}
        for shard, names in checkpoint.group_tensors(shapes).items():
            with safe_open(shard, framework="pt", device=str(device)) as source:
                for name in names:
                    tensor = source.get_tensor(name)
                    if tuple(tensor.shape) != shapes[name]:
                        raise ValueError(f"Incompatible weight {name}")
                    weights[name] = tensor
        return cls(
            config, weights, device=device, dtype=dtype,
            fused_rms=fused_rms, packed_projections=packed_projections, unpacked_prefill=unpacked_prefill,
            dense_prefill=dense_prefill, decode_splits=decode_splits,
            fused_decode_pointwise=fused_decode_pointwise,
            tiled_decode=tiled_decode,
        )

    @torch.inference_mode()
    def allocate_cache(self, capacity):
        c = self.config
        if not 1 <= capacity <= c.max_position_embeddings:
            raise ValueError("KV capacity is outside the model context limit")
        shape = (c.num_hidden_layers, 1, capacity, c.num_key_value_heads, c.head_dim)
        # FlashAttention can load padding in a partial tile before applying the mask.
        keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        position = torch.zeros(1, dtype=torch.int32, device=self.device)
        return CudaState(keys, values, position, capacity)

    @torch.inference_mode()
    def prefill(self, token_ids, capacity):
        state = self.allocate_cache(capacity)
        return self.prefill_into(token_ids, state), state

    @torch.inference_mode()
    def prefill_into(self, token_ids, state):
        if state.staged_by is not None:
            raise ValueError("A speculative block is staged")
        ids = torch.as_tensor(token_ids)
        if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected a nonempty one-dimensional sequence of token IDs")
        if ids.numel() > state.capacity:
            raise ValueError("KV cache capacity exceeded")
        if ids.device.type == "cpu" and (ids.min() < 0 or ids.max() >= self.config.vocab_size):
            raise ValueError("Token ID is outside the vocabulary")
        ids = ids.to(device=self.device)
        state.position.zero_()
        logits = self._forward(ids, state, prefill=True)
        state.remaining = state.capacity - ids.numel()
        return logits

    @torch.inference_mode()
    def decode(self, token_id, state):
        if state.staged_by is not None:
            raise ValueError("A speculative block is staged")
        if state.remaining <= 0:
            raise ValueError("KV cache capacity exceeded")
        if isinstance(token_id, int) and not 0 <= token_id < self.config.vocab_size:
            raise ValueError("Token ID is outside the vocabulary")
        ids = torch.as_tensor(token_id, device=self.device)
        if ids.numel() != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected one integer token ID")
        logits = self._forward(ids.view(1), state, prefill=False)
        state.remaining -= 1
        return logits

    def _forward(self, ids, state, *, prefill, block_backend=None):
        c = self.config
        steps = ids.numel()
        packed = self.packed_projections and not (prefill and self.unpacked_prefill)
        dense = prefill and self.dense_prefill and steps >= self.dense_prefill
        block = block_backend is not None
        tiled = not prefill and self.decode_attention is not None and block_backend != "flash"
        fused = not prefill and (not block or tiled) and self.fused_decode_pointwise
        if self.flash_attention is None:
            positions = state.position + self.positions[:steps]
            cos = self.cos.index_select(0, positions).unsqueeze(1)
            sin = self.sin.index_select(0, positions).unsqueeze(1)
            if prefill:
                mask = None
            elif block:
                mask = self.positions[:state.capacity] <= positions[:, None]
            else:
                mask = self.positions[:state.capacity] <= state.position

        hidden = self.embeddings.index_select(0, ids)
        for index, w in enumerate(self.layers):
            if fused and index:
                hidden, x = self.norm_residual(hidden, update, w["input_layernorm.weight"], c.rms_norm_eps)
            else:
                x = self.norm(hidden, w["input_layernorm.weight"], c.rms_norm_eps)
            if packed:
                q, k, v = F.linear(x, w["self_attn.qkv_proj.weight"]).split(self.qkv_widths, dim=-1)
                if not (block and tiled and fused):
                    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            else:
                q = F.linear(x, w["self_attn.q_proj.weight"])
                k = F.linear(x, w["self_attn.k_proj.weight"])
                v = F.linear(x, w["self_attn.v_proj.weight"])
            q = q.view(steps, c.num_attention_heads, c.head_dim)
            k = k.view(steps, c.num_key_value_heads, c.head_dim)
            v = v.view(steps, c.num_key_value_heads, c.head_dim)
            q_weight, k_weight = w["self_attn.q_norm.weight"], w["self_attn.k_norm.weight"]
            if fused:
                q, k = self.norm_qk(q, k, q_weight, k_weight, c.rms_norm_eps)
            else:
                q = self.norm(q, q_weight, c.rms_norm_eps)
                k = self.norm(k, k_weight, c.rms_norm_eps)

            keys, values = state.keys[index], state.values[index]
            if dense:
                self.prefill_rotary(self.positions[:steps], q, k, c.head_dim, self.cos_sin)
                keys[:, :steps].copy_(k.unsqueeze(0))
                values[:, :steps].copy_(v.unsqueeze(0))
                joined = self.flash_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True)
            elif tiled:
                joined = self.decode_attention(q, keys, values, k, v, state.position, self.cos, self.sin)
            elif self.flash_attention is not None:
                # The kernel appends K/V and applies Qwen's split-half RoPE from the device position.
                joined = self.flash_attention(
                    q.unsqueeze(0), keys, values, k=k.unsqueeze(0), v=v.unsqueeze(0),
                    cache_seqlens=state.position, rotary_cos=self.cos, rotary_sin=self.sin,
                    rotary_interleaved=False, causal=prefill or block,
                    num_splits=0 if prefill or block else self.decode_splits,
                )
            else:
                q, k = rotate(q, cos, sin), rotate(k, cos, sin)
                keys[0].index_copy_(0, positions, k)
                values[0].index_copy_(0, positions, v)
                joined = sdpa(
                    q, k if prefill else keys[0], v if prefill else values[0],
                    causal=prefill, mask=mask,
                )

            update = F.linear(joined.reshape(steps, -1), w["self_attn.o_proj.weight"])
            if fused:
                hidden, x = self.norm_residual(hidden, update, w["post_attention_layernorm.weight"], c.rms_norm_eps)
            else:
                hidden = hidden + update
                x = self.norm(hidden, w["post_attention_layernorm.weight"], c.rms_norm_eps)
            if packed:
                gate, up = F.linear(x, w["mlp.gate_up_proj.weight"]).chunk(2, dim=-1)
            else:
                gate = F.linear(x, w["mlp.gate_proj.weight"])
                up = F.linear(x, w["mlp.up_proj.weight"])
            activation = self.swiglu(gate, up) if fused else F.silu(gate) * up
            update = F.linear(activation, w["mlp.down_proj.weight"])
            if not fused:
                hidden = hidden + update

        state.position.add_(steps)
        if fused:
            _, normalized = self.norm_residual(hidden, update, self.final_norm, c.rms_norm_eps)
            if not block:
                normalized = normalized[0]
        else:
            normalized = self.norm(hidden if block else hidden[-1], self.final_norm, c.rms_norm_eps)
        return F.linear(normalized, self.lm_head)


def _capture_decode(engine, state, start, step):
    # Use a separate warmup stream and keep all cache addresses stable for replay.
    current = torch.cuda.current_stream(engine.device)
    warmup = torch.cuda.Stream(device=engine.device)
    warmup.wait_stream(current)
    with torch.cuda.stream(warmup):
        for _ in range(3):
            state.position.fill_(start)
            step()
    current.wait_stream(warmup)
    state.position.fill_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = step()
    return graph, logits


class DecodeGraph:
    """Replay one decode step, including argmax, with stable cache and token addresses."""

    @torch.inference_mode()
    def __init__(self, engine, state, primer):
        self.engine = engine
        self.state = state
        self.token = torch.empty(1, dtype=torch.long, device=engine.device)
        self.prefill(primer)
        if not state.remaining:
            raise ValueError("Decode graph requires room for one more token")

        self.graph, self.logits = _capture_decode(engine, state, len(primer), self._step)

    def _step(self):
        logits = self.engine._forward(self.token, self.state, prefill=False)
        torch.argmax(logits, dim=0, out=self.token.view(()))
        return logits

    @torch.inference_mode()
    def prefill(self, ids):
        logits = self.engine.prefill_into(ids, self.state)
        torch.argmax(logits, dim=0, out=self.token.view(()))
        return logits

    @torch.inference_mode()
    def decode(self, token):
        if self.state.staged_by is not None:
            raise ValueError("A speculative block is staged")
        if not self.state.remaining:
            raise ValueError("KV cache capacity exceeded")
        if token is not self.token:
            self.token.copy_(torch.as_tensor(token, device=self.token.device))
        self.graph.replay()
        self.state.remaining -= 1
        return self.logits


class BlockDecodeGraph:
    """Stage a causal block and advance the logical cache by its verified prefix."""

    @torch.inference_mode()
    def __init__(self, engine, state, primer, block_size, *, backend="flash"):
        if backend not in ("flash", "tiled"):
            raise ValueError("Use flash or tiled verification")
        if backend == "tiled" and engine.decode_attention is None:
            raise ValueError("Tiled verification requires a tiled decode engine")
        if block_size < 2 or state.capacity - len(primer) < block_size:
            raise ValueError("Decode block must contain at least two tokens and fit in cache capacity")
        self.engine = engine
        self.state = state
        self.block_size = block_size
        self.backend = backend
        self._owner = object()
        engine.prefill_into(primer, state)
        self.tokens = torch.full((block_size,), int(primer[-1]), dtype=torch.long, device=engine.device)
        self.predictions = torch.empty_like(self.tokens)
        self.graph, self.logits = _capture_decode(engine, state, len(primer), self._step)

    def _step(self):
        logits = self.engine._forward(self.tokens, self.state, prefill=False, block_backend=self.backend)
        torch.argmax(logits, dim=-1, out=self.predictions)
        return logits

    @torch.inference_mode()
    def stage(self, tokens):
        if self.state.staged_by is not None:
            raise ValueError("A speculative block is staged")
        if self.state.remaining < self.block_size:
            raise ValueError("KV cache capacity exceeded")
        ids = torch.as_tensor(tokens)
        if ids.shape != self.tokens.shape or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Expected a block of {self.block_size} integer IDs")
        if ids.device.type == "cpu" and (ids.min() < 0 or ids.max() >= self.engine.config.vocab_size):
            raise ValueError("Token ID is outside the vocabulary")
        self.tokens.copy_(ids, non_blocking=True)
        self.graph.replay()
        self.state.staged_by = self._owner
        return self.logits

    @torch.inference_mode()
    def commit(self, count):
        if self.state.staged_by is not self._owner:
            raise ValueError("This graph has no staged block")
        if type(count) is not int or not 0 <= count <= self.block_size:
            raise ValueError("Invalid speculative prefix length")
        # The suffix remains outside the visible prefix. Both attention paths
        # mask it on reads and overwrite each slot before admitting it again.
        self.state.position.sub_(self.block_size - count)
        self.state.remaining -= count
        self.state.staged_by = None


class GraphQwen3Engine(CudaQwen3Engine):
    def prefill(self, token_ids, capacity):
        if capacity <= len(token_ids):
            return super().prefill(token_ids, capacity)
        graph = DecodeGraph(self, self.allocate_cache(capacity), token_ids)
        return graph.prefill(token_ids), graph

    def decode(self, token_id, state):
        if isinstance(state, DecodeGraph):
            return state.decode(token_id)
        return super().decode(token_id, state)

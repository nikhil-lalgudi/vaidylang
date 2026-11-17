"""Single-sequence Qwen3 inference on CUDA."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from engine import Config, weight_shapes


@dataclass
class CudaState:
    keys: torch.Tensor
    values: torch.Tensor
    position: torch.Tensor
    remaining: int

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
    def __init__(self, config: Config, weights: dict, *, dtype, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
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
            self.layers.append({
                name.removeprefix(prefix): value
                for name, value in weights.items() if name.startswith(prefix)
            })

        self.positions = torch.arange(config.max_position_embeddings, device=device)
        frequency = config.rope_theta ** (-torch.arange(
            0, config.head_dim, 2, dtype=torch.float32, device=device,
        ) / config.head_dim)
        angles = self.positions[:, None].float() * frequency
        self.cos = angles.cos().to(dtype)
        self.sin = angles.sin().to(dtype)

    @classmethod
    def from_checkpoint(cls, path: Path, *, dtype, device="cuda"):
        config = Config.from_file(path / "config.json")
        weights = {}
        with safe_open(path / "model.safetensors", framework="pt", device=str(device)) as source:
            for name, shape in weight_shapes(config).items():
                tensor = source.get_tensor(name)
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"Incompatible weight {name}")
                weights[name] = tensor
        return cls(config, weights, device=device, dtype=dtype)

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

    def _forward(self, ids, state, *, prefill):
        c = self.config
        steps = ids.numel()
        if self.flash_attention is None:
            positions = state.position + self.positions[:steps]
            cos = self.cos.index_select(0, positions).unsqueeze(1)
            sin = self.sin.index_select(0, positions).unsqueeze(1)
            mask = None if prefill else self.positions[:state.capacity] <= state.position

        hidden = self.embeddings.index_select(0, ids)
        for index, w in enumerate(self.layers):
            x = rms_norm(hidden, w["input_layernorm.weight"], c.rms_norm_eps)
            q = F.linear(x, w["self_attn.q_proj.weight"]).view(steps, c.num_attention_heads, c.head_dim)
            k = F.linear(x, w["self_attn.k_proj.weight"]).view(steps, c.num_key_value_heads, c.head_dim)
            v = F.linear(x, w["self_attn.v_proj.weight"]).view_as(k)
            q = rms_norm(q, w["self_attn.q_norm.weight"], c.rms_norm_eps)
            k = rms_norm(k, w["self_attn.k_norm.weight"], c.rms_norm_eps)

            keys, values = state.keys[index], state.values[index]
            if self.flash_attention is not None:
                # The kernel appends K/V and applies Qwen's split-half RoPE from the device position.
                joined = self.flash_attention(
                    q.unsqueeze(0), keys, values, k=k.unsqueeze(0), v=v.unsqueeze(0),
                    cache_seqlens=state.position, rotary_cos=self.cos, rotary_sin=self.sin,
                    rotary_interleaved=False, causal=prefill,
                )
            else:
                q, k = rotate(q, cos, sin), rotate(k, cos, sin)
                keys[0].index_copy_(0, positions, k)
                values[0].index_copy_(0, positions, v)
                joined = sdpa(
                    q, k if prefill else keys[0], v if prefill else values[0],
                    causal=prefill, mask=mask,
                )

            hidden = hidden + F.linear(joined.reshape(steps, -1), w["self_attn.o_proj.weight"])
            x = rms_norm(hidden, w["post_attention_layernorm.weight"], c.rms_norm_eps)
            gate = F.silu(F.linear(x, w["mlp.gate_proj.weight"]))
            up = F.linear(x, w["mlp.up_proj.weight"])
            hidden = hidden + F.linear(gate * up, w["mlp.down_proj.weight"])

        state.position.add_(steps)
        return F.linear(rms_norm(hidden[-1], self.final_norm, c.rms_norm_eps), self.lm_head)


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

        # https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs
        current = torch.cuda.current_stream(engine.device)
        warmup = torch.cuda.Stream(device=engine.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(3):
                state.position.fill_(len(primer))
                self._step()
        current.wait_stream(warmup)
        state.position.fill_(len(primer))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()

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
        if not self.state.remaining:
            raise ValueError("KV cache capacity exceeded")
        if token is not self.token:
            self.token.copy_(torch.as_tensor(token, device=self.token.device))
        self.graph.replay()
        self.state.remaining -= 1
        return self.logits


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

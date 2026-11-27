"""Single-sequence, float32 reference engine for dense Qwen3."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from checkpoint import Checkpoint


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        sizes = (
            self.vocab_size, self.hidden_size, self.intermediate_size,
            self.num_hidden_layers, self.num_attention_heads,
            self.num_key_value_heads, self.head_dim, self.max_position_embeddings,
        )
        if any(size <= 0 for size in sizes):
            raise ValueError("Model dimensions must be positive")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("RoPE requires even head dimensions and GQA requires equal head groups")

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        return cls.from_mapping(json.loads(path.read_text()))

    @classmethod
    def from_mapping(cls, values) -> "Config":
        if (
            values.get("model_type") != "qwen3"
            or values.get("hidden_act") != "silu"
            or values.get("attention_bias")
            or values.get("use_sliding_window")
            or values.get("rope_scaling") is not None
            or values.get("rope_parameters") is not None
        ):
            raise ValueError("Use a dense Qwen3 checkpoint with standard RoPE and full attention")
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


@dataclass
class State:
    keys: np.ndarray
    values: np.ndarray
    length: int = 0


def weight_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    key_value = config.num_key_value_heads * config.head_dim
    intermediate = config.intermediate_size
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, hidden)
    per_layer = {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (query, hidden),
        "self_attn.k_proj.weight": (key_value, hidden),
        "self_attn.v_proj.weight": (key_value, hidden),
        "self_attn.o_proj.weight": (hidden, query),
        "self_attn.q_norm.weight": (config.head_dim,),
        "self_attn.k_norm.weight": (config.head_dim,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    for index in range(config.num_hidden_layers):
        shapes.update({f"model.layers.{index}.{name}": shape for name, shape in per_layer.items()})
    return shapes


def read_weights(path: Path, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Read the float tensors this engine uses from a single safetensors file."""
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    header_size = int.from_bytes(mapped[:8].tobytes(), "little")
    data_start = 8 + header_size
    if not 0 < header_size <= 100_000_000 or data_start > mapped.size:
        raise ValueError(f"Invalid safetensors header in {path.name}")
    header = json.loads(mapped[8:data_start].tobytes())
    dtypes = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"), "BF16": np.dtype("<u2")}
    weights = {}
    for name, shape in shapes.items():
        info = header.get(name)
        if info is None or tuple(info["shape"]) != shape or info["dtype"] not in dtypes:
            raise ValueError(f"Missing or incompatible weight {name}")
        start, end = info["data_offsets"]
        dtype = dtypes[info["dtype"]]
        if (
            not 0 <= start <= end <= mapped.size - data_start
            or end - start != math.prod(shape) * dtype.itemsize
        ):
            raise ValueError(f"Invalid data offsets for {name}")
        raw = mapped[data_start + start:data_start + end].view(dtype).reshape(shape)
        if info["dtype"] == "BF16":
            bits = raw.astype(np.uint32)
            bits <<= 16
            weights[name] = bits.view(np.float32)
        else:
            weights[name] = raw.astype(np.float32, copy=False)
    return weights


def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * scale


def rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    first, second = np.split(x, 2, axis=-1)
    return x * cos + np.concatenate((-second, first), axis=-1) * sin


class Qwen3Engine:
    def __init__(self, config: Config, weights: dict[str, np.ndarray]):
        self.config = config
        self.embeddings = weights["model.embed_tokens.weight"]
        self.lm_head = self.embeddings if config.tie_word_embeddings else weights["lm_head.weight"]
        self.final_norm = weights["model.norm.weight"]
        self.layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}."
            self.layers.append({name.removeprefix(prefix): weights[name] for name in weights if name.startswith(prefix)})
        self.inv_freq = config.rope_theta ** (-np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim)

    @classmethod
    def from_checkpoint(cls, directory: Path) -> "Qwen3Engine":
        checkpoint = directory if isinstance(directory, Checkpoint) else Checkpoint.open(directory)
        config = Config.from_mapping(checkpoint.config)
        shapes = weight_shapes(config)
        weights = {}
        for shard, names in checkpoint.group_tensors(shapes).items():
            weights.update(read_weights(shard, {name: shapes[name] for name in names}))
        return cls(config, weights)

    def prefill(self, token_ids, capacity: int) -> tuple[np.ndarray, State]:
        """Start a request and return logits for the first generated token."""
        config = self.config
        if not 1 <= capacity <= config.max_position_embeddings:
            raise ValueError("KV capacity is outside the model context limit")
        shape = (config.num_hidden_layers, config.num_key_value_heads, capacity, config.head_dim)
        state = State(np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32))
        return self._forward(token_ids, state), state

    def decode(self, token_id: int, state: State) -> np.ndarray:
        """Append one token to the request and return logits for its successor."""
        return self._forward([token_id], state)

    def _forward(self, token_ids, state: State) -> np.ndarray:
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty one-dimensional sequence of token IDs")
        config = self.config
        if ids.min() < 0 or ids.max() >= config.vocab_size:
            raise ValueError("Token ID is outside the vocabulary")
        start, end = state.length, state.length + ids.size
        if end > state.keys.shape[2]:
            raise ValueError("KV cache capacity exceeded")
        positions = np.arange(start, end)
        angles = positions.astype(np.float32)[:, None] * self.inv_freq
        angles = np.concatenate((angles, angles), axis=-1)
        cos, sin = np.cos(angles), np.sin(angles)
        # A decode query can see its own key and every key already in the cache.
        mask = np.arange(end) <= positions[:, None] if ids.size > 1 else None
        hidden = self.embeddings[ids]
        for index, weights in enumerate(self.layers):
            residual = hidden
            hidden = rms_norm(hidden, weights["input_layernorm.weight"], config.rms_norm_eps)
            hidden = residual + self._attention(hidden, weights, state, index, cos, sin, mask, start, end)
            residual = hidden
            hidden = rms_norm(hidden, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
            gate = hidden @ weights["mlp.gate_proj.weight"].T
            up = hidden @ weights["mlp.up_proj.weight"].T
            with np.errstate(over="ignore"):
                silu_gate = gate / (1.0 + np.exp(-gate))
            hidden = residual + (silu_gate * up) @ weights["mlp.down_proj.weight"].T
        logits = self.lm_head @ rms_norm(hidden[-1], self.final_norm, config.rms_norm_eps)
        state.length = end
        return logits

    def _attention(self, x, weights, state, layer, cos, sin, mask, start, end):
        config = self.config
        q_heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        steps = end - start
        q = (x @ weights["self_attn.q_proj.weight"].T).reshape(steps, q_heads, dim)
        k = (x @ weights["self_attn.k_proj.weight"].T).reshape(steps, kv_heads, dim)
        v = (x @ weights["self_attn.v_proj.weight"].T).reshape(steps, kv_heads, dim)
        q = rms_norm(q, weights["self_attn.q_norm.weight"], config.rms_norm_eps).transpose(1, 0, 2)
        k = rms_norm(k, weights["self_attn.k_norm.weight"], config.rms_norm_eps).transpose(1, 0, 2)
        query = rotate(q, cos, sin).reshape(kv_heads, q_heads // kv_heads, steps, dim)
        state.keys[layer, :, start:end] = rotate(k, cos, sin)
        state.values[layer, :, start:end] = v.transpose(1, 0, 2)
        keys = state.keys[layer, :, None, :end]
        values = state.values[layer, :, None, :end]
        scores = (query @ keys.swapaxes(-1, -2)) * (1.0 / math.sqrt(dim))
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        joined = (scores @ values).reshape(q_heads, steps, dim).transpose(1, 0, 2).reshape(steps, q_heads * dim)
        return joined @ weights["self_attn.o_proj.weight"].T


def load_engine(directory, *, with_vision=False, cpu_backend="python"):
    """Load a supported CPU engine from the root architecture in config.json."""
    if cpu_backend not in ("auto", "native", "python"):
        raise ValueError(f"Unknown CPU backend {cpu_backend!r}")
    checkpoint = Checkpoint.open(directory)
    if checkpoint.model_type == "qwen3":
        if with_vision:
            raise ValueError("Qwen3 CPU engine supports text prompts")
        if cpu_backend != "python":
            from native_cpu import NativeQwen3Engine, load_library

            library = load_library()
            if library is not None:
                return NativeQwen3Engine(checkpoint, Config.from_mapping(checkpoint.config), library)
            if cpu_backend == "native":
                raise ValueError("Build the C++ CPU library or set VAIDYLANG_CPU_LIBRARY")
        return Qwen3Engine.from_checkpoint(checkpoint)
    if checkpoint.model_type == "deepseek_v41":
        if cpu_backend == "native":
            raise ValueError("The native engine backend requires a Qwen3 checkpoint")
        from deepseek import DeepSeekEngine

        return DeepSeekEngine.from_checkpoint(checkpoint, with_vision=with_vision)
    raise ValueError(f"Unsupported CPU model_type {checkpoint.model_type!r}")


def tiny_weights() -> tuple[Config, dict[str, np.ndarray]]:
    config = Config(
        vocab_size=64, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
    )
    rng = np.random.default_rng(7)
    weights = {}
    for name, shape in weight_shapes(config).items():
        if len(shape) == 1:
            weights[name] = (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
        else:
            weights[name] = (rng.standard_normal(shape) / math.sqrt(shape[-1])).astype(np.float32)
    return config, weights

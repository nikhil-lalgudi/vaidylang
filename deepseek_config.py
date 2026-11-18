"""DeepSeek V4.1 text configuration and public checkpoint tensor inventory.

Topology and tensor names follow DeepSeek's model.py and convert.py at the
revision recorded in deepseek-cpu-reference.md. See LICENSE.DeepSeek.
"""

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekConfig:
    root_config: dict
    text_config: dict
    quantization_config: dict
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int
    sliding_window: int
    compress_ratios: tuple
    kv_source_layer_ids: tuple = ()
    index_source_layer_ids: tuple = ()
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    rope_scaling: dict | None = None
    rms_norm_eps: float = 1e-20
    scoring_func: str = "sqrtsoftplus"
    gate_temperature: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    swiglu_limit: float = 0.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = ()
    engram_num_embeddings: tuple = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 0
    num_nextn_predict_layers: int = 0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple = ()
    dspark_markov_rank: int = 0
    dspark_n_routed_experts: int = 0
    dspark_num_experts_per_tok: int = 0

    @classmethod
    def from_mapping(cls, root):
        root = deepcopy(root)
        text = root.get("text_config")
        if root.get("model_type") != "deepseek_v41" or not isinstance(text, dict) or text.get("model_type") != "deepseek_v41_text":
            raise ValueError("Expected deepseek_v41 with nested deepseek_v41_text config")
        if text.get("hidden_act", "silu") != "silu" or text.get("attention_bias") or text.get("tie_word_embeddings"):
            raise ValueError("DeepSeek CPU text execution requires SwiGLU, bias-free attention, and an untied head")
        quant = root.get("quantization_config") or {}
        if quant and (
            quant.get("quant_method") != "fp8" or quant.get("scale_fmt") != "ue8m0"
            or quant.get("activation_scheme") != "dynamic" or quant.get("expert_dtype") != "fp4"
            or quant.get("weight_block_size") != [32, 32]
        ):
            raise ValueError("DeepSeek quantized CPU execution requires dynamic FP8, FP4 experts and 32x32 E8M0 weight blocks")
        fields = {name: text[name] for name in cls.__dataclass_fields__ if name in text}
        for name in (
            "compress_ratios", "kv_source_layer_ids", "index_source_layer_ids",
            "engram_layer_ids", "engram_num_embeddings", "dspark_target_layer_ids",
        ):
            if name in fields:
                if not isinstance(fields[name], (list, tuple)) or any(type(i) is not int for i in fields[name]):
                    raise ValueError(f"DeepSeek {name} must be an integer sequence")
                fields[name] = tuple(fields[name])
        try:
            return cls(root_config=root, text_config=text, quantization_config=quant, **fields)
        except TypeError as error:
            raise ValueError(f"Invalid DeepSeek text config: {error}") from error

    def __post_init__(self):
        dimensions = (
            self.vocab_size, self.hidden_size, self.moe_intermediate_size,
            self.num_hidden_layers, self.num_attention_heads, self.head_dim, self.qk_rope_head_dim,
            self.q_lora_rank, self.o_lora_rank, self.o_groups, self.n_routed_experts,
            self.num_experts_per_tok, self.max_position_embeddings, self.sliding_window,
            self.hc_mult, self.hc_sinkhorn_iters,
        )
        if any(type(i) is not int or i <= 0 for i in dimensions):
            raise ValueError("DeepSeek model dimensions must be positive integers")
        if self.num_key_value_heads != 1 or self.num_attention_heads % self.o_groups:
            raise ValueError("DeepSeek uses a single KV latent and O groups must divide the query heads")
        if self.qk_rope_head_dim % 2 or self.qk_rope_head_dim > self.head_dim:
            raise ValueError("DeepSeek rotary tail must be even and no wider than the attention head")
        if self.n_shared_experts != 1 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("DeepSeek requires one shared expert and enough routed experts for top-k")
        if self.scoring_func not in ("sqrtsoftplus", "sigmoid", "softmax") or min(
            self.rms_norm_eps, self.hc_eps, self.gate_temperature,
        ) <= 0:
            raise ValueError("Invalid DeepSeek routing, normalization or hyper-connection config")
        if self.rope_scaling and self.rope_scaling.get("rope_type") != "yarn":
            raise ValueError("DeepSeek compressed attention supports YaRN rope_scaling")
        self._validate_schedules()

    def _validate_schedules(self):
        n = self.num_hidden_layers
        draft = self.num_nextn_predict_layers
        if draft < 0 or len(self.compress_ratios) != n + draft or any(r < 0 for r in self.compress_ratios):
            raise ValueError("compress_ratios needs one nonnegative entry per backbone and draft layer")
        if any(self.compress_ratios[n:]):
            raise ValueError("DSpark draft attention must be uncompressed")
        for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids", "dspark_target_layer_ids"):
            ids = getattr(self, key)
            if len(set(ids)) != len(ids) or any(i < 0 or i >= n for i in ids):
                raise ValueError(f"DeepSeek {key} must contain distinct backbone layer IDs")
        kv_sources = self.kv_source_layer_ids
        index_sources = self.index_source_layer_ids
        if set(kv_sources) - set(index_sources) or any(self.compress_ratios[i] == 0 for i in index_sources):
            raise ValueError("Compressed KV sources must own indexes and use a positive compression ratio")
        if index_sources and (
            min(self.index_n_heads, self.index_topk) <= 0 or self.index_head_dim < self.qk_rope_head_dim
        ):
            raise ValueError("DeepSeek index dimensions or rotary tail are invalid")
        for i, ratio in enumerate(self.compress_ratios[:n]):
            if ratio and any(owner is None or self.compress_ratios[owner] != ratio for owner in (
                cache_owner(kv_sources, i), cache_owner(index_sources, i),
            )):
                raise ValueError(f"Layer {i} needs preceding KV and index sources with compression ratio {ratio}")
        source = self.candidate_source_layer_id
        if source >= 0 and (
            source not in index_sources or min(self.candidate_topk_blocks, self.candidate_block_size) <= 0
            or any(self.compress_ratios[i] != self.compress_ratios[source] for i in index_sources if i > source)
        ):
            raise ValueError("Candidate filtering needs a valid source and matching downstream index ratios")
        if self.engram_layer_ids and (
            len(self.engram_layer_ids) != len(self.engram_num_embeddings)
            or self.engram_max_ngram_size < 2
            or min(self.engram_n_heads, self.engram_head_dim, self.engram_vocab_size,
                   self.engram_compressed_vocab_size, *self.engram_num_embeddings) <= 0
            or not 0 <= self.engram_pad_token_id < self.vocab_size
        ):
            raise ValueError("Invalid Engram layout, token mapping or table dimensions")
        if bool(draft) != bool(self.dspark_block_size) or draft and (
            self.dspark_block_size < 2 or not self.dspark_target_layer_ids or self.dspark_markov_rank < 1
            or not 0 <= self.dspark_noise_token_id < self.vocab_size
            or not 0 < self.expert_counts(n)[1] <= self.expert_counts(n)[0]
        ):
            raise ValueError("Invalid DSpark stages, block, target features, or draft expert configuration")

    def expert_counts(self, layer):
        if layer < self.num_hidden_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_num_experts_per_tok or self.num_experts_per_tok,
        )


def cache_owner(sources, layer):
    return max((i for i in sources if i <= layer), default=None)


def is_vision_tensor(name):
    return name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".gate.bias_vl")


def block_shapes(config, layer_id):
    c = config
    d, h, hd, qr = c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank
    hc, mid = c.hc_mult, c.moe_intermediate_size
    mix = (hc + 2) * hc
    shapes = {
        "attn.attn_sink": (h,),
        "attn.wq_a.weight": (qr, d),
        "attn.q_norm.weight": (qr,),
        "attn.wq_b.weight": (h * hd, qr),
        "attn.wkv.weight": (hd, d),
        "attn.kv_norm.weight": (hd,),
        "attn.wo_a.weight": (c.o_groups * c.o_lora_rank, h * hd // c.o_groups),
        "attn.wo_b.weight": (d, c.o_groups * c.o_lora_rank),
        "attn_norm.weight": (d,),
        "ffn_norm.weight": (d,),
    }
    for branch in ("attn", "ffn"):
        shapes.update({
            f"hc_{branch}_fn": (mix, hc * d), f"hc_{branch}_base": (mix,), f"hc_{branch}_scale": (3,),
        })
    if layer_id in c.kv_source_layer_ids:
        shapes.update({"attn.compressor.wkv.weight": (hd, d), "attn.compressor.norm.weight": (hd,)})
        if c.compress_ratios[layer_id] > 1:
            shapes["attn.compressor.wgate.weight"] = (hd, d)
    if layer_id in c.index_source_layer_ids:
        ih, ihd = c.index_n_heads, c.index_head_dim
        shapes.update({"attn.indexer.wq_b.weight": (ih * ihd, qr), "attn.indexer.weights_proj.weight": (ih, d)})
        if layer_id in c.kv_source_layer_ids:
            shapes.update({"attn.indexer.wk.weight": (ihd, hd), "attn.indexer.k_norm.weight": (ihd,)})
    if layer_id in c.engram_layer_ids:
        rows = c.engram_num_embeddings[c.engram_layer_ids.index(layer_id)]
        cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim
        shapes.update({
            "engram.embed.weight": (rows, c.engram_head_dim), "engram.wkv.weight": ((hc + 1) * d, cols),
            "engram.q_weight": (hc, d), "engram.k_weight": (hc, d),
        })
    experts, _ = c.expert_counts(layer_id)
    shapes.update({"ffn.gate.weight": (experts, d), "ffn.gate.bias": (experts,)})
    for expert in ["shared_experts", *(f"experts.{i}" for i in range(experts))]:
        for matrix, shape in (("w1", (mid, d)), ("w2", (d, mid)), ("w3", (mid, d))):
            shapes[f"ffn.{expert}.{matrix}.weight"] = shape
    prefix = f"layers.{layer_id}" if layer_id < c.num_hidden_layers else f"mtp.{layer_id - c.num_hidden_layers}"
    return {f"{prefix}.{name}": shape for name, shape in shapes.items()}


def weight_shapes(config):
    shapes = {
        "embed.weight": (config.vocab_size, config.hidden_size),
        "head.weight": (config.vocab_size, config.hidden_size),
        "norm.weight": (config.hidden_size,),
    }
    for layer in range(config.num_hidden_layers):
        shapes.update(block_shapes(config, layer))
    if config.num_nextn_predict_layers:
        shapes.update(draft_shapes(config))
    return shapes


def draft_shapes(config):
    stages = config.num_nextn_predict_layers
    if not stages:
        return {}
    hidden = config.hidden_size
    vocab = config.vocab_size
    rank = config.dspark_markov_rank
    last = f"mtp.{stages - 1}"
    shapes = {
        "mtp.0.main_proj.weight": (hidden, hidden * len(config.dspark_target_layer_ids)),
        "mtp.0.main_norm.weight": (hidden,),
        f"{last}.norm.weight": (hidden,),
        f"{last}.markov_head.embed.weight": (vocab, rank),
        f"{last}.markov_head.head.weight": (vocab, rank),
        f"{last}.confidence_head.proj.weight": (1, hidden + rank),
    }
    for stage in range(stages):
        shapes.update(block_shapes(config, config.num_hidden_layers + stage))
    return shapes

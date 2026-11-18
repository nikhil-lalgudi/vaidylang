"""DeepSeek Engram token normalization, hash layout, and gated lookup.

Derived from the pinned DeepSeek inference/engram.py and inference/model.py.
See deepseek-cpu-reference.md and LICENSE.DeepSeek.
"""

import json
import math

import numpy as np

from deepseek_ops import rms_norm, sigmoid


def compressed_token_map(config, directory):
    if not config.engram_layer_ids:
        return None
    sidecar = directory / "engram_token_map.json"
    if sidecar.exists():
        values = json.loads(sidecar.read_text())
    else:
        tokenizer_path = directory / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise ValueError("Engram requires tokenizer.json or a matching engram_token_map.json")
        values = _map_tokenizer(tokenizer_path, config.vocab_size)
    ids = np.asarray(values)
    if (
        ids.shape != (config.vocab_size,) or ids.dtype.kind not in "iu"
        or not np.array_equal(np.unique(ids), np.arange(config.engram_compressed_vocab_size))
    ):
        raise ValueError("Engram token map must match vocab_size and engram_compressed_vocab_size")
    return ids.astype(np.int64)


def _map_tokenizer(path, vocab_size):
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as error:
        raise ValueError("Install tokenizers or provide engram_token_map.json for Engram") from error

    backend = Tokenizer.from_file(str(path))
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    keys = {}
    ids = []
    for token in range(vocab_size):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
        ids.append(keys.setdefault(key, len(keys)))
    return ids


def _prime(value):
    if value < 2 or value % 2 == 0:
        return value == 2
    return not any(value % divisor == 0 for divisor in range(3, math.isqrt(value) + 1, 2))


def bucket_layout(config):
    """Primes are unique across all n-gram sizes, heads, and layers."""
    seen = set()
    layout = {}
    for layer in config.engram_layer_ids:
        primes = []
        for _ in range(config.engram_max_ngram_size - 1):
            current = config.engram_vocab_size
            for _ in range(config.engram_n_heads):
                while current in seen or not _prime(current):
                    current += 1
                seen.add(current)
                primes.append(current)
        layout[layer] = np.asarray(primes, dtype=np.int64).reshape(-1, config.engram_n_heads)
    return layout


class EngramHash:
    DEAD = -1

    def __init__(self, config, token_map):
        self.token_map = token_map
        self.max_ngram = config.engram_max_ngram_size
        self.pad = int(token_map[config.engram_pad_token_id])
        self.primes = bucket_layout(config)
        self.offsets = {}
        self.multipliers = {}
        bound = max(1, int(np.iinfo(np.int64).max) // config.engram_compressed_vocab_size // 2)
        for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
            primes = self.primes[layer].ravel()
            if int(primes.sum()) != rows:
                raise ValueError(f"layers.{layer}.engram.embed.weight needs {primes.sum()} hash bucket rows, config has {rows}")
            self.offsets[layer] = np.cumsum(np.r_[0, primes[:-1]])
            rng = np.random.default_rng(10007 * layer)
            self.multipliers[layer] = 2 * rng.integers(0, bound, self.max_ngram, dtype=np.int64) + 1

    def hashes(self, layer, mapped_tokens):
        tokens = [self.pad] * self.max_ngram
        for lookback, token in enumerate(reversed(mapped_tokens[-self.max_ngram:])):
            if token == self.DEAD:
                break
            tokens[lookback] = token
        rolling = np.bitwise_xor.accumulate(np.asarray(tokens, dtype=np.int64) * self.multipliers[layer])
        return (rolling[1:, None] % self.primes[layer]).ravel() + self.offsets[layer]


def gated_lookup(x, projected, q_weight, k_weight, eps):
    key, value = projected[:-x.shape[-1]].reshape(x.shape), projected[-x.shape[-1]:]
    dot = np.sum(rms_norm(x, q_weight, eps) * rms_norm(key, k_weight, eps), axis=-1)
    dot /= math.sqrt(x.shape[-1])
    gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot))
    return x + gate[:, None] * value

"""NumPy reference operations derived from the pinned DeepSeek model and kernels.

See deepseek-cpu-reference.md and LICENSE.DeepSeek.
"""

import math

import numpy as np

from engine import rms_norm as rms_norm


def softmax(x, axis=-1):
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def rope_frequencies(config, compressed=False):
    width = config.qk_rope_head_dim
    theta = config.compress_rope_theta if compressed else config.rope_theta
    dims = np.arange(width // 2, dtype=np.float32)
    freqs = theta ** (-2.0 * dims / width)
    if compressed and config.rope_scaling:
        yarn = config.rope_scaling
        original = yarn["original_max_position_embeddings"]
        if original > 0:
            numerator = width / (2 * math.log(theta))
            low = max(0, math.floor(numerator * math.log(original / (2 * math.pi * yarn["beta_fast"]))))
            high = min(width - 1, math.ceil(numerator * math.log(original / (2 * math.pi * yarn["beta_slow"]))))
            ramp = np.clip((dims - low) / max(high - low, 1e-3), 0, 1)
            freqs = freqs * (1 - ramp) + freqs * ramp / yarn["factor"]
    return freqs


def rotary(x, size, angle, inverse=False):
    result = x.copy()
    angle = np.asarray(angle, dtype=x.dtype)
    cos, sin = np.cos(angle), np.sin(-angle if inverse else angle)
    first, second = x[..., -size::2], x[..., -size + 1::2]
    result[..., -size::2] = first * cos - second * sin
    result[..., -size + 1::2] = first * sin + second * cos
    return result


def sparse_attention(q, kv, sink, selected=None):
    if selected is not None:
        selected = np.asarray(selected)
        kv = kv[selected[selected >= 0]]
    scores = (q @ kv.T) / math.sqrt(q.shape[-1])
    # The learned sink is one more logit with a zero value vector.
    weights = softmax(np.concatenate((scores, np.asarray(sink)[..., None]), axis=-1))
    return weights[..., :-1] @ kv


def pool_latents(values, scores):
    return (values * softmax(scores, axis=0)).sum(axis=0)


def top_indices(scores, count):
    # Stable ties make the reduced CPU fixture deterministic.
    indices = np.argsort(-scores, kind="stable")[:count]
    return indices[np.isfinite(scores[indices])]


def candidate_mask(scores, count, width):
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    blocks = np.pad(scores, (0, -len(scores) % width), constant_values=-np.inf).reshape(-1, width)
    peaks = blocks.max(axis=-1)
    peaks[-1] = np.inf
    indices = np.argsort(-peaks, kind="stable")[:count]
    mask = np.zeros(len(peaks), dtype=bool)
    mask[indices] = True
    return np.repeat(mask, width)[:len(scores)]


def index_scores(q, keys, weights):
    dots = np.maximum(q @ keys.T, 0)
    return (dots * weights[:, None]).sum(axis=0) / math.sqrt(q.shape[0] * q.shape[1])


def split_hc(mixes, scale, base, count, iterations, eps):
    pre = sigmoid(mixes[..., :count] * scale[0] + base[:count]) + eps
    post = 2 * sigmoid(mixes[..., count:2 * count] * scale[1] + base[count:2 * count])
    comb = mixes[..., 2 * count:] * scale[2] + base[2 * count:]
    comb = softmax(comb.reshape(*comb.shape[:-1], count, count)) + eps
    comb /= comb.sum(axis=-2, keepdims=True) + eps
    for _ in range(iterations - 1):
        comb /= comb.sum(axis=-1, keepdims=True) + eps
        comb /= comb.sum(axis=-2, keepdims=True) + eps
    return pre, post, comb


def hc_pre(x, pre):
    return np.sum(np.asarray(pre)[..., None] * x, axis=-2)


def hc_post(y, x, post, comb):
    return post[..., None] * y[..., None, :] + comb.swapaxes(-1, -2) @ x


def route(logits, bias, count, score_function, normalize, scale):
    if score_function == "sqrtsoftplus":
        scores = np.sqrt(np.logaddexp(0, logits))
    elif score_function == "sigmoid":
        scores = sigmoid(logits)
    else:
        scores = softmax(logits)
    indices = top_indices(scores + bias, count)
    weights = scores[indices]
    if normalize and count > 1:
        weights /= weights.sum() + 1e-20
    return indices, weights * scale


def swiglu(gate, up, limit):
    if limit > 0:
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
    return gate * sigmoid(gate) * up


def bf16_round(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return np.where(np.isfinite(x), rounded.view(np.float32), x)

"""Small CPU autoregressive drafter for the Qwen3 speculative benchmark."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


WEIGHTS_ENV = "QWEN3_SPECULATIVE_DRAFT_WEIGHTS"


@dataclass(frozen=True)
class DraftWeights:
    vocabulary: np.ndarray
    token_classes: dict[int, int]
    embeddings: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray
    confidence: float
    context: int


@lru_cache(maxsize=4)
def load_weights(path):
    with np.load(path, allow_pickle=False) as artifact:
        vocabulary = artifact["vocabulary"]
        embeddings = artifact["embeddings"]
        hidden_weight = artifact["hidden_weight"]
        hidden_bias = artifact["hidden_bias"]
        output_weight = artifact["output_weight"]
        output_bias = artifact["output_bias"]
        confidence = float(artifact["confidence"].item())

    if (vocabulary.ndim != 1 or not len(vocabulary) or vocabulary.dtype.kind not in "iu"
            or (vocabulary < 0).any() or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("Draft vocabulary must contain unique nonnegative token IDs")
    classes = len(vocabulary) + 1  # Class zero represents other tokens.
    if (embeddings.ndim != 2 or embeddings.shape[0] != classes + 1 or not embeddings.shape[1]
            or hidden_weight.ndim != 2 or not hidden_weight.shape[0]
            or hidden_weight.shape[0] % embeddings.shape[1] or not hidden_weight.shape[1]
            or hidden_bias.shape != (hidden_weight.shape[1],)
            or output_weight.shape != (hidden_weight.shape[1], classes)
            or output_bias.shape != (classes,) or not 0 <= confidence <= 1):
        raise ValueError("Draft weight dimensions or confidence are invalid")
    for array in (embeddings, hidden_weight, hidden_bias, output_weight, output_bias):
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Draft weights must contain finite float32 values")
        array.setflags(write=False)
    vocabulary.setflags(write=False)
    return DraftWeights(
        vocabulary, {int(token): i for i, token in enumerate(vocabulary, start=1)},
        embeddings, hidden_weight, hidden_bias, output_weight, output_bias,
        confidence, hidden_weight.shape[0] // embeddings.shape[1],
    )


class LearnedDraft:
    """Propose shortlisted tokens until the next prediction is uncertain."""

    def __init__(self, path=None):
        path = path or os.environ.get(WEIGHTS_ENV)
        if not path:
            raise ValueError(f"Set {WEIGHTS_ENV} to the trained draft .npz file")
        self.weights = load_weights(Path(path).expanduser().resolve())

    def propose(self, history, limit):
        weights = self.weights
        recent = history[-weights.context:]
        padding = len(weights.vocabulary) + 1
        context = [padding] * (weights.context - len(recent))
        context.extend(weights.token_classes.get(token_id, 0) for token_id in recent)
        proposal = []
        for _ in range(limit):
            features = weights.embeddings[context].ravel()
            hidden = features @ weights.hidden_weight + weights.hidden_bias
            np.maximum(hidden, 0, out=hidden)
            logits = hidden @ weights.output_weight + weights.output_bias
            prediction = int(logits.argmax())
            if prediction == 0 or 1 / np.exp(logits - logits[prediction]).sum() < weights.confidence:
                break
            proposal.append(int(weights.vocabulary[prediction - 1]))
            context.pop(0)
            context.append(prediction)
        return proposal

"""Greedy draft verification for a single request."""

from contextlib import nullcontext
from time import perf_counter_ns


class NgramDraft:
    """Copy the continuation of the latest matching suffix in this request."""

    def __init__(self, max_order=4):
        self.max_order = max_order
        self.following = {}
        self.indexed = 0

    def propose(self, history, limit):
        for end in range(self.indexed, len(history) - 1):
            for order in range(1, min(self.max_order, end + 1) + 1):
                self.following[tuple(history[end + 1 - order:end + 1])] = end + 1
        self.indexed = len(history) - 1
        for order in range(min(self.max_order, len(history)), 0, -1):
            start = self.following.get(tuple(history[-order:]))
            if start is not None:
                return history[start:start + limit]
        return []


def verified_tokens(draft, predictions):
    """Return accepted draft tokens followed by the correction or bonus token."""
    for index, proposed in enumerate(draft):
        if proposed != predictions[index]:
            return [*draft[:index], predictions[index]]
    return [*draft, predictions[len(draft)]]


def speculative_tokens(single, block, ids, new_tokens, draft_factory=NgramDraft, *, cycles=None, annotate=None):
    if new_tokens < 1 or single.state is not block.state:
        raise ValueError("Expected at least one output token and a shared decode cache")
    annotate = annotate or (lambda _: nullcontext())
    draft = draft_factory()
    with annotate("prefill"):
        single.prefill(ids)
        first = int(single.token.item())
    history = list(map(int, ids))
    history.append(first)
    yield first
    remaining = new_tokens - 1
    token_in_scalar_graph = True
    while remaining:
        started = perf_counter_ns()
        capacity = block.state.remaining
        limit = min(block.block_size - 1, remaining - 1) if capacity >= block.block_size else 0
        with annotate("draft"):
            proposal = list(draft.propose(history, limit))[:limit] if limit else []
        drafted = perf_counter_ns()
        if proposal:
            inputs = [history[-1], *proposal]
            inputs.extend([inputs[-1]] * (block.block_size - len(inputs)))
            with annotate("verify"):
                block.stage(inputs)
                try:
                    emitted = verified_tokens(proposal, block.predictions.tolist())
                except BaseException:
                    block.commit(0)
                    raise
                block.commit(len(emitted))
            token_in_scalar_graph = False
        else:
            with annotate("scalar"):
                single.decode(single.token if token_in_scalar_graph else history[-1])
                emitted = [int(single.token.item())]
            token_in_scalar_graph = True
        finished = perf_counter_ns()
        if cycles is not None:
            cycles.append({
                "draft_ms": (drafted - started) / 1e6,
                "verify_ms": (finished - drafted) / 1e6 if proposal else 0.0,
                "scalar_ms": (finished - drafted) / 1e6 if not proposal else 0.0,
                "cycle_ms": (finished - started) / 1e6,
                "proposed": len(proposal),
                "accepted": len(emitted) - 1,
                "emitted": len(emitted),
                "block_used": bool(proposal),
            })
        history.extend(emitted)
        remaining -= len(emitted)
        yield from emitted

"""Greedy speculative verification with a complete CPU request-state transaction."""

from copy import deepcopy
from itertools import islice
from time import perf_counter_ns

from speculative import NgramDraft, verified_tokens


class VerifiedState:
    """Stage cache changes on a private copy and publish only verified inputs."""

    def __init__(self, engine, state):
        self.engine = engine
        self.state = state
        self._pending = None

    def stage(self, inputs, *, valid_count=None, trace=None):
        if self._pending is not None:
            raise ValueError("Commit or discard the staged CPU block before staging another")
        inputs = tuple(inputs)
        count = len(inputs) if valid_count is None else valid_count
        if not 0 < count <= min(len(inputs), self.state.remaining):
            raise ValueError("Staged inputs exceed the block width or remaining cache capacity")
        # Only actual inputs have positions. Masked suffix slots cannot update any cache.
        inputs = inputs[:count]
        staged = deepcopy(self.state)
        logits = self.engine.forward(inputs, staged, trace=trace)
        self._pending = (inputs, staged)
        return logits

    def commit(self, count):
        if self._pending is None:
            raise ValueError("No CPU block is staged")
        inputs, staged = self._pending
        if not 0 <= count <= len(inputs):
            raise ValueError("Commit exceeds the staged inputs")
        self._pending = None
        if count == len(inputs):
            self.state = staged
        elif count:
            # Replay onto another private copy so a failed partial commit leaves the
            # published request intact. Retaining only a KV length would lose overwritten
            # ring slots, compressor accumulators, hash history, and target hidden state.
            prefix = deepcopy(self.state)
            self.engine.forward(inputs[:count], prefix)
            self.state = prefix


def history_only(draft):
    """Adapt a request-local n-gram or learned proposer to the target-state interface."""
    return lambda history, limit, state: draft.propose(history, limit)


def speculative_tokens(
    engine, ids, new_tokens, block_size, propose=None, *, cycles=None, capacity=None, eos_token_id=None
):
    """Generate greedy tokens; block_size counts the previous token plus proposed tokens."""
    if new_tokens < 1 or block_size < 1:
        raise ValueError("Use positive output and verifier block sizes")
    ids = list(ids)
    if capacity is None:
        capacity = min(len(ids) + new_tokens - 1, engine.config.max_position_embeddings)
    logits, state = engine.prefill(ids, capacity)
    transaction = VerifiedState(engine, state)
    propose = history_only(NgramDraft()) if propose is None else propose
    first = int(logits.argmax())
    history = [*ids, first]
    yield first
    remaining = new_tokens - 1
    while remaining and transaction.state.remaining and history[-1] != eos_token_id:
        started = perf_counter_ns()
        limit = min(block_size - 1, remaining - 1, transaction.state.remaining - 1)
        proposal = list(islice(propose(tuple(history), limit, transaction.state), limit)) if limit else []
        drafted = perf_counter_ns()
        accepted = 0
        if proposal:
            inputs = [history[-1], *proposal]
            valid_count = len(inputs)
            inputs.extend([inputs[-1]] * (block_size - valid_count))
            rows = transaction.stage(inputs, valid_count=valid_count)
            try:
                emitted = verified_tokens(proposal, rows.argmax(axis=-1).tolist())
                accepted = len(emitted) - 1
                if eos_token_id in emitted:
                    emitted = emitted[:emitted.index(eos_token_id) + 1]
                    accepted = min(accepted, len(emitted))
            except BaseException:
                transaction.commit(0)
                raise
            transaction.commit(len(emitted))
        else:
            emitted = [int(engine.decode(history[-1], transaction.state).argmax())]
        finished = perf_counter_ns()
        if cycles is not None:
            cycles.append({
                "draft_ms": (drafted - started) / 1e6,
                "verify_ms": (finished - drafted) / 1e6 if proposal else 0.0,
                "scalar_ms": (finished - drafted) / 1e6 if not proposal else 0.0,
                "cycle_ms": (finished - started) / 1e6,
                "proposed": len(proposal),
                "accepted": accepted,
                "emitted": len(emitted),
                "block_used": bool(proposal),
            })
        history.extend(emitted)
        remaining -= len(emitted)
        yield from emitted


def target_tokens(engine, ids, new_tokens, **options):
    return speculative_tokens(engine, ids, new_tokens, 1, **options)

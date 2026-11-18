"""CPU request and complete-state transaction tests."""

from dataclasses import dataclass, field
import unittest

import numpy as np

from speculative_cpu import VerifiedState, speculative_tokens, target_tokens


@dataclass
class State:
    capacity: int
    ids: list = field(default_factory=list)
    ring: dict = field(default_factory=dict)
    compressed: list = field(default_factory=list)
    target_features: list = field(default_factory=list)
    hash_tail: list = field(default_factory=list)

    @property
    def length(self):
        return len(self.ids)

    @property
    def remaining(self):
        return self.capacity - self.length


class CycleEngine:
    """The oracle next token is (last token + 1) mod 7."""

    class config:
        max_position_embeddings = 32

    def prefill(self, ids, capacity):
        state = State(capacity)
        return self.forward(ids, state)[-1], state

    def decode(self, token, state):
        return self.forward([token], state)[0]

    def forward(self, ids, state, trace=None):
        results = []
        for token in ids:
            pos = state.length
            state.ring[pos % 3] = token
            state.ids.append(token)
            state.target_features.append((token, pos))
            state.hash_tail = state.ids[-3:]
            if state.length % 2 == 0:
                state.compressed.append(tuple(state.ids[-2:]))
            if trace is not None:
                trace(pos, "ring", dict(state.ring))
            if token == -1:
                raise ValueError("bad staged token")
            results.append(np.eye(7, dtype=np.float32)[(token + 1) % 7])
        return np.asarray(results)


def perfect(history, limit, _):
    return [(history[-1] + step + 1) % 7 for step in range(limit)]


class SpeculativeCPUControllerTests(unittest.TestCase):
    def setUp(self):
        self.engine = CycleEngine()

    def test_partial_commit_and_full_abort_restore_every_state_field(self):
        engine = self.engine
        _, original = engine.prefill([1, 2, 3], 20)
        transaction = VerifiedState(engine, original)
        rows = transaction.stage([4, 5, 0, 0])
        self.assertEqual(rows.argmax(axis=1).tolist(), [5, 6, 1, 1])
        self.assertIs(transaction.state, original)
        transaction.commit(2)
        _, expected = engine.prefill([1, 2, 3, 4, 5], 20)
        self.assertEqual(transaction.state, expected)
        transaction.stage([6, 6, 6])
        transaction.commit(0)
        self.assertEqual(transaction.state, expected)

    def test_failure_after_cache_write_is_atomic_and_recoverable(self):
        engine = self.engine
        _, original = engine.prefill([1, 2, 3], 20)
        transaction = VerifiedState(engine, original)
        trace_rows = []
        with self.assertRaisesRegex(ValueError, "bad staged token"):
            transaction.stage([4, -1], trace=lambda *row: trace_rows.append(row))
        self.assertEqual(len(trace_rows), 2)
        self.assertIs(transaction.state, original)
        self.assertEqual(original.ids, [1, 2, 3])
        transaction.stage([4, -1, -1], valid_count=1)
        transaction.commit(1)
        self.assertEqual(transaction.state.ids, [1, 2, 3, 4])

    def test_complete_partial_and_zero_acceptance(self):
        engine = self.engine
        for proposer, expected_accepted, expected_emitted in (
            (perfect, 3, 4),
            (lambda h, n, s: [perfect(h, n, s)[0], 2, 2][:n], 1, 2),
            (lambda h, n, s: [0] * n, 0, 1),
        ):
            with self.subTest(expected_accepted=expected_accepted):
                cycles = []
                tokens = list(speculative_tokens(engine, [0], 12, 4, proposer, cycles=cycles))
                self.assertEqual(tokens, list(target_tokens(engine, [0], 12)))
                self.assertEqual(cycles[0]["accepted"], expected_accepted)
                self.assertEqual(cycles[0]["emitted"], expected_emitted)
                self.assertEqual(cycles[0]["proposed"], 3)

    def test_blocks_eos_context_short_budget_and_consecutive_requests(self):
        engine = self.engine
        for width in (2, 4, 8):
            for kwargs in ({}, {"eos_token_id": 1}, {"eos_token_id": 3}, {"capacity": 4}):
                for n in (1, 2, 18):
                    with self.subTest(width=width, n=n, kwargs=kwargs):
                        expected = list(target_tokens(engine, [0], n, **kwargs))
                        first = list(speculative_tokens(engine, [0], n, width, perfect, **kwargs))
                        second = list(speculative_tokens(engine, [0], n, width, perfect, **kwargs))
                        self.assertEqual(first, expected)
                        self.assertEqual(second, expected)
        cycles = []
        tokens = list(speculative_tokens(engine, [0], 18, 8, perfect, eos_token_id=3, cycles=cycles))
        self.assertEqual(tokens, [1, 2, 3])
        self.assertEqual(cycles[0]["accepted"], 2)

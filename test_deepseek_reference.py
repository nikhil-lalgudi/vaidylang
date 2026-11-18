import hashlib
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import numpy as np

from harness import digest


class ScriptedEngine:
    config = SimpleNamespace(vocab_size=6)

    def __init__(self, rows):
        self.rows = iter(np.asarray(rows, dtype=np.float32))
        self.fed = []

    def prefill(self, ids, capacity):
        self.fed.extend(ids)
        return next(self.rows), None

    def decode(self, token, state):
        self.fed.append(token)
        return next(self.rows)


class ReferenceTests(unittest.TestCase):
    tokenizer = SimpleNamespace(decode=str, backend=SimpleNamespace(id_to_token=str))

    def test_replay_advances_reference_after_candidate_eos_and_separates_divergences(self):
        from tools.deepseek_logits import capture_logit_archive, read_logit_capture, replay_case
        from tools.deepseek_prompts import CompiledCase, smoke_case

        rows = [[0, 2, 1, 1, 1, -2], [2, 0, 4, 3, 1, -5], [0, 0, 0, 0, -3, 7]]
        candidate_rows = [[0, 2, 1, 1, 1, -2.5], [2, 0, 1, 3, 1, 5], rows[2]]
        case = CompiledCase('test', 'test', 3, [1, 2], self.tokenizer)
        candidate = ScriptedEngine(candidate_rows)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'logits.npz'
            with capture_logit_archive(path) as archive:
                reference = smoke_case(ScriptedEngine(rows), case, {5}, archive=archive)
            with ZipFile(path) as archive:
                vectors = read_logit_capture(archive, reference, 6)
        self.assertEqual(reference['output_ids'], [1, 2, 5])
        self.assertEqual(reference['logit_steps'][0]['logits_sha256'], hashlib.sha256(vectors[0].tobytes()).hexdigest())
        result = replay_case(candidate, case, {5}, reference, vectors=vectors, atol=0.75, rtol=0)
        self.assertEqual(candidate.fed, [1, 2, 1, 2])
        self.assertEqual(result['candidate_ids_on_reference_prefix'], [1, 5, 5])
        self.assertEqual(result['first_logit_difference'], 0)
        self.assertEqual(result['first_outside_tolerance'], 1)
        self.assertEqual(result['first_token_difference'], 1)
        step = result['logit_steps'][1]
        self.assertEqual(step['prefix_sha256'], digest([1, 2, 1]))
        self.assertEqual((step['reference_token'], step['competing_token']), ('2', '5'))
        self.assertEqual(step['reference_margin'], 9.0)
        self.assertEqual(step['candidate_margin'], -4.0)
        self.assertEqual(step['max_error_token_id'], 5)
        self.assertEqual(step['outside_tolerance'], 2)
        hash_only = replay_case(ScriptedEngine(candidate_rows), case, {5}, reference)
        self.assertIsNone(hash_only['logit_steps'][1]['reference_margin'])
        self.assertEqual(hash_only['first_logit_difference'], 0)

    def test_replay_rejects_changed_prompt_and_trace_prefix(self):
        from tools.deepseek_logits import replay_case
        from tools.deepseek_prompts import CompiledCase, smoke_case

        rows = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        case = CompiledCase('test', 'test', 2, [2, 1], self.tokenizer)
        reference = smoke_case(ScriptedEngine(rows), case, set())
        with self.assertRaisesRegex(ValueError, 'prompt_ids'):
            replay_case(ScriptedEngine(rows), case, set(), dict(reference, prompt_ids=[1, 2]))
        reference['logit_steps'][1]['prefix_sha256'] = digest([2, 1, 5])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            replay_case(ScriptedEngine(rows), case, set(), reference)

    @unittest.skipUnless(find_spec("torch"), "requires PyTorch")
    def test_upstream_adapter_uses_full_prefill_and_one_token_decode(self):
        import torch
        from tools.deepseek_upstream_runner import UpstreamEngine

        class Model:
            max_seq_len = 7

            def __init__(self):
                self.calls = []

            def forward(self, ids, position, **kwargs):
                self.calls.append((ids.tolist(), position, kwargs))
                return torch.tensor([1]), torch.tensor([[0.0, 3.0]]), None

        model = Model()
        engine = UpstreamEngine(model, 2, 'cpu')
        arrays = SimpleNamespace(images=[object()], token_types=[-1, 0])
        first, state = engine.prefill([0, 1], 3, images=arrays)
        second = engine.decode(0, state)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(model.calls[0][:2], ([[0, 1]], 0))
        self.assertEqual(model.calls[0][2]['token_types'].tolist(), [[-1, 0]])
        self.assertEqual(model.calls[0][2]['images'], [arrays.images])
        self.assertEqual(model.calls[1], ([[0]], 2, {}))

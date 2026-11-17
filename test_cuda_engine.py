from importlib.util import find_spec
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from engine import Qwen3Engine, tiny_weights


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class CudaEngineTests(unittest.TestCase):
    def test_cached_decode_matches_cpu_and_preserves_independent_requests(self):
        from cuda_engine import CudaQwen3Engine

        config, weights = tiny_weights()
        reference = Qwen3Engine(config, weights)
        engine = CudaQwen3Engine(config, weights, dtype=torch.float32)

        first, state = engine.prefill([3, 1, 4], capacity=5)
        other, other_state = engine.prefill([1, 5], capacity=3)
        np.testing.assert_allclose(first.cpu().numpy(), reference.prefill([3, 1, 4], 3)[0], atol=2e-5)
        np.testing.assert_allclose(other.cpu().numpy(), reference.prefill([1, 5], 2)[0], atol=2e-5)
        for token, prefix in ((7, [3, 1, 4, 7]), (8, [3, 1, 4, 7, 8])):
            decoded = engine.decode(torch.tensor([token], device="cuda"), state)
            np.testing.assert_allclose(decoded.cpu().numpy(), reference.prefill(prefix, len(prefix))[0], atol=2e-5)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            engine.decode(9, state)
        self.assertEqual(state.position.item(), 5)
        self.assertEqual(other_state.position.item(), 2)
        decoded = engine.decode(6, other_state)
        np.testing.assert_allclose(decoded.cpu().numpy(), reference.prefill([1, 5, 6], 3)[0], atol=2e-5)

    @unittest.skipUnless(find_spec("sgl_kernel"), "requires FA3")
    def test_bf16_cache_initializes_attention_padding(self):
        from unittest.mock import patch
        from cuda_engine import CudaQwen3Engine

        engine = CudaQwen3Engine(*tiny_weights(), dtype=torch.bfloat16)
        empty = torch.empty

        def poisoned_empty(*args, **kwargs):
            return empty(*args, **kwargs).fill_(float("nan"))

        with patch("cuda_engine.torch.empty", side_effect=poisoned_empty):
            state = engine.allocate_cache(132)
        self.assertTrue(torch.isfinite(state.keys).all().item())
        self.assertTrue(torch.isfinite(state.values).all().item())

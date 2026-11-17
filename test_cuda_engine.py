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
    def test_block_commit_and_rollback_match_scalar_cache(self):
        from cuda_engine import BlockDecodeGraph, CudaQwen3Engine, DecodeGraph

        engine = CudaQwen3Engine(*tiny_weights(), dtype=torch.float32, packed_projections=True)
        prompt = [3, 1, 4, 1, 5]
        state = engine.allocate_cache(12)
        single = DecodeGraph(engine, state, prompt)
        block = BlockDecodeGraph(engine, state, prompt, 4)
        single.prefill(prompt)
        _, reference = engine.prefill(prompt, 12)

        candidates = [9, 2, 10, 6]
        logits = block.stage(candidates)
        with self.assertRaisesRegex(ValueError, "staged"):
            single.decode(9)
        for index, token in enumerate(candidates[:2]):
            expected = engine.decode(token, reference)
            torch.testing.assert_close(logits[index], expected, atol=2e-5, rtol=2e-5)
        block.commit(2)
        self.assertEqual(state.position.item(), len(prompt) + 2)
        self.assertEqual(state.remaining, 5)

        block.stage([7, 8, 9, 10])
        block.commit(0)
        expected = engine.decode(3, reference)
        torch.testing.assert_close(single.decode(3), expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(state.position.item(), len(prompt) + 3)
        block.stage([5, 6, 7, 8])
        block.commit(3)
        with self.assertRaisesRegex(ValueError, "capacity"):
            block.stage([9, 9, 9, 9])

    def test_draft_rejection_preserves_greedy_continuation(self):
        from cuda_engine import BlockDecodeGraph, CudaQwen3Engine, DecodeGraph
        from speculative import speculative_tokens

        engine = CudaQwen3Engine(*tiny_weights(), dtype=torch.float32, packed_projections=True)
        prompt = [3, 1, 4, 1, 5]
        new_tokens = 16
        state = engine.allocate_cache(len(prompt) + new_tokens - 1)
        single = DecodeGraph(engine, state, prompt)
        single.prefill(prompt)
        expected = [single.token.item()]
        for _ in range(new_tokens - 1):
            single.decode(single.token)
            expected.append(single.token.item())

        class MixedDraft:
            should_reject = False

            def propose(self, history, limit):
                start = len(history) - len(prompt)
                proposal = expected[start:start + limit]
                self.should_reject = not self.should_reject
                if self.should_reject:
                    proposal[-1] = (proposal[-1] + 1) % engine.config.vocab_size
                return proposal

        block = BlockDecodeGraph(engine, state, prompt, 4)
        cycles = []
        self.assertEqual(
            list(speculative_tokens(single, block, prompt, new_tokens, MixedDraft, cycles=cycles)),
            expected,
        )
        self.assertTrue(any(cycle["accepted"] for cycle in cycles))
        self.assertTrue(any(cycle["accepted"] < cycle["proposed"] for cycle in cycles))
        self.assertEqual(state.position.item(), len(prompt) + new_tokens - 1)

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

    def test_decode_graph_replays_with_new_tokens_and_reset_cache(self):
        from cuda_engine import CudaQwen3Engine, DecodeGraph

        engine = CudaQwen3Engine(*tiny_weights(), dtype=torch.float32)
        graph = DecodeGraph(engine, engine.allocate_cache(7), [3, 1, 4])
        for prompt, continuation in (([3, 1, 4], [7, 8]), ([1, 5], [9, 2])):
            expected, state = engine.prefill(prompt, capacity=7)
            torch.testing.assert_close(graph.prefill(prompt), expected, atol=0, rtol=0)
            for token in continuation:
                expected = engine.decode(token, state)
                torch.testing.assert_close(graph.decode(token), expected, atol=0, rtol=0)
                self.assertEqual(graph.token.item(), expected.argmax().item())
            self.assertEqual(graph.state.position.item(), len(prompt) + len(continuation))

    @unittest.skipUnless(find_spec("triton"), "requires Triton")
    def test_fused_rms_norm_preserves_bf16_rounding(self):
        from cuda_engine import rms_norm
        from cuda_kernels import fused_rms_norm

        generator = torch.Generator(device="cuda").manual_seed(7)
        for width in (128, 1024):
            x = torch.randn((3, 5, width), device="cuda", dtype=torch.bfloat16, generator=generator)
            weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, generator=generator)
            torch.testing.assert_close(
                fused_rms_norm(x, weight, 1e-6), rms_norm(x, weight, 1e-6), atol=0, rtol=0,
            )

    def test_packed_projections_preserve_prefill_and_decode_logits(self):
        from cuda_engine import CudaQwen3Engine

        config, weights = tiny_weights()
        reference = CudaQwen3Engine(config, weights, dtype=torch.float32)
        ids = [3, 1, 4, 1, 5]
        for unpacked_prefill in (False, True):
            with self.subTest(unpacked_prefill=unpacked_prefill):
                packed = CudaQwen3Engine(
                    config, weights, dtype=torch.float32, packed_projections=True,
                    unpacked_prefill=unpacked_prefill,
                )
                expected, expected_state = reference.prefill(ids, capacity=7)
                actual, actual_state = packed.prefill(ids, capacity=7)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                for token in (9, 2):
                    expected = reference.decode(token, expected_state)
                    actual = packed.decode(token, actual_state)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(find_spec("triton"), "requires Triton")
    def test_fused_decode_pointwise_preserves_bf16_rounding(self):
        from cuda_kernels import fused_qk_rms_norm, fused_residual_rms_norm, fused_rms_norm, fused_swiglu

        generator = torch.Generator(device="cuda").manual_seed(711)

        def sample(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        hidden, update, weight = sample(1, 1024), sample(1, 1024), sample(1024)
        residual, norm = fused_residual_rms_norm(hidden, update, weight, 1e-6)
        torch.testing.assert_close(residual, hidden + update, atol=0, rtol=0)
        torch.testing.assert_close(norm, fused_rms_norm(hidden + update, weight, 1e-6), atol=0, rtol=0)

        q, k, qw, kw = sample(1, 16, 128), sample(1, 8, 128), sample(128), sample(128)
        q_out, k_out = fused_qk_rms_norm(q, k, qw, kw, 1e-6)
        torch.testing.assert_close(q_out, fused_rms_norm(q, qw, 1e-6), atol=0, rtol=0)
        torch.testing.assert_close(k_out, fused_rms_norm(k, kw, 1e-6), atol=0, rtol=0)

        gate, up = sample(1, 3073), sample(1, 3073)
        torch.testing.assert_close(fused_swiglu(gate, up), torch.nn.functional.silu(gate) * up, atol=0, rtol=0)

    @unittest.skipUnless(find_spec("triton"), "requires Triton")
    def test_tiled_decode_masks_padding_and_appends_gqa_cache(self):
        from cuda_attention import tiled_decode_attention
        from cuda_engine import rotate, sdpa

        generator = torch.Generator(device="cuda").manual_seed(711)
        device = "cuda"

        def rand(*shape):
            return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

        keys, values = rand(1, 75, 2, 128), rand(1, 75, 2, 128)
        keys[:, 63:].fill_(float("nan"))
        values[:, 63:].fill_(float("nan"))
        angles = rand(75, 64).float()
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()

        for offset in (63, 64):
            with self.subTest(offset=offset):
                q, k, v = rand(1, 4, 128), rand(1, 2, 128), rand(1, 2, 128)
                old_k, old_v = keys[:, :offset].clone(), values[:, :offset].clone()
                position = torch.tensor([offset], device=device, dtype=torch.int32)
                result = tiled_decode_attention(q, keys, values, k, v, position, cos, sin)

                cos_step, sin_step = cos[offset].float(), sin[offset].float()
                rotated_q = rotate(q.float(), cos_step, sin_step).bfloat16()
                rotated_k = rotate(k.float(), cos_step, sin_step).bfloat16()
                expected = sdpa(
                    rotated_q, torch.cat((old_k[0], rotated_k)), torch.cat((old_v[0], v)),
                    causal=False,
                )
                torch.testing.assert_close(result, expected, atol=0.01, rtol=0.01)
                torch.testing.assert_close(keys[:, :offset], old_k)
                torch.testing.assert_close(values[:, :offset], old_v)
                torch.testing.assert_close(keys[0, offset], rotated_k[0])
                torch.testing.assert_close(values[0, offset], v[0], atol=0, rtol=0)
                self.assertTrue(keys[:, offset + 1:].isnan().all().item())
                self.assertTrue(values[:, offset + 1:].isnan().all().item())

    @unittest.skipUnless(find_spec("triton"), "requires Triton")
    def test_tiled_block_attention_matches_scalar_across_tile_boundary(self):
        from cuda_attention import tiled_decode_attention

        generator = torch.Generator(device="cuda").manual_seed(711)

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        keys, values = rand(1, 75, 2, 128), rand(1, 75, 2, 128)
        keys[:, 63:].fill_(float("nan"))
        values[:, 63:].fill_(float("nan"))
        scalar_keys, scalar_values = keys.clone(), values.clone()
        angles = rand(75, 64).float()
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        q, k = rand(4, 4, 128), rand(4, 2, 128)
        v = rand(4, 3, 128)[:, :2]
        position = torch.tensor([63], device="cuda", dtype=torch.int32)
        block = tiled_decode_attention(q, keys, values, k, v, position, cos, sin)
        scalar = torch.cat([
            tiled_decode_attention(
                q[row:row + 1], scalar_keys, scalar_values, k[row:row + 1],
                v[row:row + 1], position + row, cos, sin,
            )
            for row in range(4)
        ])
        torch.testing.assert_close(block, scalar, atol=0, rtol=0)
        torch.testing.assert_close(keys, scalar_keys, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(values, scalar_values, atol=0, rtol=0, equal_nan=True)

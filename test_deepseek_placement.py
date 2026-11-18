from datetime import timedelta
import errno
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest

import numpy as np


def _gloo_rank(rank, store):
    """Spawned independently on each rank. Only the local arrays enter collectives."""
    import torch
    import torch.distributed as dist

    from deepseek_placement import embedding_contribution, partition_tensor

    dist.init_process_group("gloo", init_method=store, world_size=2, rank=rank, timeout=timedelta(seconds=15))
    try:
        metadata = {"world_size": 2, "experts": 8, "draft_experts": 8}
        dense = np.arange(32, dtype=np.float32).reshape(8, 4)
        x = np.array([.25, -.5, 3., .75], dtype=np.float32)
        local = partition_tensor("layers.0.attn.wo_b.weight", dense, **metadata)[rank]
        partial = torch.from_numpy(local @ x[rank * 2:rank * 2 + 2])
        dist.all_reduce(partial)
        torch.testing.assert_close(partial, torch.from_numpy(dense @ x), rtol=0, atol=0)

        local = partition_tensor("head.weight", dense, **metadata)[rank]
        logits = torch.from_numpy(local @ x)
        gathered = [torch.empty_like(logits) for _ in range(2)]
        dist.all_gather(gathered, logits)
        torch.testing.assert_close(torch.cat(gathered), torch.from_numpy(dense @ x), rtol=0, atol=0)

        ids = np.array([3, 4, 7])
        local = partition_tensor("embed.weight", dense, **metadata)[rank]
        embedded = torch.from_numpy(embedding_contribution(ids, local, rank))
        dist.all_reduce(embedded)
        torch.testing.assert_close(embedded, torch.from_numpy(dense[ids]), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DeepSeekPlacementTests(unittest.TestCase):
    def test_rank_slices_ownership_and_engram_padding(self):
        from deepseek_placement import embedding_contribution, partition_tensor

        embedding = np.arange(24).reshape(8, 3)
        engram = np.arange(29, dtype=np.uint8).reshape(29, 1)
        ids = np.array([0, 3, 5, 7])
        for size in (1, 2, 4, 8):
            opts = {"world_size": size, "experts": 16, "draft_experts": 8}
            with self.subTest(world_size=size):
                shards = partition_tensor("embed.weight", embedding, **opts)
                np.testing.assert_array_equal(np.concatenate(shards), embedding)
                np.testing.assert_array_equal(
                    sum(embedding_contribution(ids, shard, rank) for rank, shard in enumerate(shards)),
                    embedding[ids],
                )
                assert not np.shares_memory(shards[0], embedding)
                padded = partition_tensor("layers.1.engram.embed.scale", engram, **opts)
                np.testing.assert_array_equal(np.concatenate(padded)[:29], engram)
                np.testing.assert_array_equal(np.concatenate(padded)[29:], 127)
                owners = partition_tensor("layers.1.ffn.experts.9.w2.scale", embedding, **opts)
                self.assertEqual([r for r, shard in enumerate(owners) if shard is not None], [9 // (16 // size)])
                draft = partition_tensor("mtp.0.ffn.experts.6.w2.weight", embedding, **opts)
                self.assertEqual([r for r, shard in enumerate(draft) if shard is not None], [6 // (8 // size)])
                replicated = partition_tensor("layers.1.attn.compressor.norm.weight", embedding, **opts)
                for copy in replicated:
                    np.testing.assert_array_equal(copy, embedding)

    def test_quantized_row_and_column_linears_match_single_rank(self):
        from deepseek_placement import parallel_linear, partition_quantized, partition_tensor
        from deepseek_quant import QuantizedWeight

        rng = np.random.default_rng(104)
        raw = rng.integers(0, 32, size=(16, 32), dtype=np.uint8) + 35
        scales = rng.integers(125, 129, size=(8, 8), dtype=np.uint8)
        x = rng.normal(size=(3, 32)).astype(np.float32)
        options = dict(block_shape=(2, 4), encoding="fp8", activation_group_size=4)
        full = QuantizedWeight(raw, scales, raw.shape, **options)
        for size in (1, 2, 4, 8):
            opts = {"world_size": size, "experts": 8}
            for stem, axis in (("layers.2.attn.wq_b", 0), ("layers.2.attn.wo_b", 1)):
                with self.subTest(world_size=size, tensor=stem):
                    values = partition_tensor(stem + ".weight", raw, **opts)
                    local_scales = partition_tensor(stem + ".scale", scales, **opts)
                    np.testing.assert_array_equal(np.concatenate(values, axis=axis), raw)
                    np.testing.assert_array_equal(np.concatenate(local_scales, axis=axis), scales)
                    weights = partition_quantized(stem + ".weight", raw, scales, raw.shape, **opts, **options)
                    np.testing.assert_array_equal(np.concatenate([w.weight for w in weights], axis=axis), full.weight)
                    np.testing.assert_allclose(
                        parallel_linear(x, weights, axis=axis), full(x), atol=3e-6, rtol=2e-6,
                    )

    def test_expert_outputs_reduce_before_adding_replicated_shared_expert(self):
        from deepseek_placement import expert_contribution, partition_tensor
        from deepseek_quant import QuantizedWeight

        scales = np.full((2, 1), 127, dtype=np.uint8)
        packed = [0x22, 0x44, 0x66, 0x52, 0x27, 0x38, 0xC2, 0x26]
        experts = {
            i: (QuantizedWeight(
                np.array([[byte], [byte]], dtype=np.uint8), scales, (2, 2),
                block_shape=(1, 2), encoding="fp4", activation_group_size=2,
            ), np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
            for i, byte in enumerate(packed)
        }
        x = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
        selected = np.array([[1, 6], [7, 0]])
        route = np.array([[.25, .75], [.5, .5]], dtype=np.float32)

        # The selected E2M1 pairs are [2,2], [1,-2], [4,1], [1,1].
        # After the one-sided gate and two-sided up clamp at 4, the gates
        # are [4,-3] for row 0 and [4,3] for row 1.
        gates = np.array([[4., -3.], [4., 3.]], dtype=np.float64)
        mixed = (gates / (1 + np.exp(-gates)) * route).sum(axis=-1)
        shared = x / (1 + np.exp(-x)) * x
        expected = shared + mixed[:, None] * x
        for size in (1, 2, 4, 8):
            with self.subTest(world_size=size):
                local_experts = [{} for _ in range(size)]
                for index, parts in experts.items():
                    rank = index // (8 // size)
                    local_experts[rank][index] = parts
                    placements = partition_tensor(
                        f"layers.0.ffn.experts.{index}.w1.weight", parts[0].weight, size, experts=8,
                    )
                    self.assertEqual([i for i, p in enumerate(placements) if p is not None], [rank])
                contributions = [expert_contribution(x, selected, route, local, 4.) for local in local_experts]
                np.testing.assert_allclose(sum(contributions) + shared, expected, atol=2e-6, rtol=2e-6)

    def test_engram_rows_and_index_head_scores_reduce(self):
        from deepseek_placement import embedding_contribution, index_score_contribution, partition_tensor
        from deepseek_quant import dequantize

        weight = np.full((9, 2), 0x38, dtype=np.uint8)
        scale = (127 + np.arange(9, dtype=np.uint8)).reshape(9, 1)
        ids = np.array([0, 4, 8])
        keys = np.array([[1., 0.], [-1., 1.]], dtype=np.float32)
        query = np.tile(np.array([[1., -3.], [2., 7.]], dtype=np.float32), (4, 1))
        weights = np.tile(np.array([.25, -.5], dtype=np.float32), 4)
        # Four pairs of heads, contributing [.25 - 1, 0 - 2.5] each.
        for size in (1, 2, 4, 8):
            with self.subTest(world_size=size):
                opts = dict(world_size=size, experts=8)
                w = partition_tensor("layers.1.engram.embed.weight", weight, **opts)
                s = partition_tensor("layers.1.engram.embed.scale", scale, **opts)
                rows = [dequantize(wi, si, wi.shape, block_shape=(1, 2), encoding="fp8") for wi, si in zip(w, s)]
                values = sum(embedding_contribution(ids, table, rank) for rank, table in enumerate(rows))
                np.testing.assert_array_equal(values, [[1, 1], [16, 16], [256, 256]])
                q = np.split(query, size)
                w = np.split(weights, size)
                scores = sum(index_score_contribution(qi, wi, keys, total_heads=8) for qi, wi in zip(q, w))
                np.testing.assert_array_equal(scores, [-.75, -2.5])

    def test_invalid_world_size_expert_and_partition_dimension(self):
        from deepseek_placement import partition_quantized, partition_tensor

        matrix = np.zeros((8, 3))
        with self.assertRaisesRegex(ValueError, "world size"):
            partition_tensor("embed.weight", matrix, 0, experts=8)
        with self.assertRaisesRegex(ValueError, "experts.*world size"):
            partition_tensor("embed.weight", matrix, 2, experts=7)
        with self.assertRaisesRegex(ValueError, "expert 8.*8 experts"):
            partition_tensor("layers.1.ffn.experts.8.w1.weight", matrix, 2, experts=8)
        with self.assertRaisesRegex(ValueError, "wo_b.*axis 1.*3.*world size 2"):
            partition_tensor("layers.0.attn.wo_b.weight", matrix, 2, experts=8)
        with self.assertRaisesRegex(ValueError, "wo_b.*rank boundary.*scale block"):
            partition_quantized(
                "layers.0.attn.wo_b.weight", np.zeros((2, 6), dtype=np.uint8),
                np.full((1, 2), 127, dtype=np.uint8), (2, 6), 2, experts=8,
                block_shape=(2, 4), encoding="fp8", activation_group_size=4,
            )

    def test_metadata_validates_index_and_accounts_for_placement(self):
        from checkpoint import Checkpoint
        from deepseek_placement import placement_report

        shapes = {
            "embed.weight": ("BF16", (8, 2)),
            "head.weight": ("BF16", (8, 2)),
            "vision.stem.weight": ("BF16", (2, 2)),
            "layers.0.engram.embed.weight": ("F8_E4M3", (3, 2)),
            "layers.0.engram.embed.scale": ("F8_E8M0", (3, 1)),
            "layers.0.ffn.experts.0.w1.weight": ("I8", (2, 1)),
            "layers.0.ffn.experts.1.w1.weight": ("I8", (2, 1)),
        }
        tensor_sizes = {"BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
        offset = 0
        header = {}
        for name, (dtype, shape) in shapes.items():
            size = tensor_sizes[dtype] * int(np.prod(shape))
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
        config = {
            "model_type": "deepseek_v41",
            "text_config": {
                "model_type": "deepseek_v41_text", "vocab_size": 8, "hidden_size": 8,
                "max_position_embeddings": 32, "num_hidden_layers": 2, "sliding_window": 4,
                "head_dim": 2, "num_attention_heads": 2, "o_groups": 2, "o_lora_rank": 2,
                "compress_ratios": [2, 2], "kv_source_layer_ids": [0], "index_source_layer_ids": [0],
                "index_n_heads": 2, "index_head_dim": 2, "index_topk": 1, "n_routed_experts": 2,
                "hc_mult": 2, "engram_max_ngram_size": 2, "engram_n_heads": 1,
                "engram_layer_ids": [0], "engram_head_dim": 2, "engram_compressed_vocab_size": 8,
            },
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(config))
            shard = "x.safetensors"
            index = {"metadata": {"total_size": offset}, "weight_map": dict.fromkeys(shapes, shard)}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            checkpoint = Checkpoint.open(root, metadata_only=True)
            report = placement_report(checkpoint, {shard: header}, 2, capacity=16)
            self.assertEqual(report["source_tensor_bytes"], offset)
            self.assertEqual(report["source_tensor_count"], len(shapes))
            self.assertEqual(report["ranks"][0]["text_runtime_weight_bytes"], 54)
            self.assertEqual(report["ranks"][0]["text_runtime_scale_bytes"], 2)
            self.assertEqual(report["ranks"][0]["vision_bytes"], 8)
            self.assertEqual(report["per_rank_state_bytes"]["compressed_kv"], 32)
            self.assertEqual(report["per_rank_state_bytes"]["index_key"], 32)
            bad_header = dict(header)
            bad_header.pop("head.weight")
            with self.assertRaisesRegex(ValueError, "missing.*head.weight"):
                placement_report(checkpoint, {shard: bad_header}, 2, capacity=16)

    def test_two_process_gloo_collectives(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires PyTorch with Gloo")
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            self.skipTest("this PyTorch build has no Gloo backend")
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as error:
            if error.errno in (errno.EPERM, errno.EACCES):
                self.skipTest("sandbox denies the local loopback socket required by Gloo")
            raise
        with TemporaryDirectory() as temporary:
            store = (Path(temporary) / "store").as_uri()
            torch.multiprocessing.spawn(_gloo_rank, args=(store,), nprocs=2)


if __name__ == "__main__":
    unittest.main()

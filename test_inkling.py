import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from checkpoint import Checkpoint
from engine import load_engine
from harness import main
from native_cpu import load_library
from test_engine import write_checkpoint_weights


def inkling_fixture(directory):
    """Reduced original Inkling, with the published raw tensor names and interleaving."""
    prefix = "model.llm."
    config = {
        "model_type": "inkling_mm_model", "eos_token_id": 2,
        "text_config": {
            "model_max_length": 64, "vocab_size": 32, "unpadded_vocab_size": 29,
            "hidden_size": 8, "num_hidden_layers": 3, "rms_norm_eps": 1e-6,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 4,
            "d_rel": 2, "rel_extent": 5, "sliding_window_size": 3, "local_layer_ids": [0, 2],
            "log_scaling_n_floor": 3, "log_scaling_alpha": 0.3, "sconv_kernel_size": 4,
            "use_embed_norm": True, "use_sconv": True, "logits_mup_width_multiplier": 3.0,
            "dense_mlp_idx": 1, "dense_intermediate_size": 12, "intermediate_size": 6,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 2,
            "shared_expert_sink": True, "route_scale": 2.0, "use_gate_bias": True,
            "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True,
        },
    }
    shapes = {"embed.weight": (32, 8), "unembed.weight": (32, 8),
              "embed_norm.weight": (8,), "norm.weight": (8,)}
    for layer in range(3):
        heads, grouped, extent = (4, 2, 5) if layer == 1 else (2, 1, 3)
        per_layer = {
            "attn_norm.weight": (8,), "mlp_norm.weight": (8,),
            "attn_sconv.weight": (8, 1, 4), "mlp_sconv.weight": (8, 1, 4),
            "attn.wq_du.weight": (heads * 4, 8), "attn.wo_ud.weight": (8, heads * 4),
            "attn.wk_dv.weight": (grouped * 4, 8), "attn.wv_dv.weight": (grouped * 4, 8),
            "attn.wr_du.weight": (heads * 2, 8), "attn.rel_logits_proj.proj": (2, extent),
            "attn.q_norm.weight": (4,), "attn.k_norm.weight": (4,),
            "attn.k_sconv.weight": (grouped * 4, 1, 4), "attn.v_sconv.weight": (grouped * 4, 1, 4),
        }
        if layer == 0:
            per_layer.update({"mlp.w13_dn.weight": (24, 8), "mlp.w2_md.weight": (8, 12),
                              "mlp.global_scale": (1,)})
        else:
            per_layer.update({
                "mlp.gate.weight": (6, 8), "mlp.gate.bias": (4,), "mlp.gate.global_scale": (1,),
                "mlp.experts.w13_weight": (4, 12, 8), "mlp.experts.w2_weight": (4, 8, 6),
                "mlp.shared_experts.shared_w13_weight": (2, 12, 8),
                "mlp.shared_experts.shared_w2_weight": (2, 8, 6),
            })
        shapes.update({f"layers.{layer}.{name}": shape for name, shape in per_layer.items()})

    rng = np.random.default_rng(98)
    tensors = {}
    for name, shape in shapes.items():
        values = (0.27 * rng.standard_normal(shape)).astype("<f4")
        if "norm.weight" in name or name.endswith("global_scale"):
            values += 1
        if name.endswith(("gate.bias", "gate.global_scale")):
            tensors[prefix + name] = ("F32", values)
        else:
            tensors[prefix + name] = ("BF16", (values.view("<u4") >> 16).astype("<u2"))
    (directory / "config.json").write_text(json.dumps(config))
    shards = [{}, {}]
    weight_map = {}
    for index, (name, tensor) in enumerate(tensors.items()):
        shard = index % 2
        shards[shard][name] = tensor
        weight_map[name] = f"part-{shard}.safetensors"
    for index, shard in enumerate(shards):
        write_checkpoint_weights(directory / f"part-{index}.safetensors", shard)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@unittest.skipUnless(hasattr(load_library(), "vaidy_inkling_load"), "requires a C++ build with Inkling support")
class InklingTests(unittest.TestCase):
    def test_cached_decode_matches_recomputed_prefix_after_sliding_window_wrap(self):
        with TemporaryDirectory() as location:
            directory = Path(location)
            inkling_fixture(directory)
            engine = load_engine(directory, cpu_backend="native")
            history = [2, 11, 3, 7, 9]
            logits, state = engine.prefill(history, capacity=9)
            self.assertEqual(logits.shape, (29,))
            self.assertTrue(np.isfinite(logits).all())
            for token in [1, 5, 9, 6]:
                history.append(token)
                logits = engine.decode(token, state)
                expected, _ = engine.prefill(history, capacity=len(history))
                np.testing.assert_allclose(logits, expected, atol=2e-6, rtol=2e-6)
            # Full-prefix FP32 Transformers eager oracle, b62646bbda836d467060b7e3d64028202079b77d.
            np.testing.assert_allclose(logits, [
                0.15294814, -0.10066217, -0.28588992, -0.46399283, 0.31535983,
                -0.26894027, 0.24980283, -0.26932544, -0.48106021, 0.46249509,
                0.33425725, -0.48469663, -0.04622816, 0.42694107, -0.19346231,
                -0.55839944, -0.06944205, 0.37832373, -0.09181370, -0.28546792,
                0.12328880, 0.05226176, 0.07525501, 0.29686809, -0.25609428,
                0.08555719, 0.04115857, -0.23932922, -0.19718830,
            ], atol=2e-6, rtol=2e-6)
            with self.assertRaisesRegex(ValueError, "capacity"):
                engine.decode(1, state)
            self.assertEqual(state.length, 9)

    def test_harness_generates_from_inkling_shards(self):
        with TemporaryDirectory() as location, contextlib.redirect_stdout(io.StringIO()) as stdout:
            directory = Path(location)
            inkling_fixture(directory)
            result = main(["generate", "--checkpoint", location, "--ids", "2", "11", "3", "7",
                           "--new-tokens", "4"])
            self.assertEqual(result, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["model_type"], "inkling_mm_model")
            self.assertEqual(report["cpu_backend"], "native")
            self.assertEqual(len(report["output_ids"]), 4)
            self.assertTrue(all(0 <= token < 29 for token in report["output_ids"]))
            self.assertEqual(Checkpoint.open(directory).describe()["max_position_embeddings"], 64)

    def test_invalid_token_and_foreign_request_leave_the_cache_usable(self):
        with TemporaryDirectory() as location:
            directory = Path(location)
            inkling_fixture(directory)
            engine = load_engine(directory, cpu_backend="native")
            _, state = engine.prefill([1, 3], capacity=4)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                engine.decode(32, state)
            other = load_engine(directory, cpu_backend="native")
            with self.assertRaisesRegex(ValueError, "different model"):
                other.decode(7, state)
            self.assertEqual(state.length, 2)
            expected, _ = engine.prefill([1, 3, 7], capacity=4)
            np.testing.assert_allclose(engine.decode(7, state), expected, atol=1e-6)

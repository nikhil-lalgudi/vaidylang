from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from checkpoint import Checkpoint
from harness import main
from test_engine import write_checkpoint_weights


DEEPSEEK_CONFIG = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "text_config": {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "max_position_embeddings": 1048576,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4", "scale_fmt": "ue8m0"},
}
INDEX = "model.safetensors.index.json"
SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"
EXPERT = "layers.1.ffn.experts.0.w1.weight"
SCALE = "layers.1.ffn.experts.0.w1.scale"


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        (self.path / "config.json").write_text(json.dumps(DEEPSEEK_CONFIG))
        self.index({EXPERT: SHARD_1, SCALE: SHARD_2})

    def index(self, weight_map):
        (self.path / INDEX).write_text(json.dumps({"metadata": {"total_size": 4}, "weight_map": weight_map}))

    def test_deepseek_public_layout_resolves_nested_config_and_shards(self):
        checkpoint = Checkpoint.open(self.path, metadata_only=True)
        self.assertEqual(checkpoint.model_type, "deepseek_v41")
        self.assertEqual(checkpoint.text_config["vocab_size"], 129280)
        self.assertEqual(checkpoint.config["quantization_config"]["expert_dtype"], "fp4")
        self.assertEqual(checkpoint.shard_for(EXPERT), self.path / SHARD_1)
        self.assertEqual(checkpoint.shard_for(SCALE), self.path / SHARD_2)
        self.assertEqual(checkpoint.missing_shards, (SHARD_1, SHARD_2))

    def test_default_open_reports_missing_checkpoint_shards(self):
        with self.assertRaisesRegex(FileNotFoundError, SHARD_1):
            Checkpoint.open(self.path)

    def test_index_rejects_a_shard_outside_checkpoint_directory(self):
        self.index({EXPERT: "../model.safetensors"})
        with self.assertRaisesRegex(ValueError, "Invalid shard filename"):
            Checkpoint.open(self.path, metadata_only=True)

    def test_cli_inspects_the_deepseek_index_without_loading_shards(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            status = main(["inspect-checkpoint", "--checkpoint", str(self.path), "--metadata-only"])
        self.assertEqual(status, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["model_type"], "deepseek_v41")
        self.assertEqual(report["shard_count"], 2)
        self.assertEqual(report["indexed_tensor_count"], 2)
        self.assertEqual(report["missing_shards"], [SHARD_1, SHARD_2])

    @unittest.skipUnless(find_spec("torch") and find_spec("safetensors"), "requires Torch and Safetensors")
    def test_native_quantized_storage_loads_from_both_deepseek_shards(self):
        import torch

        if not hasattr(torch, "float8_e8m0fnu"):
            self.skipTest("requires a PyTorch build with native float8_e8m0fnu")
        write_checkpoint_weights(self.path / SHARD_1, {EXPERT: ("I8", np.array([-113, 81], dtype="i1"))})
        write_checkpoint_weights(self.path / SHARD_2, {SCALE: ("F8_E8M0", np.array([127, 128], dtype="u1"))})
        checkpoint = Checkpoint.open(self.path)
        self.assertEqual(checkpoint.read_tensor(EXPERT).tolist(), [-113, 81])
        scales = checkpoint.read_tensor(SCALE)
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(scales.view(torch.uint8).tolist(), [127, 128])

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from checkpoint import Checkpoint
from deepseek_fixture import write_fixture
from prompt_input import PromptTokenizer, deepseek_prompt

try:
    from PIL import Image
    import tokenizers
    import torch
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "requires torch, tokenizers, and Pillow")
class DeepSeekVisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.red, self.blue = self.root / "red.png", self.root / "blue.png"
        Image.new("RGB", (17, 11), (255, 0, 0)).save(self.red)
        Image.new("RGB", (17, 11), (0, 0, 255)).save(self.blue)
        self.checkpoint = write_fixture(self.root / "checkpoint", vision=True)

    def messages_for(self, *names):
        return [{"role": "user", "content": [
            {"type": "text", "text": "Before"},
            *[{"type": "image_url", "image_url": {"url": name}} for name in names],
            {"type": "text", "text": "After"},
        ]}]

    def encode(self, *names):
        from deepseek_vision import prepare_image_prompt
        checkpoint = Checkpoint.open(self.checkpoint, metadata_only=True)
        tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
        prompt, images = deepseek_prompt(self.messages_for(*names))
        ids = tokenizer.encode(prompt)
        return prepare_image_prompt(checkpoint, tokenizer, ids, images, self.root)

    def test_image_spans_track_order_roles_and_true_pixel_bytes(self):
        red_first = self.encode("red.png", "blue.png")
        blue_first = self.encode("blue.png", "red.png")
        self.assertEqual(red_first.ids, blue_first.ids)
        self.assertEqual(red_first.token_types, blue_first.token_types)
        self.assertEqual(len(red_first.images), 2)
        for i, image in enumerate(red_first.images):
            self.assertEqual(image.types.tolist(), [0, 1, 1, 2, 3])
            self.assertEqual((image.n_vit_h, image.n_vit_w), (2, 3))
            self.assertEqual(tuple(image.patches.shape), (6, 3, 8, 8))
            stop = image.start + image.types.numel()
            self.assertEqual(red_first.token_types[image.start:stop], image.types.tolist())
            self.assertEqual(red_first.ids[image.start:stop], [94] * image.types.numel())
            self.assertTrue(torch.equal(image.patches, blue_first.images[1 - i].patches))
        self.assertFalse(torch.equal(red_first.images[0].patches, red_first.images[1].patches))

    def test_tall_and_wide_images_keep_a_nonempty_resized_axis(self):
        for size in ((1, 1000), (1000, 1)):
            with self.subTest(size=size):
                path = self.root / "extreme.png"
                Image.new("RGB", size, (255, 0, 0)).save(path)
                image, = self.encode(path.name).images
                self.assertLessEqual(image.types.numel(), 12)
                self.assertGreaterEqual(min(image.n_vit_h, image.n_vit_w), 1)
                self.assertTrue(torch.any(image.patches[:, 0] == 1))

    def test_images_change_the_reduced_prefill_and_engram_history(self):
        from engine import load_engine
        engine = load_engine(self.checkpoint, with_vision=True)
        red = self.encode("red.png")
        blue = self.encode("blue.png")
        actual, state = engine.prefill(red.ids, 64, images=red.images)
        swapped, swapped_state = engine.prefill(blue.ids, 64, images=blue.images)
        np.testing.assert_array_equal(state.mapped_tokens, swapped_state.mapped_tokens)
        image = red.images[0]
        span = slice(image.start, image.start + image.types.numel())
        self.assertEqual(state.mapped_tokens[span], [-1] * image.types.numel())
        self.assertGreater(float(np.max(np.abs(actual - swapped))), 1e-4)
        a = engine.decode(5, state)
        b = engine.decode(5, swapped_state)
        self.assertGreater(float(np.max(np.abs(a - b))), 1e-4)

    def test_cli_generates_from_image_and_processes_metadata_without_weights(self):
        root = Path(__file__).resolve().parent
        message_file = self.root / "messages.json"
        message_file.write_text(json.dumps(self.messages_for("red.png")), encoding="utf-8")
        args = ["--checkpoint", str(self.checkpoint), "--messages", str(message_file), "--new-tokens", "3"]
        def invoke(command):
            return subprocess.run([sys.executable, str(root / "harness.py"), command, *args],
                                  encoding="utf-8", capture_output=True, cwd="/")
        result = invoke("generate")
        self.assertEqual(result.returncode, 0, result.stderr)
        generation = json.loads(result.stdout)
        self.assertTrue(generation["output_ids"])
        self.assertEqual(len(generation["images"]), 1)
        for path in self.checkpoint.glob("*.safetensors"):
            path.unlink()
        result = invoke("prepare-prompt")
        self.assertEqual(result.returncode, 0, result.stderr)
        prepared = json.loads(result.stdout)
        for key in ("prompt_ids", "prompt_sha256", "token_types", "images"):
            self.assertEqual(prepared[key], generation[key])
        self.assertNotIn("output_ids", prepared)


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from checkpoint import Checkpoint
from prompt_input import PromptTokenizer, deepseek_prompt
from deepseek_fixture import write_fixture, write_fixture_tokenizer
from harness import digest, generate, parse_args

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


@unittest.skipIf(Tokenizer is None, "requires tokenizers")
class PromptInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = write_fixture(Path(self.directory.name))
        self.backend = write_fixture_tokenizer(self.checkpoint / "tokenizer.json")

    def invoke(self, *options):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("harness.py")), "generate",
             "--checkpoint", str(self.checkpoint), *options],
            encoding="utf-8", capture_output=True,
        )

    def generate(self, *options):
        completed = self.invoke(*options)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_text_runs_through_engine_and_replays_by_id(self):
        result = self.generate("--text", "Hello, world!", "--new-tokens", "4")
        replay = self.generate("--ids", "0", "4", "9", "10", "5", "6", "--new-tokens", "4")
        self.assertEqual(result["prompt_ids"], [0, 4, 9, 10, 5, 6])
        for key in ("prompt_ids", "prompt_sha256", "output_ids", "output_sha256"):
            self.assertEqual(result[key], replay[key])
        self.assertEqual(result["output_text"], self.backend.decode(result["output_ids"]))
        self.assertEqual(result["output_sha256"], digest(result["output_ids"]))
        self.assertNotIn("output_text", replay)

    def test_messages_file_preserves_assistant_history(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Again?"},
        ]
        path = self.checkpoint / "messages.json"
        path.write_text(json.dumps(messages), encoding="utf-8")
        result = self.generate("--messages", str(path), "--new-tokens", "1")
        self.assertEqual(result["prompt_ids"], [0, 8, 13, 14, 4, 9, 10, 5, 6, 11, 1, 4, 12, 5, 6])
        self.assertEqual(len(result["output_ids"]), 1)

    def test_mismatched_tokenizer_is_rejected(self):
        write_fixture_tokenizer(self.checkpoint / "tokenizer.json", size=95)
        completed = self.invoke("--text", "Hello, world!")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("vocabulary", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_invalid_messages_are_reported_before_generation(self):
        path = self.checkpoint / "messages.json"
        path.write_text('[{"role":"user","content":{"text":"hello"}}]', encoding="utf-8")
        completed = self.invoke("--messages", str(path))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("content", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_text_stops_at_eos_and_keeps_the_replay_id(self):
        logits = np.zeros(96)
        logits[1] = 1
        engine = SimpleNamespace(prefill=lambda ids, capacity: (logits, None))
        args = parse_args(["generate", "--checkpoint", str(self.checkpoint), "--text", "Hello, world!"])
        with patch("harness.load_engine", return_value=engine):
            result = generate(args)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertEqual(result["output_ids"], [1])
        self.assertEqual(result["output_text"], "")

    def test_file_retains_utf8_and_whitespace(self):
        path = self.checkpoint / "prompt.txt"
        path.write_text("Hello,\n world! 世界", encoding="utf-8")
        inline = self.generate("--text", path.read_text(encoding="utf-8"), "--new-tokens", "1")
        self.assertEqual(inline, self.generate("--text-file", str(path), "--new-tokens", "1"))

    def test_reasoning_effort_alias_matches_the_published_numeric_value(self):
        options = ("--text", "Hello, world!", "--new-tokens", "1", "--thinking", "--reasoning-effort")
        self.assertEqual(self.generate(*options, "max"), self.generate(*options, "100"))

    def test_raw_ids_need_no_tokenizer(self):
        (self.checkpoint / "tokenizer.json").unlink()
        result = self.generate("--ids", "2", "11", "3", "7", "9")
        self.assertEqual(result["output_ids"], [53, 11, 82, 71, 12, 2, 55, 6, 39, 55, 35, 14, 75, 50, 84, 50])


@unittest.skipUnless(os.getenv("VAIDYLANG_DEEPSEEK_METADATA"), "requires pinned DeepSeek config and tokenizer")
class PublishedPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = Checkpoint.open(os.environ["VAIDYLANG_DEEPSEEK_METADATA"], metadata_only=True)
        cls.tokenizer = PromptTokenizer.from_checkpoint(checkpoint)

    def test_pinned_tokenizer_and_chat_ids(self):
        self.assertEqual(self.tokenizer.sha256, "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b")
        cases = [
            ([{"role": "user", "content": "Hello, 世界!"}], {},
             [0, 128803, 19923, 14, 223, 3427, 3, 128804, 128822]),
            ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a color."},
              {"role": "assistant", "content": "Blue."}, {"role": "user", "content": "Dime otro."}], {},
             [0, 128799, 7153, 9708, 16, 128803, 3240, 260, 3605, 16, 128804, 128822,
              29689, 16, 1, 128803, 38, 723, 40012, 16, 128804, 128822]),
            ([{"role": "user", "content": "2+2?"}], {"thinking": True, "reasoning_effort": 50},
             [0, 128799, 77666, 288, 9504, 482, 28, 223, 1328, 343, 16801, 223, 19, 15, 1457,
              14, 270, 4077, 270, 1990, 14, 270, 850, 15146, 270, 22805, 868, 128803, 20, 13,
              20, 33, 128804, 128821]),
        ]
        for messages, options, expected in cases:
            with self.subTest(messages=messages):
                prompt, images = deepseek_prompt(messages, **options)
                self.assertFalse(images)
                self.assertEqual(self.tokenizer.encode(prompt), expected)


if __name__ == "__main__":
    unittest.main()

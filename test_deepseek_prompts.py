from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from checkpoint import Checkpoint
from deepseek_fixture import write_fixture, write_fixture_tokenizer
from harness import digest
from prompt_input import PromptTokenizer

try:
    import tokenizers
except ImportError:
    tokenizers = None


class CorpusTests(unittest.TestCase):
    def test_corpus_covers_100_distinct_prompts(self):
        from tools.deepseek_prompts import DEFAULT_SUITE, read_suite

        rows = read_suite(DEFAULT_SUITE.read_bytes())
        self.assertEqual(len(rows), 100)
        self.assertEqual(set(Counter(r["category"] for r in rows).values()), {10})
        docs = [row["document"] for row in rows if "document" in row]
        self.assertEqual(len(docs), 10)
        self.assertGreater(max(spec["target_tokens"] for spec in docs), 1_048_000)
        images = DEFAULT_SUITE.parent / "deepseek-images"
        self.assertEqual(len(list(images.glob("*.png"))), 10)


@unittest.skipIf(tokenizers is None, "requires tokenizers")
class WorkloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_fixture(self.root)
        write_fixture_tokenizer(self.root / "tokenizer.json")
        self.checkpoint = Checkpoint.open(self.root, metadata_only=True)
        self.tokenizer = PromptTokenizer.from_checkpoint(self.checkpoint)

    def test_generated_document_fills_budget_leaves_room_and_preserves_needle(self):
        from tools.deepseek_prompts import compile_case

        row = {
            "id": "long", "category": "long", "new_tokens": 8,
            "document": {"heading": "Operations log", "fact": "Access word: Blue.",
                         "filler": "Hello, world!", "question": "Access word?", "target_tokens": 115},
        }
        compiled = compile_case(self.checkpoint, self.tokenizer, row, self.root)
        self.assertLessEqual(len(compiled.ids) + row["new_tokens"], 128)
        self.assertLessEqual(115 - len(compiled.ids), 3)
        self.assertTrue(compiled.text.startswith("Operations log\n\nAccess word: Blue."))
        self.assertTrue(compiled.text.endswith("Access word?"))
        self.assertEqual(compiled.record()["prompt_sha256"], digest(compiled.ids))
        shorter = compile_case(self.checkpoint, self.tokenizer, row, self.root, document_cap=30)
        self.assertLessEqual(len(shorter.ids), 30)
        self.assertGreaterEqual(len(shorter.ids), 28)

    def test_invalid_suite_rejects_duplicate_ids_and_multiple_sources(self):
        from tools.deepseek_prompts import read_suite

        row = {"id": "test", "category": "test", "new_tokens": 8, "text": "hi"}
        duplicate = "\n".join(json.dumps(case) for case in (row, row))
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            read_suite(duplicate.encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "exactly one prompt source"):
            read_suite(json.dumps(dict(row, messages=[])).encode("utf-8"))

    def test_smoke_records_the_prefix_before_every_generated_token(self):
        from engine import load_engine
        from tools.deepseek_prompts import compile_case, smoke_case

        row = {"id": "basic", "category": "basic", "new_tokens": 16, "text": "Hello, world!"}
        compiled = compile_case(self.checkpoint, self.tokenizer, row, self.root, output_cap=3)
        report = smoke_case(load_engine(self.root), compiled, {1})
        self.assertEqual(len(report["output_ids"]), len(report["logit_steps"]))
        self.assertEqual(report["output_sha256"], digest(report["output_ids"]))
        for index, step in enumerate(report["logit_steps"]):
            self.assertEqual(step["prefix_sha256"], digest(compiled.ids + report["output_ids"][:index]))
            self.assertEqual(step["top_logits"][0][0], report["output_ids"][index])
            self.assertEqual(len(step["logits_sha256"]), 64)

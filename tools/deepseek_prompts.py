"""Freeze DeepSeek prompt IDs and run reduced-checkpoint generation on CPU."""

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from checkpoint import Checkpoint
from engine import load_engine
from harness import checkpoint_identity, digest, greedy_ids, record_prompt
from prompt_input import PromptTokenizer, deepseek_prompt, eos_ids, prepare_prompt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "fixtures" / "deepseek-prompts.jsonl"


def read_suite(data):
    rows, ids = [], set()
    for line, text in enumerate(data.decode("utf-8").split("\n"), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(field), str) or not row[field] for field in ("id", "category")
        ):
            raise ValueError(f"Suite line {line} requires an ID and category")
        if row["id"] in ids:
            raise ValueError(f"Suite line {line} has duplicate ID {row['id']!r}")
        if sum(key in row for key in ("text", "messages", "document")) != 1:
            raise ValueError(f"Suite line {line} needs exactly one prompt source")
        if type(row.get("new_tokens")) is not int or row["new_tokens"] < 1:
            raise ValueError(f"Suite line {line} requires a positive new_tokens")
        if type(row.get("thinking", False)) is not bool:
            raise ValueError(f"Suite line {line} requires a boolean thinking value")
        if "text" in row and not isinstance(row["text"], str):
            raise ValueError(f"Suite line {line} requires text as a string")
        if "messages" in row and not isinstance(row["messages"], list):
            raise ValueError(f"Suite line {line} requires messages as an array")
        if "expected" in row and not isinstance(row["expected"], str):
            raise ValueError(f"Suite line {line} requires expected as a string")
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Prompt suite is empty")
    return rows


def _compile_document(spec, encode, cap):
    if not isinstance(spec, dict) or any(
        not isinstance(spec.get(field), str) or not spec[field].strip()
        for field in ("heading", "fact", "filler", "question")
    ):
        raise ValueError("Documents require a heading, fact, filler, and question")
    target = spec.get("target_tokens")
    if type(target) is not int or target < 1:
        raise ValueError("Document target_tokens must be a positive integer")
    if cap is not None:
        target = min(target, cap)
    head = spec["heading"] + "\n\n" + spec["fact"] + "\n\n"
    footer = "\n\n" + spec["question"]
    unit = spec["filler"].strip("\n") + "\n"
    minimum = encode(head + footer)
    if len(minimum) > target:
        raise ValueError(f"The document heading, fact and question exceed {target} tokens")
    if len(minimum) == target:
        return head + footer, minimum

    # Estimate a unit's token cost in its repeated context. Verify the final
    # length with the real tokenizer because BPE can change at both seams.
    sample = encode(head + unit * 16 + footer)
    stride = (len(sample) - len(minimum)) / 16
    if stride <= 0:
        raise ValueError("Document filler did not increase the encoded length")
    repeats = max(1, math.floor((target - len(minimum)) / stride))
    lower, upper = 0, None
    best = head + footer, minimum
    for _ in range(16):
        text = head + unit * repeats + footer
        ids = encode(text)
        gap = target - len(ids)
        if gap >= 0:
            lower, best = repeats, (text, ids)
            if gap == 0:
                return best
            grown = repeats + max(1, math.floor(gap / stride))
        else:
            upper = repeats
            grown = repeats - max(1, math.ceil(-gap / stride))
        if upper is not None:
            if upper - lower <= 1:
                return best
            grown = min(upper - 1, max(lower + 1, grown))
        repeats = grown
    raise ValueError("Could not converge on the document token budget")


@dataclass(frozen=True)
class CompiledCase:
    id: str
    category: str
    new_tokens: int
    ids: list
    tokenizer: PromptTokenizer
    image_prompt: object = None
    text: str | None = None
    expected: str | None = None

    def record(self):
        result = {
            "id": self.id, "category": self.category, "new_tokens": self.new_tokens,
            "prompt_tokens": len(self.ids), **record_prompt(self.ids, images=self.image_prompt),
        }
        if self.expected is not None:
            result["expected"] = self.expected
        return result


def compile_case(checkpoint, tokenizer, row, directory, *, document_cap=None, output_cap=None):
    new_tokens = row["new_tokens"]
    if output_cap is not None:
        new_tokens = min(new_tokens, output_cap)
    available = checkpoint.text_config["max_position_embeddings"] - new_tokens
    thinking = row.get("thinking", False)
    effort = row.get("reasoning_effort")
    images, text = None, row.get("text")
    if "document" in row:
        def encode(document):
            rendered, media = deepseek_prompt(
                [{"role": "user", "content": document}], thinking=thinking, reasoning_effort=effort,
            )
            if media:
                raise ValueError("Use structured messages for image documents")
            return tokenizer.encode(rendered)

        cap = min(available, document_cap) if document_cap is not None else available
        text, ids = _compile_document(row["document"], encode, cap)
    else:
        ids, _, images = prepare_prompt(
            checkpoint, text=text, messages=row.get("messages"), thinking=thinking, reasoning_effort=effort,
            base_directory=directory, tokenizer=tokenizer,
        )
    if not 0 < len(ids) <= available:
        raise ValueError(f"Encoded prompt and output budget exceed context ({len(ids)} + {new_tokens})")
    return CompiledCase(
        row["id"], row["category"], new_tokens, ids, tokenizer, images, text, row.get("expected"),
    )


def smoke_case(engine, compiled, stop_ids):
    steps, selected = [], []

    def observe(logits, token):
        if logits.shape != (engine.config.vocab_size,) or not np.isfinite(logits).all():
            raise ValueError("Unexpected or nonfinite logits")
        winners = np.argsort(-logits, kind="stable")[:5]
        steps.append({
            "prefix_sha256": digest(compiled.ids + selected),
            "logits_sha256": hashlib.sha256(logits.astype("<f4", copy=False).tobytes()).hexdigest(),
            "top_logits": [[int(i), float(logits[i])] for i in winners],
        })
        selected.append(token)

    output = greedy_ids(
        engine, compiled.ids, compiled.new_tokens, stop_ids, observe=observe,
        images=compiled.image_prompt.images if compiled.image_prompt else None,
    )
    return {
        **compiled.record(), "output_ids": output, "output_sha256": digest(output),
        "finish_reason": "eos" if output[-1] in stop_ids else "length",
        "output_text": compiled.tokenizer.decode([token for token in output if token not in stop_ids]),
        "logit_steps": steps,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as raw:
            temporary = Path(raw.name)
            if path.suffix == ".gz":
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                        stream.write("\n")
            else:
                raw.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "smoke"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True, help="Use .json.gz for full-length prompt IDs")
    parser.add_argument("--case", action="append", help="Include only these case IDs")
    parser.add_argument("--document-cap", type=int, help="Cap long-document inputs after chat encoding")
    parser.add_argument("--output-cap", type=int, help="Cap new tokens for each case")
    args = parser.parse_args(argv)
    try:
        if any(cap is not None and cap < 1 for cap in (args.document_cap, args.output_cap)):
            raise ValueError("Token caps must be positive")
        checkpoint = Checkpoint.open(args.checkpoint, metadata_only=True)
        if checkpoint.model_type != "deepseek_v41":
            raise ValueError("The workload requires a DeepSeek V4.1 checkpoint")
        tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
        suite_data = args.suite.read_bytes()
        rows = read_suite(suite_data)
        if args.case:
            missing = set(args.case) - {row["id"] for row in rows}
            if missing:
                raise ValueError(f"Unknown case IDs: {', '.join(sorted(missing))}")
            rows = [row for row in rows if row["id"] in args.case]

        results = []
        engine = None
        stop_ids = eos_ids(checkpoint)
        for row in rows:
            try:
                compiled = compile_case(
                    checkpoint, tokenizer, row, args.suite.parent,
                    document_cap=args.document_cap, output_cap=args.output_cap,
                )
                if args.mode == "smoke" and engine is None:
                    has_vision = bool(checkpoint.config.get("vision_config"))
                    engine = load_engine(args.checkpoint, with_vision=has_vision)
                record = compiled.record() if engine is None else smoke_case(engine, compiled, stop_ids)
            except (OSError, ValueError) as error:
                raise ValueError(f"{row['id']}: {error}") from error
            results.append(record)

        runtime = {
            "device": "cpu", "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "tokenizers": version("tokenizers"),
        }
        if any("images" in case for case in results):
            runtime.update(torch=version("torch"), pillow=version("pillow"))
        report = {
            "schema_version": 1, "mode": args.mode, "case_count": len(results),
            "categories": dict(Counter(case["category"] for case in results)), "runtime": runtime,
            "suite_sha256": hashlib.sha256(suite_data).hexdigest(),
            "tokenizer_sha256": tokenizer.sha256,
            "config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
            "document_cap": args.document_cap, "output_cap": args.output_cap,
            "stop_ids": sorted(stop_ids), "temperature": 0, "cases": results,
        }
        if engine is not None:
            report["checkpoint_identity"] = checkpoint_identity(args.checkpoint)
        _write_json(args.output, report)
        print(f"{args.mode}: {len(results)} cases, "
              f"{max(case['prompt_tokens'] for case in results):,} longest prompt tokens")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

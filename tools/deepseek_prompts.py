"""Freeze DeepSeek prompts, capture generation, and replay reference continuations."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from zipfile import BadZipFile, ZipFile

import numpy as np

from checkpoint import Checkpoint
from engine import load_engine
from harness import checkpoint_identity, digest, greedy_ids, record_prompt
from prompt_input import PromptTokenizer, deepseek_prompt, eos_ids, prepare_prompt
from tools.deepseek_logits import (
    capture_logit_archive, read_logit_capture, replay_case, summarize_logits, write_logit_capture,
)

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
        checker = row.get("checker")
        if checker is not None and (
            not isinstance(checker, dict) or checker.get("type") not in ("json", "tool", "image_order")
            or not isinstance(checker.get("schema"), dict) or "expected" not in row
        ):
            raise ValueError(f"Suite line {line} has an invalid answer checker")
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Prompt suite is empty")
    return rows


def _compile_document(spec, encode, cap):
    if not isinstance(spec, dict) or any(
        not isinstance(spec.get(field), str) or not spec[field].strip()
        for field in ("heading", "filler", "question")
    ):
        raise ValueError("Documents require a heading, filler, and question")
    target = spec.get("target_tokens")
    if type(target) is not int or target < 1:
        raise ValueError("Document target_tokens must be a positive integer")
    if cap is not None:
        target = min(target, cap)
    head = spec["heading"] + "\n\n"
    footer = "\n\n" + spec["question"]
    unit = spec["filler"].strip("\n") + "\n"
    sections = spec.get("facts", [{"at": 0, "text": spec.get("fact")}])
    if not isinstance(sections, list) or not sections or any(
        not isinstance(section, dict) or type(section.get("at")) not in (int, float)
        or not 0 <= section["at"] <= 1 or not isinstance(section.get("text"), str)
        or not section["text"].strip()
        for section in sections
    ):
        raise ValueError("Documents require facts with text and positions from 0 to 1")
    sections = sorted(sections, key=lambda section: section["at"])

    def render(repeats):
        parts, previous = [head], 0
        for section in sections:
            position = round(repeats * section["at"])
            parts.extend((unit * (position - previous), section["text"], "\n\n"))
            previous = position
        parts.extend((unit * (repeats - previous), footer))
        return "".join(parts)

    shortest = render(0)
    minimum = encode(shortest)
    if len(minimum) > target:
        raise ValueError(f"The required document text exceeds {target} tokens")
    if len(minimum) == target:
        return shortest, minimum

    # Estimate a unit's token cost in its repeated context. Verify the final
    # length with the real tokenizer because BPE can change at section boundaries.
    sample = encode(render(16))
    stride = (len(sample) - len(minimum)) / 16
    if stride <= 0:
        raise ValueError("Document filler did not increase the encoded length")
    repeats = max(1, math.floor((target - len(minimum)) / stride))
    lower, upper = 0, None
    best = shortest, minimum
    for _ in range(16):
        text = render(repeats)
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
    checker: dict | None = None
    thinking: bool = False

    def record(self):
        result = {
            "id": self.id, "category": self.category, "new_tokens": self.new_tokens,
            "prompt_tokens": len(self.ids), **record_prompt(self.ids, images=self.image_prompt),
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.checker is not None:
            result["checker"] = self.checker
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
        row.get("checker"), thinking,
    )


def smoke_case(engine, compiled, stop_ids, *, archive=None, upstream=False, score=False):
    steps, captured = [], []
    prefix_hash = hashlib.sha256(np.asarray(compiled.ids, dtype="<i8").tobytes())

    def observe(logits, token):
        values, step = summarize_logits(logits, engine.config.vocab_size, prefix_hash.hexdigest())
        steps.append(step)
        if archive is not None:
            captured.append(values.copy())
        prefix_hash.update(np.asarray([token], dtype="<i8").tobytes())

    images = compiled.image_prompt
    if images is not None and not upstream:
        images = images.images
    output = greedy_ids(
        engine, compiled.ids, compiled.new_tokens, stop_ids, observe=observe, images=images,
    )
    if archive is not None:
        write_logit_capture(archive, compiled.id, captured)
    result = {
        **compiled.record(), "output_ids": output, "output_sha256": digest(output),
        "finish_reason": "eos" if output[-1] in stop_ids else "length",
        "output_text": compiled.tokenizer.decode([token for token in output if token not in stop_ids]),
        "logit_steps": steps,
    }
    if score and compiled.expected is not None:
        from tools.deepseek_answers import score_answer

        # DeepSeek marks the reasoning boundary and tool calls with protocol tokens.
        text = compiled.tokenizer.backend.decode(
            [token for token in output if token not in stop_ids], skip_special_tokens=False,
        )
        result["answer_score"] = score_answer(compiled.expected, compiled.checker, text, thinking=compiled.thinking)
    return result


def _read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    return json.loads(path.read_bytes())


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
    parser.add_argument("mode", choices=("freeze", "smoke", "upstream", "replay"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True, help="Use .json.gz for full-length prompt IDs")
    parser.add_argument("--case", action="append", help="Include only these case IDs")
    parser.add_argument("--document-cap", type=int, help="Cap long-document inputs after chat encoding")
    parser.add_argument("--output-cap", type=int, help="Cap new tokens for each case")
    parser.add_argument("--capture-logits", type=Path, metavar="NPZ", help="Save float32 vocabulary vectors")
    parser.add_argument("--reference", type=Path, help="Greedy smoke or upstream report for replay")
    parser.add_argument("--reference-logits", type=Path, metavar="NPZ", help="Reference capture for full-vector errors")
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--upstream-source", type=Path, help="Pinned upstream inference/ directory")
    parser.add_argument("--converted-checkpoint", type=Path, help="Upstream modelR-mpN rank shards")
    parser.add_argument("--score", action="store_true", help="Score CPU smoke output against the answer checks")
    args = parser.parse_args(argv)
    try:
        upstream, replay = args.mode == "upstream", args.mode == "replay"
        if replay != bool(args.reference) or (args.reference_logits and not replay):
            raise ValueError("Replay requires --reference and accepts an optional --reference-logits")
        if upstream and not (args.upstream_source and args.converted_checkpoint):
            raise ValueError("Upstream capture requires --upstream-source and --converted-checkpoint")
        if args.capture_logits and args.mode == "freeze":
            raise ValueError("Freeze does not execute the model")
        paths = [path.resolve() for path in (
            args.output, args.capture_logits, args.reference, args.reference_logits,
        ) if path is not None]
        if len(paths) != len(set(paths)):
            raise ValueError("Reports and captures must use distinct paths")
        if any(not math.isfinite(tol) or tol < 0 for tol in (args.atol, args.rtol)):
            raise ValueError("Logit tolerances must be finite and nonnegative")
        reference = _read_json(args.reference) if replay else None
        if replay:
            if reference["mode"] not in ("smoke", "upstream"):
                raise ValueError("Replay requires a greedy smoke or upstream report")
            for option in ("document_cap", "output_cap"):
                if getattr(args, option) is None:
                    setattr(args, option, reference[option])
        if any(cap is not None and cap < 1 for cap in (args.document_cap, args.output_cap)):
            raise ValueError("Token caps must be positive")
        checkpoint = Checkpoint.open(args.checkpoint, metadata_only=True)
        if checkpoint.model_type != "deepseek_v41":
            raise ValueError("The workload requires a DeepSeek V4.1 checkpoint")
        tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
        suite_data = args.suite.read_bytes()
        rows = read_suite(suite_data)
        reference_cases = {case["id"]: case for case in reference["cases"]} if replay else {}
        stop_ids = eos_ids(checkpoint)
        if replay and len(reference_cases) != len(reference["cases"]):
            raise ValueError("Reference report has duplicate case IDs")
        if replay and (reference["tokenizer_sha256"] != tokenizer.sha256
                       or reference["stop_ids"] != sorted(stop_ids)):
            raise ValueError("Reference tokenizer or EOS IDs differ from the checkpoint")
        selection = args.case or (list(reference_cases) if replay else None)
        if selection is not None:
            missing = set(selection) - {row["id"] for row in rows}
            if replay:
                missing |= set(selection) - reference_cases.keys()
            if missing:
                raise ValueError(f"Unknown case IDs: {', '.join(sorted(missing))}")
            rows = [row for row in rows if row["id"] in selection]
        if not rows:
            raise ValueError("No cases selected")
        compiled_cases = []
        for row in rows:
            try:
                compiled_cases.append(compile_case(
                    checkpoint, tokenizer, row, args.suite.parent,
                    document_cap=args.document_cap, output_cap=args.output_cap,
                ))
            except (OSError, ValueError) as error:
                raise ValueError(f"{row['id']}: {error}") from error

        results, engine = [], None
        writer = not upstream or int(os.environ.get("RANK", "0")) == 0
        with ExitStack() as stack:
            archive = stack.enter_context(capture_logit_archive(args.capture_logits)) if writer and args.capture_logits else None
            reference_archive = stack.enter_context(ZipFile(args.reference_logits)) if args.reference_logits else None
            if upstream:
                from tools.deepseek_upstream_runner import open_upstream_engine

                capacity = max(len(case.ids) + case.new_tokens - 1 for case in compiled_cases)
                engine = stack.enter_context(open_upstream_engine(
                    args.upstream_source, args.converted_checkpoint, checkpoint, tokenizer, capacity,
                ))
            elif args.mode != "freeze":
                engine = load_engine(args.checkpoint, with_vision=any(case.image_prompt for case in compiled_cases))
            for compiled in compiled_cases:
                try:
                    if replay:
                        expected = reference_cases[compiled.id]
                        vectors = (read_logit_capture(reference_archive, expected, engine.config.vocab_size)
                                   if reference_archive is not None else None)
                        record = replay_case(
                            engine, compiled, stop_ids, expected, vectors=vectors, archive=archive,
                            atol=args.atol, rtol=args.rtol,
                        )
                    elif engine is not None:
                        record = smoke_case(
                            engine, compiled, stop_ids, archive=archive, upstream=upstream,
                            score=writer and (args.score or upstream),
                        )
                    else:
                        record = compiled.record()
                except (OSError, ValueError, KeyError) as error:
                    raise ValueError(f"{compiled.id}: {error}") from error
                if writer:
                    results.append(record)
        if not writer:
            return 0

        runtime = {
            "device": "cuda" if upstream else "cpu", "machine": platform.machine(), "python": platform.python_version(),
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
        if args.capture_logits:
            report["logit_capture"] = args.capture_logits.name
        if replay:
            report.update(
                reference_mode=reference["mode"], reference_suite_sha256=reference["suite_sha256"],
                full_vector_comparison=args.reference_logits is not None, atol=args.atol, rtol=args.rtol,
            )
        if upstream:
            from tools.deepseek_upstream_runner import REVISION

            report.update(upstream_revision=REVISION, world_size=int(os.environ.get("WORLD_SIZE", "1")))
        elif engine is not None:
            report["checkpoint_identity"] = checkpoint_identity(args.checkpoint)
        scores = [case["answer_score"]["passed"] for case in results if "answer_score" in case]
        if scores:
            report["answer_scores"] = {"passed": sum(scores), "scored": len(scores)}
        _write_json(args.output, report)
        print(f"{args.mode}: {len(results)} cases, "
              f"{max(case['prompt_tokens'] for case in results):,} longest prompt tokens")
        return 0
    except (OSError, ValueError, KeyError, ImportError, BadZipFile) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

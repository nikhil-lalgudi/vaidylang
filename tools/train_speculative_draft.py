"""Distill a small CPU drafter from greedy CUDA Qwen3 continuations."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gpu_bench import PINNED_CHECKPOINT
from harness import checkpoint_identity, digest
from speculative_bench import TARGET_SETTINGS, target_tokens


TRAIN_PROMPTS = (
    "Implement binary search in Python, returning the insertion point if the item is missing. Add type hints.",
    "Write a Python context manager for timing a block with perf_counter. Include an example.",
    "Implement a Python function that reads a CSV into a list of dictionaries and rejects duplicate headers.",
    "Write a Python function that yields successive windows of size n from an iterable.",
    "Use Python to implement an LRU cache with a fixed capacity. Show the source code only.",
    "Write a Python program that converts Celsius temperatures from stdin to Fahrenheit.",
    "Give me a Python dataclass with validated fields for latitude and longitude. Add example usage.",
    "Write a JavaScript function that waits for the first successful promise and rejects if they all fail.",
    "Create a JavaScript function for throttling scroll events to once per animation frame.",
    "Show a TypeScript function that groups objects by a string key. Include a reusable generic signature.",
    "Use TypeScript to model JSON values as a recursive union and write a safe parse wrapper.",
    "Write a TypeScript function that reports the first date range overlap, returning null if none overlap.",
    "Write HTML and CSS for a responsive photo grid with accessible captions. Return only the code.",
    "Write a PostgreSQL query to list products that have never sold. The tables are products and line_items.",
    "Given readings(device_id, recorded_at, value), show the latest reading per device using a SQL window function.",
    "Write a SQLite statement for a composite index that helps lookups by customer and creation date.",
    "Implement a Rust function to count character frequencies and return them as a sorted vector.",
    "Show a Go function for reading the first nonempty line of a file. Handle errors explicitly.",
    "In C++, implement a function that removes consecutive duplicates from a sorted vector in place.",
    "Write a Bash script that checks whether a path is a regular file and prints its size in bytes.",
    "Explain why a cast-iron pan seems to keep food hot for longer. Use familiar examples.",
    "Draft a friendly reminder that neighbors need to bring their bins in by Sunday evening.",
    "Write a short email declining an invitation to speak at a workshop because of a prior commitment.",
    "Explain the difference between a fixed-rate and variable-rate energy plan in two paragraphs.",
    "Give me a 20-minute agenda for a weekly meetup of first-time language learners.",
    "Suggest an indoor afternoon activity for a family with a six-year-old and a teenager. Use items they already own.",
    "Turn these notes into a volunteer thank-you letter: 20 bags of litter collected, Saturday rain, 12 helpers.",
    "Explain the difference between a comet and an asteroid to a ten-year-old.",
    "Write a polite two-sentence reminder that a library book is three days overdue.",
    "Help me tell my manager I need to move our meeting to next week because of a scheduling conflict.",
    "Suggest five ways to use leftover rice for lunch and identify which ingredients each one needs.",
    "Give me a short checklist for preparing a room for a remote video interview.",
)

VALIDATION_PROMPTS = (
    "Write a Python function for merging adjacent inclusive integer intervals with examples.",
    "Use TypeScript to implement a helper that waits for an AbortSignal with a timeout.",
    "Write one SQL query that finds the three most-used tags in posts and expands ties at rank three.",
    "Implement breadth-first traversal of a graph in JavaScript. Show only the function.",
    "Write a warm email to a house guest explaining where to find extra towels and the Wi-Fi password.",
    "What causes the seasons and why are they opposite in the northern and southern hemispheres?",
    "Outline a weekday routine for learning to play the piano in fifteen minutes a day.",
    "Draft two sentences for a club newsletter thanking members for donating books.",
)


def training_examples(sequences, classes, context):
    features, labels = [], []
    padding = len(classes) + 1
    for prompt, continuation in sequences:
        history = [padding] * context + [classes.get(token, 0) for token in prompt[-context:]]
        for token in continuation:
            label = classes.get(token, 0)
            features.append(history[-context:])
            labels.append(label)
            history.append(label)
    return features, labels


def score(logits, labels, confidence):
    probability, prediction = logits.softmax(dim=1).max(dim=1)
    proposed = (prediction != 0) & (probability >= confidence)
    correct = (prediction != 0) & (prediction == labels)
    proposals = int(proposed.sum())
    return {
        "tokens": len(labels),
        "shortlist_coverage": float((labels != 0).float().mean()),
        "top1_accuracy": float(correct.float().mean()),
        "proposals": proposals,
        "proposal_coverage": proposals / len(labels),
        "proposal_accuracy": float(correct[proposed].float().mean()) if proposals else None,
    }


def train_draft(train, validation, *, seed, context, vocabulary_size, embedding_size,
                hidden_size, epochs, confidence):
    import torch
    from torch import nn

    counts = Counter(token for _, continuation in train for token in continuation)
    vocabulary = sorted(counts, key=lambda token: (-counts[token], token))[:vocabulary_size]
    if not vocabulary:
        raise ValueError("No training tokens were generated")
    classes = {token: i for i, token in enumerate(vocabulary, start=1)}
    train_x, train_y = map(torch.tensor, training_examples(train, classes, context))
    val_x, val_y = map(torch.tensor, training_examples(validation, classes, context))
    if not len(val_y):
        raise ValueError("No validation tokens were generated")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        embed = nn.Embedding(len(classes) + 2, embedding_size)
        hidden = nn.Linear(context * embedding_size, hidden_size)
        output = nn.Linear(hidden_size, len(classes) + 1)
        model = nn.Sequential(embed, nn.Flatten(start_dim=1), hidden, nn.ReLU(), output)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            for batch in torch.randperm(len(train_y), generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(train_x[batch]), train_y[batch])
                loss.backward()
                optimizer.step()

    with torch.inference_mode():
        training_metrics = score(model(train_x), train_y, confidence)
        validation_metrics = score(model(val_x), val_y, confidence)
    weights = {
        "vocabulary": np.asarray(vocabulary, dtype=np.int64),
        "embeddings": embed.weight.detach().numpy(),
        "hidden_weight": hidden.weight.detach().numpy().T.copy(),
        "hidden_bias": hidden.bias.detach().numpy(),
        "output_weight": output.weight.detach().numpy().T.copy(),
        "output_bias": output.bias.detach().numpy(),
        "confidence": np.asarray(confidence),
    }
    return weights, {"training": training_metrics, "validation": validation_metrics}


def generate_continuations(checkpoint, prompts, new_tokens):
    import torch
    from transformers import AutoTokenizer

    from cuda_engine import CudaQwen3Engine, DecodeGraph

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    inputs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    ) for prompt in prompts]
    generation_path = checkpoint / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    eos = generation_config.get("eos_token_id", tokenizer.eos_token_id)
    stop_tokens = set(eos if isinstance(eos, list) else [eos])

    engine = CudaQwen3Engine.from_checkpoint(checkpoint, dtype=torch.bfloat16, **TARGET_SETTINGS)
    state = engine.allocate_cache(max(map(len, inputs)) + new_tokens)
    graph = DecodeGraph(engine, state, inputs[0])
    sequences = []
    with torch.inference_mode():
        for ids in inputs:
            continuation = []
            for token in target_tokens(graph, ids, new_tokens, annotate=None):
                continuation.append(token)
                if token in stop_tokens:
                    break
            sequences.append((ids, continuation))
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output draft weights (.npz)")
    parser.add_argument("--report", type=Path, required=True, help="Output training report (.json)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--vocabulary-size", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.new_tokens, args.context, args.vocabulary_size, args.embedding_size, args.hidden_size,
           args.epochs) < 1 or not 0 <= args.confidence <= 1 or not 0 <= args.seed < 2**64:
        parser.error("Expected positive sizes and epochs, a confidence in [0, 1], and an unsigned 64-bit seed")
    if args.output.resolve() == args.report.resolve():
        parser.error("Weights and report require different output paths")
    if checkpoint_identity(args.checkpoint) != PINNED_CHECKPOINT:
        parser.error("Checkpoint configuration or weights differ from the pinned revision")

    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("Expose one NVIDIA CUDA GPU with native BF16")
    torch.set_num_threads(1)
    started = perf_counter()
    sequences = generate_continuations(args.checkpoint, TRAIN_PROMPTS + VALIDATION_PROMPTS, args.new_tokens)
    generated = perf_counter()
    train, validation = sequences[:len(TRAIN_PROMPTS)], sequences[len(TRAIN_PROMPTS):]
    settings = {key: getattr(args, key) for key in (
        "seed", "context", "vocabulary_size", "embedding_size", "hidden_size", "epochs", "confidence",
    )}
    weights, metrics = train_draft(train, validation, **settings)
    finished = perf_counter()
    report = {
        "checkpoint": PINNED_CHECKPOINT,
        "target": TARGET_SETTINGS,
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "new_tokens_limit": args.new_tokens,
        "settings": settings,
        "generation_seconds": generated - started,
        "training_seconds": finished - generated,
    }
    for name, split in (("training", train), ("validation", validation)):
        report[name] = {
            **metrics[name],
            "prompts": [
                {"input_sha256": digest(prompt), "output_sha256": digest(continuation), "tokens": len(continuation)}
                for prompt, continuation in split
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.savez(output, **weights)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

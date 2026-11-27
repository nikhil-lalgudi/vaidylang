"""Exercise the standalone executable against the NumPy reference."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from engine import Qwen3Engine, load_engine, tiny_weights, weight_shapes
from harness import greedy_ids
from test_engine import write_checkpoint_weights


def run(executable, path, ids, count):
    return subprocess.run(
        [executable, str(path), str(count), *map(str, ids)], capture_output=True, text=True,
    )


def main(executable):
    config, weights = tiny_weights()
    for sharded in (False, True):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            current = config if not sharded else replace(config, tie_word_embeddings=False)
            (path / "config.json").write_text(json.dumps({
                **asdict(current), "model_type": "qwen3", "hidden_act": "silu",
            }))
            stored = {name: ("F32", weight.astype("<f4")) for name, weight in weights.items()}
            if sharded:
                stored["lm_head.weight"] = ("F16", weights["model.embed_tokens.weight"].astype("<f2"))
                bits = weights["model.norm.weight"].view(np.uint32)
                stored["model.norm.weight"] = ("BF16", (bits >> 16).astype("<u2"))
                names = list(stored)
                files = {name: f"model-{index % 2}.safetensors" for index, name in enumerate(names)}
                for shard in sorted(set(files.values())):
                    write_checkpoint_weights(path / shard, {name: stored[name] for name in names if files[name] == shard})
                (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": files}))
            else:
                write_checkpoint_weights(path / "model.safetensors", stored)
            reference = Qwen3Engine.from_checkpoint(path)
            for ids, count in [([3], 1), ([3, 1, 4, 1, 5], 12), (list(range(64)) * 3, 5)]:
                result = run(executable, path, ids, count)
                assert result.returncode == 0, result.stderr
                assert json.loads(result.stdout) == greedy_ids(reference, ids, count)
            invalid = run(executable, path, [3, 64], 2)
            assert invalid.returncode != 0 and "vocabulary" in invalid.stderr
            invalid = run(executable, path, [0], current.max_position_embeddings + 1)
            assert invalid.returncode != 0 and "context limit" in invalid.stderr
            if os.environ.get("VAIDYLANG_CPU_LIBRARY"):
                native = load_engine(path, cpu_backend="native")
                args = [sys.executable, "harness.py", "generate", "--checkpoint", str(path),
                        "--ids", "3", "1", "4", "--new-tokens", "8"]
                replay = json.loads(subprocess.check_output(args, text=True))
                assert replay["cpu_backend"] == "native"
                assert replay["output_ids"] == greedy_ids(reference, [3, 1, 4], 8)
                for ids in ([3], [3, 1, 4, 1, 5], list(range(64)) * 3):
                    actual, state = native.prefill(ids, len(ids) + 4)
                    expected, oracle = reference.prefill(ids, len(ids) + 4)
                    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
                    saved = actual.copy()
                    for token in (9, 2, 7, 8):
                        np.testing.assert_allclose(
                            native.decode(token, state), reference.decode(token, oracle), rtol=3e-5, atol=3e-5,
                        )
                    assert state.length == len(ids) + 4
                    np.testing.assert_array_equal(actual, saved)
                    with np.testing.assert_raises_regex(ValueError, "capacity exceeded"):
                        native.decode(0, state)
                with np.testing.assert_raises_regex(ValueError, "outside the vocabulary"):
                    native.prefill(np.array([2**32], dtype=np.uint64), 1)
                _, state = native.prefill([2, 3], 4)
                other = load_engine(path, cpu_backend="native")
                with np.testing.assert_raises_regex(ValueError, "different model"):
                    other.decode(4, state)
                assert state.length == 2
            shard = path / (files[names[0]] if sharded else "model.safetensors")
            shard.write_bytes(shard.read_bytes()[:24])
            invalid = run(executable, path, [1], 2)
            assert invalid.returncode != 0 and "header" in invalid.stderr


def check_large_bf16():
    config, _ = tiny_weights()
    config = replace(config, hidden_size=257, vocab_size=65537)
    rng = np.random.default_rng(15)
    tensors = {}
    for name, shape in weight_shapes(config).items():
        if name == "model.embed_tokens.weight":
            embedding = np.zeros(shape, dtype="<u2")
            tokens = [0, 1, 3, 9, 17, config.vocab_size - 1]
            values = rng.standard_normal((len(tokens), shape[1]), dtype=np.float32)
            embedding[tokens] = (values.view(np.uint32) >> 16).astype("<u2")
            tensors[name] = ("BF16", embedding)
        else:
            data = rng.standard_normal(shape, dtype=np.float32)
            values = 1 + 0.1 * data if len(shape) == 1 else data / np.sqrt(np.float32(shape[-1]))
            tensors[name] = ("F32", values.astype("<f4"))
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "config.json").write_text(json.dumps({
            **asdict(config), "model_type": "qwen3", "hidden_act": "silu",
        }))
        write_checkpoint_weights(path / "model.safetensors", tensors)
        reference = Qwen3Engine.from_checkpoint(path)
        engine = load_engine(path, cpu_backend="native")
        ids = [3, config.vocab_size - 1, 1]
        actual, state = engine.prefill(ids, 4)
        expected, oracle = reference.prefill(ids, 4)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(engine.decode(17, state), reference.decode(17, oracle), rtol=3e-5, atol=3e-5)


def check_split_projections():
    config, _ = tiny_weights()
    # Odd widths exercise vector tails, uneven splits, and residual adds.
    config = replace(config, hidden_size=513, intermediate_size=2049)
    rng = np.random.default_rng(82)
    tensors = {}
    for name, shape in weight_shapes(config).items():
        data = rng.standard_normal(shape, dtype=np.float32)
        values = 1 + 0.1 * data if len(shape) == 1 else data / np.sqrt(np.float32(shape[-1]))
        # The first layer can keep both gate/up segments packed. The second
        # layer must preserve its BF16 gate when the FP32 up segment is loaded.
        if name.endswith(("gate_proj.weight", "down_proj.weight")) or name == "model.layers.0.mlp.up_proj.weight":
            tensors[name] = ("BF16", (values.view(np.uint32) >> 16).astype("<u2"))
        else:
            tensors[name] = ("F32", values.astype("<f4"))
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "config.json").write_text(json.dumps({
            **asdict(config), "model_type": "qwen3", "hidden_act": "silu",
        }))
        write_checkpoint_weights(path / "model.safetensors", tensors)
        reference = Qwen3Engine.from_checkpoint(path)
        native = load_engine(path, cpu_backend="native")

        def check(length):
            prompt = list(range(length))
            actual, state = native.prefill(prompt, length + 2)
            expected, oracle = reference.prefill(prompt, length + 2)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
            for token in (9, 5):
                np.testing.assert_allclose(
                    native.decode(token, state), reference.decode(token, oracle), rtol=3e-5, atol=3e-5,
                )

        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(check, (1, 8, 17, 32)))


if __name__ == "__main__":
    main(sys.argv[1])
    if os.environ.get("VAIDYLANG_CPU_LIBRARY"):
        check_split_projections()
    if sys.platform == "darwin" and os.environ.get("VAIDYLANG_CPU_LIBRARY"):
        check_large_bf16()

"""Exercise the standalone executable against the NumPy reference."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from engine import Qwen3Engine, load_engine, tiny_weights
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


if __name__ == "__main__":
    main(sys.argv[1])

"""Compare CPU prompt, image, and Engram behavior with pinned upstream files.

Use --list-files to list the source and fixture files to fetch from the pinned
Hugging Face revision. Pass their local root directory as --upstream to run.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import Checkpoint
from deepseek_config import DeepSeekConfig
from deepseek_engram import EngramHash, compressed_token_map
from deepseek_vision import prepare_image_prompt
from prompt_input import PromptTokenizer, deepseek_prompt


REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
SOURCE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}"
INPUT_SHA256 = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "encoding/encoding.py": "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2",
    "encoding/tests/test_input_1.json": "6a5a86694e47169c8f0c9243a7e24cc5db05447be0d3d3a9891034e913c518de",
    "encoding/tests/test_input_2.json": "fd985cda88e8640b8e2404d8e659f7006cad204eb41f73c9e6662b1f218c01aa",
    "encoding/tests/test_input_3.json": "4207613f15c79b95ad92074bdb8b2b81c5ce9cfdf15b7a67f58ded6bb3ec4fa5",
    "encoding/tests/test_input_4.json": "4c3890a7bc042ec6396d98d024a39a73cdcffbe1d0253e852da0c35e65296f3e",
    "encoding/tests/test_input_5.json": "88289b0aac8d650aa55c4ffeb392818fff3a0ac83c2293ddfd90546804d5e5c4",
    "encoding/tests/test_output_1.txt": "91f79aa955717b414a82aabfb38bcd83ab5a998bd47db89d2a26338e8d3de550",
    "encoding/tests/test_output_2.txt": "9fa162587b73ab9b315be5c88d433f9e0e681b4b36926952f289099ed9bc1dab",
    "encoding/tests/test_output_3.txt": "cf2b801d0738c88abb691bf4a096000c0e0737faab8a54a4ba5243c5513a3711",
    "encoding/tests/test_output_4.txt": "94433b68370dcc69fa61efc5011b5d0e28685e4716fee1ee77fbbfc0c9935b54",
    "encoding/tests/test_output_5.txt": "4d673091746553d1cb2e40a3fc01dc09a7c53c40627096a3e49c97a269a81f64",
    "inference/config.json": "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
    "inference/engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "inference/image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
    "inference/examples/images/carrots.jpeg": "5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a",
    "inference/examples/images/corn.jpeg": "251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def integer_digest(values):
    return sha256(np.asarray(values, dtype="<i8").tobytes())


def verify_assets(upstream):
    inputs = {}
    for relative, expected in INPUT_SHA256.items():
        data = (upstream / relative).read_bytes()
        require(sha256(data) == expected, f"SHA-256 mismatch for {relative} at revision {REVISION}")
        inputs[relative] = data
    return inputs


def load_module(name, path, verified_source):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(verified_source, str(path), "exec"), module.__dict__)
    return module


class UpstreamTokenizer:
    """Give the pinned CPU functions their tokenizer interface over the Rust backend."""

    def __init__(self, data, config):
        self.backend_tokenizer = Tokenizer.from_buffer(data)
        self.unk_token_id = self.convert_tokens_to_ids(config["unk_token"]) if config["unk_token"] else None

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()

    def encode(self, text):
        return self.backend_tokenizer.encode(text).ids

    def convert_tokens_to_ids(self, text):
        return self.backend_tokenizer.token_to_id(text)


def check_encoding(upstream, inputs, encoding, tokenizer, reference_tokenizer):
    prompts, checks = [], []
    for number in range(1, 6):
        case_name = f"encoding/tests/test_input_{number}.json"
        case, = encoding.load_cases(str(upstream / case_name))
        require(not case["context"], f"Encoding fixture {number} has separate history")
        mode = case["thinking_mode"] or "chat"
        reference, reference_images = encoding.encode_case(case, thinking_mode="chat")
        prompt, images = deepseek_prompt(case["messages"], thinking=mode == "thinking",
                                         reasoning_effort=case["reasoning_effort"])
        expected = inputs[f"encoding/tests/test_output_{number}.txt"]
        require(prompt.encode("utf-8") == reference.encode("utf-8") == expected,
                f"Prompt mismatch in official encoding fixture {number}")
        require(images == reference_images, f"Media mismatch in official encoding fixture {number}")
        ids = tokenizer.encode(prompt)
        require(ids == reference_tokenizer.encode(reference), f"Token mismatch in encoding fixture {number}")
        prompts.append((prompt, ids, images))
        checks.append({
            "fixture": number, "thinking_mode": mode, "reasoning_effort": case["reasoning_effort"],
            "prompt_sha256": sha256(expected), "token_ids_sha256": integer_digest(ids),
            "tokens": len(ids), "images": len(images), "exact_match": True,
        })
    return prompts, checks


def check_images(upstream, checkpoint, tokenizer, reference_tokenizer, args, processor, prompt):
    text, ids, records = prompt
    image_root = upstream / "inference"
    reference_records = [record | {"url": str(image_root / record["url"])} for record in records]
    expected_ids, expected_types, expected_images = processor.prepare_vl_inputs(
        text, reference_records, reference_tokenizer, args,
    )
    actual = prepare_image_prompt(checkpoint, tokenizer, ids, records, image_root)
    require(actual.ids == expected_ids, "Expanded image prompt token IDs differ")
    require(actual.token_types == expected_types, "Expanded image prompt token types differ")
    for image, expected in zip(actual.images, expected_images, strict=True):
        require((image.start, image.n_vit_h, image.n_vit_w) == (expected.start, expected.n_vit_h, expected.n_vit_w),
                "Image span position or patch grid differs")
        torch.testing.assert_close(image.types, expected.types, atol=0, rtol=0)
        require(image.patches.dtype == expected.patches.dtype == torch.bfloat16, "Image patches are not BF16")
        torch.testing.assert_close(image.patches.view(torch.int16), expected.patches.view(torch.int16), atol=0, rtol=0)
    return actual, {
        "image_token_id": args.image_token_id,
        "original_tokens": len(ids), "expanded_tokens": len(actual.ids),
        "expanded_token_ids_sha256": integer_digest(actual.ids),
        "token_types_sha256": integer_digest(actual.token_types),
        "images": [metadata | {"file": record["url"]} for metadata, record in zip(actual.metadata, records)],
        "exact_match": True,
    }


def hash_prefixes(hasher, layers, ids, live):
    mapped, hashes = [], []
    for token, is_text in zip(ids, live, strict=True):
        mapped.append(int(hasher.token_map[token]) if is_text else hasher.DEAD)
        hashes.append([hasher.hashes(layer, mapped) for layer in layers])
    return mapped, np.asarray(hashes)


def check_engram(upstream, checkpoint, reference_tokenizer, args, engram, cases):
    config = DeepSeekConfig.from_mapping(checkpoint.config)
    require(not (upstream / "engram_token_map.json").exists(), "Remove the sidecar to test tokenizer-derived Engram")
    token_map = compressed_token_map(config, upstream)
    hasher = EngramHash(config, token_map)
    args.max_seq_len = max(len(ids) for _, ids, _, _ in cases)
    layout = engram.EngramLayout.from_args(args)
    reference = engram.NgramHashState(args, layout, reference_tokenizer)
    layers = config.engram_layer_ids
    require(layers == layout.layer_ids, "Engram layer ordering differs")
    require(config.engram_num_embeddings == layout.num_embeddings, "Engram table row counts differ")
    require(hasher.DEAD == reference.DEAD and hasher.pad == reference.pad_id, "Engram DEAD or pad ID differs")
    # NgramHashState constructs its table through upstream build_compressed_token_map.
    np.testing.assert_array_equal(token_map, reference.token_map.numpy(), err_msg="Engram token map")
    parameters = {"primes": hasher.primes, "offsets": hasher.offsets, "multipliers": hasher.multipliers}
    for name, values in parameters.items():
        np.testing.assert_array_equal([values[layer] for layer in layers], getattr(reference, name).numpy(),
                                      err_msg=f"Engram {name}")

    matrices = []
    for name, ids, live, prefill in cases:
        mapped, hashes = hash_prefixes(hasher, layers, ids, live)
        input_ids = torch.tensor([ids], dtype=torch.int64)
        mask = torch.tensor([live], dtype=torch.bool)
        reference.cache.fill_(reference.DEAD)
        full = reference(input_ids, 0, mask)
        reference.cache.fill_(reference.DEAD)
        split = [reference(input_ids[:, :prefill], 0, mask[:, :prefill])]
        for pos in range(prefill, len(ids)):
            step = slice(pos, pos + 1)
            split.append(reference(input_ids[:, step], pos, mask[:, step]))
        np.testing.assert_array_equal(mapped, reference.cache[0, :len(ids)].numpy(), err_msg=f"{name} cache")
        np.testing.assert_array_equal(hashes, full[0].numpy(), err_msg=f"{name} full prefix")
        np.testing.assert_array_equal(hashes, torch.cat(split, dim=1)[0].numpy(), err_msg=f"{name} split prefix")
        matrices.append({
            "case": name, "shape": list(hashes.shape), "prefill_tokens": prefill,
            "scalar_decode_steps": len(ids) - prefill, "masked_tokens": len(ids) - sum(live),
            "input_ids_sha256": integer_digest(ids), "mapped_history_sha256": integer_digest(mapped),
            "hash_matrix_sha256": integer_digest(hashes), "exact_match": True,
        })
    return {
        "token_ids_checked": len(token_map), "compressed_vocab_size": len(np.unique(token_map)),
        "token_map_sha256": integer_digest(token_map), "pad_compressed_id": hasher.pad, "dead_id": hasher.DEAD,
        "layers": list(layers), "table_rows": list(config.engram_num_embeddings),
        **{f"{name}_sha256": integer_digest([values[layer] for layer in layers]) for name, values in parameters.items()},
        "prefix_hash_matrices": matrices, "exact_match": True,
    }


def run(upstream):
    inputs = verify_assets(upstream)
    checkpoint = Checkpoint(upstream, json.loads(inputs["config.json"]), None, (), None)
    tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
    reference_tokenizer = UpstreamTokenizer(inputs["tokenizer.json"], json.loads(inputs["tokenizer_config.json"]))
    args = SimpleNamespace(**json.loads(inputs["inference/config.json"]), max_batch_size=1, vision_enabled=True)
    # The pinned image processor imports encoding by its upstream module name.
    encoding = load_module("encoding", upstream / "encoding/encoding.py", inputs["encoding/encoding.py"])
    processor = load_module("deepseek_upstream_images", upstream / "inference/image_processor.py",
                            inputs["inference/image_processor.py"])
    engram = load_module("deepseek_upstream_engram", upstream / "inference/engram.py", inputs["inference/engram.py"])
    prompts, encoding_checks = check_encoding(upstream, inputs, encoding, tokenizer, reference_tokenizer)
    vision, image_checks = check_images(upstream, checkpoint, tokenizer, reference_tokenizer, args,
                                        processor, prompts[4])
    cases = [(f"encoding_fixture_{i}", ids, [True] * len(ids), len(ids) - 3)
             for i, (_, ids, _) in enumerate(prompts[:4], 1)]
    suffix = tokenizer.encode("The images show carrots and corn.")
    cases.append(("encoding_fixture_5_expanded_images", vision.ids + suffix,
                  [kind == processor.TEXT for kind in vision.token_types] + [True] * len(suffix), len(vision.ids)))
    image_id = args.image_token_id
    seed = prompts[1][1][:8]
    ids = [image_id, image_id, *seed[:3], image_id, *seed[3:5], image_id, *seed[5:], image_id]
    cases.append(("leading_consecutive_interior_and_trailing_dead_tokens", ids, [i != image_id for i in ids], 6))
    return {
        "revision": REVISION, "source": SOURCE, "device": "cpu", "passed": True,
        "asset_sha256": INPUT_SHA256, "integer_digest_format": "row-major little-endian signed int64",
        "encoding": encoding_checks, "image_preprocessing": image_checks,
        "engram": check_engram(upstream, checkpoint, reference_tokenizer, args, engram, cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--upstream", type=Path, help="Local directory with pinned source and fixtures")
    source.add_argument("--list-files", action="store_true", help="List the pinned assets needed to run")
    parser.add_argument("--report", type=Path, help="Write the JSON report to this file instead of stdout")
    args = parser.parse_args()
    if args.list_files:
        print("\n".join(INPUT_SHA256))
        return
    torch.set_num_threads(1)
    report = json.dumps(run(args.upstream), indent=2) + "\n"
    if args.report is None:
        sys.stdout.write(report)
    else:
        args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

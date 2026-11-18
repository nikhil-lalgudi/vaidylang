"""Run the published DeepSeek ViT and aligner on CPU using the vision-only shard."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys

from checkpoint import Checkpoint
from deepseek_vision import bf16_digest, encode_image_rows, vision_args, vision_modules
from prompt_input import PromptTokenizer, prepare_prompt
from tools.deepseek_headers import REVISION


PINNED_SHA256 = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "model-00001-of-00048.safetensors": "886aebdafa08cc27bbae2165ed35bdfe0de9370bf88c1411283c155c6ae4ff89",
}


def verify_asset(path, reference):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != PINNED_SHA256[reference]:
        raise ValueError(f"{path} does not match the pinned {reference}")
    return actual


def probe(checkpoint, shard, paths, threads, image_token_cap=None, dtype="bfloat16"):
    try:
        import PIL
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise ValueError("Install PyTorch, safetensors and Pillow for the vision probe") from error

    tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
    config_digest = verify_asset(checkpoint.directory / "config.json", "config.json")
    if tokenizer.sha256 != PINNED_SHA256["tokenizer.json"]:
        raise ValueError("Expected the pinned published tokenizer")
    shard_digest = verify_asset(shard, "model-00001-of-00048.safetensors")
    torch.set_num_threads(threads)

    args = vision_args(checkpoint)
    if image_token_cap is not None:
        if not 4 <= image_token_cap <= args.vision_max_n_token:
            raise ValueError(f"Image token cap must be between 4 and {args.vision_max_n_token}")
        config = dict(checkpoint.config, vision_config=dict(
            checkpoint.config["vision_config"], max_image_tokens=image_token_cap,
        ))
        checkpoint = replace(checkpoint, config=config)

    modules = vision_modules(args)
    storage_dtype = getattr(torch, dtype)
    with safe_open(shard, framework="pt", device="cpu") as weights:
        for name, module in modules.items():
            module.load_state_dict({
                key: weights.get_tensor(f"{name}.{key}").to(storage_dtype) for key in module.state_dict()
            }, assign=True)
            module.eval()

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe the images in order."},
        *[{"type": "image_url", "image_url": {"url": str(path.resolve())}} for path in paths],
    ]}]
    _, _, prepared = prepare_prompt(checkpoint, messages=messages, tokenizer=tokenizer)
    inputs = []
    for path, image, metadata in zip(paths, prepared.images, prepared.metadata, strict=True):
        rows = encode_image_rows(modules, image)
        height, width = metadata["llm_grid"]
        if rows.shape != (height * width, args.dim) or not bool(rows.isfinite().all()):
            raise ValueError(f"Invalid aligned features for {path.name}")
        if rows.dtype == torch.bfloat16:
            checksum = bf16_digest(rows)
        else:
            checksum = hashlib.sha256(rows.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        inputs.append({
            "file": path.name, **metadata, "aligned_shape": list(rows.shape),
            "aligned_sha256": checksum, "aligned_l2_norm": float(rows.float().norm()),
        })
        print(f"{path.name}: {rows.shape[0]} aligned rows, all finite", flush=True)

    return {
        "schema_version": 1, "revision": REVISION, "device": "cpu", "dtype": dtype,
        "machine": platform.machine(), "torch": torch.__version__, "pillow": PIL.__version__,
        "threads": threads, "config_sha256": config_digest, "tokenizer_sha256": tokenizer.sha256,
        "vision_shard_sha256": shard_digest, "image_token_cap": image_token_cap or args.vision_max_n_token,
        "prompt_tokens": len(prepared.ids), "images": inputs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local pinned config.json and tokenizer.json")
    parser.add_argument("--shard", type=Path, required=True, help="Pinned vision-only model-00001-of-00048.safetensors")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--image-token-cap", type=int, help="Override the image resize budget for a smaller CPU probe")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Optionally expand the stored BF16 weights to FP32 for CPU arithmetic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("Threads must be positive")
        checkpoint = Checkpoint.open(args.checkpoint, metadata_only=True)
        result = probe(checkpoint, args.shard, args.image, args.threads, args.image_token_cap, args.dtype)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

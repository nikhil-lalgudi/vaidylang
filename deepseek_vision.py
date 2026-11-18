"""DeepSeek image preparation and reduced CPU vision execution."""

from dataclasses import dataclass
import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from third_party.deepseek_v41.encoding import IMAGE_PLACEHOLDER


def _vision_dependencies():
    try:
        import torch
        from third_party.deepseek_v41 import image_processor, vision
    except ImportError as error:
        raise ValueError("Install PyTorch and Pillow to use DeepSeek images") from error
    return torch, image_processor, vision


def vision_args(checkpoint):
    cfg = checkpoint.config.get("vision_config")
    if not isinstance(cfg, dict) or not cfg.get("num_hidden_layers"):
        raise ValueError("DeepSeek image prompts require a checkpoint with a vision tower")
    fields = {
        "vision_n_layers": "num_hidden_layers", "vision_dim": "hidden_size",
        "vision_n_heads": "num_attention_heads", "vision_inter_dim": "intermediate_size",
        "vision_patch_size": "patch_size", "vision_downsample_ratio": "downsample_ratio",
        "vision_max_n_token": "max_image_tokens", "vision_min_pixels": "min_pixels",
        "vision_rope_theta": "rope_theta",
    }
    values = {name: cfg.get(key) for name, key in fields.items()}
    if (
        any(type(n) is not int or n <= 0 for n in values.values())
        or values["vision_dim"] % (4 * values["vision_n_heads"])
        or values["vision_max_n_token"] < 4
    ):
        raise ValueError("Invalid DeepSeek vision dimensions or image limits")
    ratio = cfg.get("max_wh_ratio")
    if ratio is not None and (type(ratio) not in (int, float) or not 1 <= ratio < float("inf")):
        raise ValueError("Invalid DeepSeek vision max_wh_ratio")
    image_id = checkpoint.config.get("image_token_id")
    if type(image_id) is not int or not 0 <= image_id < checkpoint.text_config["vocab_size"]:
        raise ValueError("Invalid DeepSeek image token ID")
    return SimpleNamespace(
        **values, vision_max_wh_ratio=ratio, image_token_id=image_id,
        vision_enabled=True, dim=checkpoint.text_config["hidden_size"],
    )


def _image_bytes(record, base_directory):
    source = record.get("source")
    data = record.get("data")
    if data is None and isinstance(source, dict):
        data = source.get("data")
    if data is not None:
        if isinstance(data, bytes):
            return data
        if not isinstance(data, str):
            raise ValueError("Image data must be base64 text")
        return base64.b64decode(data, validate=True)

    url = record.get("url")
    if url is None and isinstance(source, dict):
        url = source.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Image requires a local file or base64 data")
    if url.startswith("data:"):
        header, separator, payload = url.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("Image data URL must use base64")
        return base64.b64decode(payload, validate=True)
    if "://" in url:
        raise ValueError("Use a local image file or base64 data")
    return (base_directory / Path(url)).read_bytes()


@dataclass(frozen=True)
class ImagePrompt:
    ids: list
    token_types: list
    images: list
    metadata: list


def prepare_image_prompt(checkpoint, tokenizer, ids, records, base_directory):
    _, processor, _ = _vision_dependencies()
    args = vision_args(checkpoint)
    if tokenizer.backend.token_to_id(IMAGE_PLACEHOLDER) != args.image_token_id:
        raise ValueError("Tokenizer image placeholder does not match the checkpoint image token ID")
    if ids.count(args.image_token_id) != len(records):
        raise ValueError("Prompt image tokens must match the image content blocks")
    expanded, token_types, images, metadata = [], [], [], []
    records = iter(records)
    for token in ids:
        if token != args.image_token_id:
            expanded.append(token)
            token_types.append(processor.TEXT)
            continue
        raw = _image_bytes(next(records), base_directory)
        patches, vit_h, vit_w, llm_h, llm_w = processor.load_image({"data": raw}, args)
        types = processor.image_token_types(llm_h, llm_w)
        start = len(expanded)
        images.append(processor.ImageInput(start, patches, vit_h, vit_w, types))
        expanded.extend([token] * types.numel())
        token_types.extend(types.tolist())
        metadata.append({
            "start": start, "tokens": types.numel(),
            "vit_grid": [vit_h, vit_w], "llm_grid": [llm_h, llm_w],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "bf16_patches_sha256": bf16_digest(patches),
        })
    return ImagePrompt(expanded, token_types, images, metadata)


def vision_modules(args):
    torch, _, vision = _vision_dependencies()
    with torch.device("meta"):
        return {"vision": vision.ViT(args), "aligner": vision.Aligner(args)}


def bf16_digest(tensor):
    torch, _, _ = _vision_dependencies()
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Expected a BF16 vision tensor")
    storage = tensor.cpu().contiguous().view(torch.int16).numpy().astype("<i2", copy=False)
    return hashlib.sha256(storage.tobytes()).hexdigest()


def encode_image_rows(modules, image):
    torch, _, _ = _vision_dependencies()
    with torch.inference_mode():
        patch_weight = modules["vision"].patch_embed.proj.weight
        patches = image.patches.to(dtype=patch_weight.dtype, device=patch_weight.device)
        grid = image.n_vit_h, image.n_vit_w
        hidden = modules["vision"](patches, *grid)
        return modules["aligner"](hidden, *grid)


def vision_weight_shapes(checkpoint):
    args = vision_args(checkpoint)
    shapes = {
        f"{name}.{key}": tuple(weight.shape)
        for name, module in vision_modules(args).items()
        for key, weight in module.state_dict().items()
    }
    shapes.update({name: (args.dim,) for name in ("image_start", "image_end", "image_newline")})
    text = checkpoint.text_config
    shapes.update({
        f"layers.{layer}.ffn.gate.bias_vl": (text["n_routed_experts"],)
        for layer in range(text["num_hidden_layers"])
    })
    return shapes


class VisionTower:
    def __init__(self, checkpoint, weights):
        torch, _, _ = _vision_dependencies()
        args = vision_args(checkpoint)
        self.image_token_id = args.image_token_id
        modules = vision_modules(args)
        for name, module in modules.items():
            module.load_state_dict({
                key: torch.from_numpy(weights[f"{name}.{key}"].copy())
                for key in module.state_dict()
            }, assign=True)
            module.eval()
        self.modules = modules
        self.weights = weights

    def embeddings(self, image):
        _, processor, _ = _vision_dependencies()
        types = image.types.cpu().numpy()
        if types.ndim != 1 or not np.isin(types, range(4)).all():
            raise ValueError("Invalid DeepSeek image token types")
        rows = encode_image_rows(self.modules, image).float().cpu().numpy()
        if (types == processor.IMAGE).sum() != len(rows):
            raise ValueError("Image token grid does not match the aligner output")
        result = self.weights["image_start"][None, :].repeat(types.size, axis=0)
        result[types == processor.IMAGE] = rows
        result[types == processor.IMAGE_END] = self.weights["image_end"]
        result[types == processor.IMAGE_NEW_LINE] = self.weights["image_newline"]
        return result

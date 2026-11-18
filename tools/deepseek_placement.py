"""Summarize rank allocation from public checkpoint headers, with no tensor downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import Checkpoint
from deepseek_config import DeepSeekConfig, is_vision_tensor, weight_shapes
from deepseek_placement import placement_report
from deepseek_quant import quantized_shapes
from deepseek_weights import quant_layout, scale_name


def validate_layout(checkpoint, headers):
    """Cross-check stored shapes against the loader's logical config map."""
    config = DeepSeekConfig.from_mapping(checkpoint.config)
    shapes = weight_shapes(config)
    expected_names = set()

    def expect(name, dtypes, shape):
        expected_names.add(name)
        shard = checkpoint.weight_map.get(name)
        if shard is None or name not in headers[shard]:
            raise ValueError(f"Missing required tensor {name} from the published index or header")
        row = headers[shard][name]
        if row["dtype"] not in dtypes or tuple(row["shape"]) != shape:
            raise ValueError(f"{name}: expected {dtypes} {shape}, found {row['dtype']} {tuple(row['shape'])}")

    for name, shape in shapes.items():
        layout = quant_layout(name) if config.quantization_config else None
        if layout is None:
            expect(name, ("BF16", "F16", "F32"), shape)
        else:
            encoding, block_shape = layout
            stored, scales = quantized_shapes(shape, block_shape, encoding)
            expect(name, ("I8",) if encoding == "fp4" else ("F8_E4M3",), stored)
            expect(scale_name(name), ("F8_E8M0",), scales)

    additional_names = checkpoint.weight_map.keys() - expected_names
    for name in sorted(additional_names):
        if not is_vision_tensor(name):
            raise ValueError(f"Unrecognized non-vision tensor {name} in the checkpoint index")
    return {
        "logical_text_tensor_count": len(shapes),
        "validated_text_storage_tensor_count": len(expected_names),
        "vision_tensor_count": len(additional_names),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory containing config and index")
    parser.add_argument("--headers", type=Path, required=True, help="Directory built by tools/deepseek_headers.py")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--capacity", type=int, help="Defaults to the published context limit")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-tokens", type=int, default=1, help="Chunk length for scratch memory estimates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        checkpoint = Checkpoint.open(args.checkpoint, metadata_only=True)
        if checkpoint.model_type != "deepseek_v41" or checkpoint.weight_map is None:
            raise ValueError("Expected a sharded DeepSeek V4.1 checkpoint with its published index")
        headers, hashes = {}, {}
        for shard in checkpoint.shards:
            contents = (args.headers / f"{shard}.header.json").read_bytes()
            hashes[shard] = hashlib.sha256(contents).hexdigest()
            headers[shard] = json.loads(contents)
        layout = validate_layout(checkpoint, headers)
        cases = [
            placement_report(
                checkpoint, headers, size, capacity=args.capacity, batch_size=args.batch_size,
                query_tokens=args.query_tokens,
            )
            for size in dict.fromkeys(args.world_sizes)
        ]
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1

    rank_dependent = ("world_size", "ranks", "per_rank_scratch_bytes", "example_placements")
    output = {
        "config_sha256": hashlib.sha256((checkpoint.directory / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((checkpoint.directory / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "shard_header_sha256": hashes,
        "verified_layout": layout,
        **{key: value for key, value in cases[0].items() if key not in rank_dependent},
        "placements": [{key: case[key] for key in rank_dependent} for case in cases],
    }
    encoded = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

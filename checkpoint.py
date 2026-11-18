"""Read local Hugging Face Safetensors checkpoints independently of model execution."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    directory: Path
    config: dict
    weight_map: dict[str, str] | None
    shards: tuple[str, ...]
    total_tensor_bytes: int | None

    @classmethod
    def open(cls, directory, *, metadata_only=False):
        """Resolve local config and shard index. Open individual tensors on demand."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if not isinstance(config, dict) or not isinstance(config.get("model_type"), str) or not config["model_type"]:
            raise ValueError("Checkpoint config requires a model_type")
        text_config = config.get("text_config", config)
        if not isinstance(text_config, dict):
            raise ValueError("Checkpoint text_config must be an object")
        if config["model_type"] == "deepseek_v41" and text_config.get("model_type") != "deepseek_v41_text":
            raise ValueError("DeepSeek V4.1 Flash requires its nested deepseek_v41_text config")

        weight_map = None
        shards = ("model.safetensors",)
        total_tensor_bytes = None
        index_path = directory / "model.safetensors.index.json"
        if not (directory / shards[0]).is_file() and index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map or not all(weight_map):
                raise ValueError("Safetensors index requires a nonempty weight_map")
            for shard in weight_map.values():
                if not isinstance(shard, str) or "/" in shard or "\\" in shard or not shard.endswith(".safetensors"):
                    raise ValueError(f"Invalid shard filename {shard!r}")
            shards = tuple(sorted(set(weight_map.values())))
            metadata = index.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("Safetensors index metadata must be an object")
            total_tensor_bytes = metadata.get("total_size")
            if total_tensor_bytes is not None and (type(total_tensor_bytes) is not int or total_tensor_bytes < 0):
                raise ValueError("Safetensors index total_size must be a nonnegative byte count")
        checkpoint = cls(directory, config, weight_map, shards, total_tensor_bytes)
        if not metadata_only:
            missing = checkpoint.missing_shards
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} checkpoint shard(s), including {directory / missing[0]}")
        return checkpoint

    @property
    def model_type(self):
        return self.config["model_type"]

    @property
    def text_config(self):
        return self.config.get("text_config", self.config)

    @property
    def missing_shards(self):
        return tuple(shard for shard in self.shards if not (self.directory / shard).is_file())

    def shard_for(self, tensor_name):
        if self.weight_map is None:
            return self.directory / self.shards[0]
        try:
            return self.directory / self.weight_map[tensor_name]
        except KeyError as error:
            raise ValueError(f"Tensor {tensor_name!r} is absent from the checkpoint index") from error

    def group_tensors(self, names):
        grouped = {}
        for name in names:
            grouped.setdefault(self.shard_for(name), []).append(name)
        return grouped

    def read_tensor(self, tensor_name, *, device="cpu"):
        """Load one Torch tensor with its stored dtype, including packed weight bytes."""
        try:
            from safetensors import safe_open

            with safe_open(self.shard_for(tensor_name), framework="pt", device=str(device)) as shard:
                return shard.get_tensor(tensor_name)
        except ImportError as error:
            raise ValueError("Install PyTorch and safetensors to read checkpoint tensors") from error

    def read_storage(self, names):
        """Yield (name, Safetensors dtype code, NumPy storage) with no float8 dependency."""
        import numpy as np

        types = {
            "F32": "<f4", "F16": "<f2", "BF16": "<u2",
            "F8_E4M3": "u1", "F8_E8M0": "u1", "I8": "i1",
        }
        for shard, grouped in self.group_tensors(names).items():
            with shard.open("rb") as stream:
                header_length = int.from_bytes(stream.read(8), "little")
                size = shard.stat().st_size
                if not 0 < header_length <= min(100_000_000, size - 8):
                    raise ValueError(f"Invalid Safetensors header length in {shard.name}")
                header = json.loads(stream.read(header_length))
            if not isinstance(header, dict):
                raise ValueError(f"Invalid Safetensors tensor metadata in {shard.name}")
            data_start = 8 + header_length
            stored = np.memmap(shard, dtype=np.uint8, mode="r")
            for name in grouped:
                try:
                    info = header[name]
                    code = info["dtype"]
                    shape = info["shape"]
                    start, end = info["data_offsets"]
                    dtype = np.dtype(types[code])
                    if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                        raise ValueError("invalid shape")
                    if (
                        type(start) is not int or type(end) is not int
                        or not 0 <= start <= end <= size - data_start
                        or end - start != math.prod(shape) * dtype.itemsize
                    ):
                        raise ValueError("invalid data offsets")
                except (TypeError, KeyError, ValueError) as error:
                    raise ValueError(f"Missing or incompatible tensor {name} in {shard.name}: {error}") from error
                view = stored[data_start + start:data_start + end].view(dtype).reshape(shape)
                yield name, code, view

    def describe(self):
        description = {
            "model_type": self.model_type,
            "architectures": self.config.get("architectures", []),
            "vocab_size": self.text_config.get("vocab_size"),
            "max_position_embeddings": self.text_config.get("max_position_embeddings"),
            "quantization_config": self.config.get("quantization_config"),
            "shard_count": len(self.shards),
            "indexed_tensor_count": len(self.weight_map) if self.weight_map is not None else None,
            "indexed_tensor_bytes": self.total_tensor_bytes,
            "missing_shards": self.missing_shards,
        }
        if self.model_type == "deepseek_v41":
            from deepseek_config import is_vision_tensor

            description["indexed_vision_tensor_count"] = (
                sum(map(is_vision_tensor, self.weight_map)) if self.weight_map is not None else None
            )
        return description

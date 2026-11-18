"""Run the pinned DeepSeek CUDA Transformer with the CPU workload's exact inputs."""

from contextlib import contextmanager
import hashlib
import json
import os
from types import SimpleNamespace

from tools.deepseek_upstream_checks import INPUT_SHA256, REVISION, load_module


_SOURCE_HASHES = {
    "engram.py": INPUT_SHA256["inference/engram.py"],
    "image_processor.py": INPUT_SHA256["inference/image_processor.py"],
    "vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "config.json": INPUT_SHA256["inference/config.json"],
}


class UpstreamEngine:
    def __init__(self, model, vocab_size, device):
        self.model = model
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.device = device

    def _forward(self, ids, position, **kwargs):
        import torch

        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, logits, _ = self.model.forward(inputs, position, **kwargs)
        if logits.shape != (1, self.config.vocab_size):
            raise ValueError("Unexpected shape from the upstream Transformer")
        return logits[0].float().cpu().numpy()

    def prefill(self, ids, capacity, *, images=None):
        if not 0 < len(ids) <= capacity <= self.model.max_seq_len:
            raise ValueError("Prompt and output budget exceed the upstream cache limit")
        kwargs = {}
        if images is not None:
            import torch

            kwargs = {
                "images": [images.images],
                "token_types": torch.tensor([images.token_types], dtype=torch.long, device=self.device),
            }
        return self._forward(ids, 0, **kwargs), SimpleNamespace(position=len(ids))

    def decode(self, token, state):
        logits = self._forward([token], state.position)
        state.position += 1
        return logits


@contextmanager
def open_upstream_engine(source, converted, checkpoint, tokenizer, capacity):
    import torch
    from safetensors.torch import load_model

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        actual = hashlib.sha256((checkpoint.directory / name).read_bytes()).hexdigest()
        if actual != INPUT_SHA256[name]:
            raise ValueError(f"Upstream capture requires {name} from revision {REVISION}")
    inputs = {name: (source / name).read_bytes() for name in _SOURCE_HASHES}
    for name, expected in _SOURCE_HASHES.items():
        if hashlib.sha256(inputs[name]).hexdigest() != expected:
            raise ValueError(f"Upstream {name} does not match revision {REVISION}")
    if not torch.cuda.is_available():
        raise ValueError("The upstream runner requires CUDA and the pinned TileLang dependencies")
    from transformers import AutoTokenizer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    shard = converted / f"model{rank}-mp{world_size}.safetensors"
    if not shard.is_file():
        raise FileNotFoundError(f"Missing upstream converted shard: {shard}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
    try:
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)
        torch.manual_seed(33377335)
        for name in _SOURCE_HASHES:
            if name.endswith(".py"):
                model_module = load_module(name[:-3], source / name, inputs[name])
        args = model_module.ModelArgs(**json.loads(inputs["config.json"]))
        args.max_batch_size, args.max_seq_len, args.temperature = 1, capacity, 0
        hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint.directory, local_files_only=True)
        if hf_tokenizer.get_vocab() != tokenizer.backend.get_vocab():
            raise ValueError("Upstream tokenizer and workload tokenizer have different token IDs")
        with torch.device(device):
            model = model_module.Transformer(args, hf_tokenizer)
        load_model(model, str(shard))
        model.eval()
        torch.set_default_device(device)
        yield UpstreamEngine(model, args.vocab_size, device)
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
        if world_size > 1:
            torch.distributed.destroy_process_group()

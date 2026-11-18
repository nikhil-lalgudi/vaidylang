"""Local tokenizer loading and prompt formatting for the CPU harness."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from third_party.deepseek_v41 import encoding


@dataclass(frozen=True)
class PromptTokenizer:
    backend: object
    sha256: str

    @classmethod
    def from_checkpoint(cls, checkpoint):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise ValueError("Install tokenizers to use text or messages") from error

        path = checkpoint.directory / "tokenizer.json"
        data = path.read_bytes()
        try:
            backend = Tokenizer.from_buffer(data)
        except Exception as error:
            raise ValueError(f"Could not load tokenizer from {path}: {error}") from error
        ids = set(backend.get_vocab().values())
        vocab_size = checkpoint.text_config.get("vocab_size")
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("Checkpoint requires a positive vocabulary size")
        # Published Qwen3 pads its embedding and head rows above the tokenizer's IDs.
        if (not ids or ids != set(range(len(ids))) or len(ids) > vocab_size
                or (checkpoint.model_type == "deepseek_v41" and len(ids) != vocab_size)):
            raise ValueError(f"Tokenizer vocabulary does not match checkpoint vocabulary size {vocab_size}")
        backend.no_padding()
        backend.no_truncation()
        return cls(backend, hashlib.sha256(data).hexdigest())

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode([i for i in ids if self.backend.id_to_token(i) is not None],
                                   skip_special_tokens=True)


def read_messages(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_reasoning_effort(value):
    if value in encoding.REASONING_EFFORT_MAPPINGS:
        return value
    return int(value)


def deepseek_prompt(messages, *, thinking=False, reasoning_effort=None):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a nonempty JSON array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in (
            "system", "user", "assistant", "tool", "latest_reminder",
        ):
            raise ValueError(f"Invalid role in message {index + 1}")
        if not isinstance(message.get("content"), (str, list, type(None))):
            raise ValueError(f"Invalid content in message {index + 1}")
    if reasoning_effort is not None:
        if not thinking:
            raise ValueError("Reasoning effort requires --thinking")
        if type(reasoning_effort) is str and reasoning_effort in encoding.REASONING_EFFORT_MAPPINGS:
            reasoning_effort = encoding.REASONING_EFFORT_MAPPINGS[reasoning_effort]
        if type(reasoning_effort) is not int or not 1 <= reasoning_effort <= 100:
            raise ValueError("Reasoning effort must be 1 to 100 or one of low, high or max")
    try:
        prompt, media = encoding.encode_messages(
            messages, thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=reasoning_effort, return_multi_modal_data=True,
        )
    except (AssertionError, AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid DeepSeek messages: {error or 'unsupported message structure'}") from error
    return prompt, media["images"]


def eos_ids(checkpoint):
    value = checkpoint.config.get("eos_token_id", checkpoint.text_config.get("eos_token_id"))
    if value is None:
        return frozenset()
    ids = value if isinstance(value, list) else [value]
    if any(type(i) is not int or not 0 <= i < checkpoint.text_config["vocab_size"] for i in ids):
        raise ValueError("Checkpoint has invalid EOS token IDs")
    return frozenset(ids)


def prepare_prompt(checkpoint, *, text=None, messages=None, thinking=False,
                   reasoning_effort=None, base_directory=None, tokenizer=None):
    media = []
    if checkpoint.model_type == "deepseek_v41":
        if messages is None:
            if not isinstance(text, str):
                raise ValueError("Expected prompt text or structured messages")
            messages = [{"role": "user", "content": encoding.parse_tagged_text(text)}]
        text, media = deepseek_prompt(messages, thinking=thinking, reasoning_effort=reasoning_effort)
    elif messages is not None or thinking or reasoning_effort is not None:
        raise ValueError("Structured messages and thinking options require a DeepSeek V4.1 checkpoint")

    tokenizer = tokenizer or PromptTokenizer.from_checkpoint(checkpoint)
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError("Prompt must encode to at least one token")
    image_prompt = None
    if media:
        from deepseek_vision import prepare_image_prompt

        image_prompt = prepare_image_prompt(
            checkpoint, tokenizer, ids, media, base_directory or Path.cwd(),
        )
        ids = image_prompt.ids
    elif checkpoint.config.get("image_token_id") in ids:
        raise ValueError("Prompt image tokens require matching image content blocks")
    return ids, tokenizer, image_prompt

"""Python adapter for the C++ CPU inference library."""

import ctypes as ct
from functools import cache
import os
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace
from weakref import finalize

import numpy as np


@cache
def load_library():
    """Resolve the local CMake build or an explicitly configured library."""
    override = os.environ.get("VAIDYLANG_CPU_LIBRARY")
    if override == "off":
        return None
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    path = Path(override) if override else Path(__file__).resolve().parent / "build" / f"libvaidylang_native{suffix}"
    if not override and not path.is_file():
        return None
    library = ct.CDLL(str(path.resolve()))
    library.vaidy_last_error.argtypes = []
    library.vaidy_last_error.restype = ct.c_char_p
    return library


@cache
def _model_functions(library, architecture):
    signatures = {
        "load": ([ct.c_char_p], ct.c_void_p),
        "free": ([ct.c_void_p], None),
        "request": ([ct.c_void_p, ct.c_int], ct.c_void_p),
        "request_free": ([ct.c_void_p], None),
        "length": ([ct.c_void_p], ct.c_int),
        "append": (
            [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t], ct.c_bool,
        ),
    }
    functions = {}
    for name, (arguments, result) in signatures.items():
        try:
            function = getattr(library, f"vaidy_{architecture}_{name}")
        except AttributeError as error:
            raise OSError(f"Rebuild the C++ CPU inference library to enable {architecture}") from error
        function.argtypes = arguments
        function.restype = result
        functions[name] = function
    return SimpleNamespace(**functions)


def check(library, result):
    if not result:
        raise ValueError(library.vaidy_last_error().decode(errors="replace"))
    return result


class NativeRequest:
    def __init__(self, library, functions, model, capacity):
        self._functions = functions
        self._pointer = check(library, functions.request(model, capacity))
        self._release = finalize(self, functions.request_free, self._pointer)
        self._lock = Lock()

    @property
    def length(self):
        with self._lock:
            return self._functions.length(self._pointer)


class _NativeEngine:
    cpu_backend = "native"

    def __init__(self, checkpoint, config, library, architecture, output_vocab):
        self.config = config
        self._library = library
        self._functions = _model_functions(library, architecture)
        self._model = check(library, self._functions.load(os.fsencode(checkpoint.directory)))
        self._release = finalize(self, self._functions.free, self._model)
        self._output_vocab = output_vocab

    def prefill(self, token_ids, capacity):
        if not isinstance(capacity, (int, np.integer)) or not 1 <= capacity <= self.config.max_position_embeddings:
            raise ValueError("KV capacity is outside the model context limit")
        state = NativeRequest(self._library, self._functions, self._model, int(capacity))
        return self._append(token_ids, state), state

    def decode(self, token_id, state):
        return self._append([token_id], state)

    def _append(self, token_ids, state):
        if not isinstance(state, NativeRequest) or state._functions is not self._functions:
            raise ValueError("Request belongs to a different model architecture")
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty one-dimensional sequence of token IDs")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("Token ID is outside the vocabulary")
        ids = np.ascontiguousarray(ids, dtype=np.intc)
        logits = np.empty(self._output_vocab, dtype=np.float32)
        with state._lock:
            check(self._library, self._functions.append(
                self._model, state._pointer, ids.ctypes.data, ids.size, logits.ctypes.data, logits.size,
            ))
        return logits


class NativeQwen3Engine(_NativeEngine):
    def __init__(self, checkpoint, config, library):
        super().__init__(checkpoint, config, library, "qwen3", config.vocab_size)


class NativeInklingEngine(_NativeEngine):
    def __init__(self, checkpoint, library):
        text = checkpoint.text_config
        config = SimpleNamespace(vocab_size=text["vocab_size"], max_position_embeddings=checkpoint.context_limit)
        output_vocab = text.get("unpadded_vocab_size")
        super().__init__(checkpoint, config, library, "inkling", config.vocab_size if output_vocab is None else output_vocab)

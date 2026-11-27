"""Python adapter for the C++ CPU inference library."""

import ctypes as ct
from functools import cache
import os
from pathlib import Path
import sys
from threading import Lock
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
    signatures = {
        "vaidy_last_error": ([], ct.c_char_p),
        "vaidy_qwen3_load": ([ct.c_char_p], ct.c_void_p),
        "vaidy_qwen3_free": ([ct.c_void_p], None),
        "vaidy_qwen3_request": ([ct.c_void_p, ct.c_int], ct.c_void_p),
        "vaidy_qwen3_request_free": ([ct.c_void_p], None),
        "vaidy_qwen3_length": ([ct.c_void_p], ct.c_int),
        "vaidy_qwen3_append": (
            [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t], ct.c_bool,
        ),
    }
    for name, (arguments, result) in signatures.items():
        try:
            function = getattr(library, name)
        except AttributeError as error:
            raise OSError(f"Incompatible C++ CPU inference library at {path}") from error
        function.argtypes = arguments
        function.restype = result
    return library


def check(library, result):
    if not result:
        raise ValueError(library.vaidy_last_error().decode(errors="replace"))
    return result


class NativeRequest:
    def __init__(self, library, model, capacity):
        self._library = library
        self._pointer = check(library, library.vaidy_qwen3_request(model, capacity))
        self._release = finalize(self, library.vaidy_qwen3_request_free, self._pointer)
        self._lock = Lock()

    @property
    def length(self):
        with self._lock:
            return self._library.vaidy_qwen3_length(self._pointer)


class NativeQwen3Engine:
    cpu_backend = "native"

    def __init__(self, checkpoint, config, library):
        self.config = config
        self._library = library
        self._model = check(library, library.vaidy_qwen3_load(os.fsencode(checkpoint.directory)))
        self._release = finalize(self, library.vaidy_qwen3_free, self._model)

    def prefill(self, token_ids, capacity):
        if not isinstance(capacity, (int, np.integer)) or not 1 <= capacity <= self.config.max_position_embeddings:
            raise ValueError("KV capacity is outside the model context limit")
        state = NativeRequest(self._library, self._model, int(capacity))
        return self._append(token_ids, state), state

    def decode(self, token_id, state):
        return self._append([token_id], state)

    def _append(self, token_ids, state):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty one-dimensional sequence of token IDs")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("Token ID is outside the vocabulary")
        ids = np.ascontiguousarray(ids, dtype=np.intc)
        logits = np.empty(self.config.vocab_size, dtype=np.float32)
        with state._lock:
            check(self._library, self._library.vaidy_qwen3_append(
                self._model, state._pointer, ids.ctypes.data, ids.size, logits.ctypes.data, logits.size,
            ))
        return logits

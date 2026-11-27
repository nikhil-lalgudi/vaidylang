from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tokenizers import Tokenizer, models, pre_tokenizers

from engine import tiny_weights
from native_cpu import load_library
from serve import create_server
from test_engine import write_checkpoint_weights
from test_inkling import inkling_fixture


def write_tokenizer(path, vocab_size):
    vocab = {f"t{token}": token for token in range(vocab_size)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "chat_template.jinja").write_text("{{ messages[-1].content }}")


@contextmanager
def running_server(checkpoint, name, backend):
    with create_server(checkpoint, name=name, port=0, cpu_backend=backend) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join()


def chat(url, model, **options):
    payload = {"model": model, "messages": [{"role": "user", "content": "t2 t11 t3 t7 t9"}],
               "max_tokens": 3, **options}
    request = Request(f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


class ServingTests(unittest.TestCase):
    def test_qwen_server_lists_model_and_returns_independent_completions(self):
        with TemporaryDirectory() as location:
            path = Path(location)
            config, weights = tiny_weights()
            (path / "config.json").write_text(json.dumps({**asdict(config), "model_type": "qwen3", "hidden_act": "silu"}))
            write_checkpoint_weights(path / "model.safetensors", {name: ("F32", value) for name, value in weights.items()})
            write_tokenizer(path, config.vocab_size)
            with running_server(path, "qwen", "python") as url:
                with urlopen(f"{url}/v1/models", timeout=5) as response:
                    self.assertEqual(json.load(response)["data"][0]["id"], "qwen")
                first = chat(url, "qwen")
                with self.assertRaises(HTTPError) as wrong_model:
                    chat(url, "missing")
                self.assertEqual(wrong_model.exception.code, 404)
                with self.assertRaises(HTTPError) as oversized:
                    chat(url, "qwen", max_tokens=config.max_position_embeddings)
                self.assertEqual(oversized.exception.code, 400)
                second = chat(url, "qwen")
            self.assertEqual(first["choices"], second["choices"])
            self.assertEqual(first["choices"][0]["message"]["role"], "assistant")
            self.assertEqual(first["usage"], {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})

    @unittest.skipUnless(hasattr(load_library(), "vaidy_inkling_load"), "requires a C++ build with Inkling support")
    def test_native_inkling_server_uses_the_real_decoder(self):
        with TemporaryDirectory() as location:
            path = Path(location)
            inkling_fixture(path)
            write_tokenizer(path, 29)
            with running_server(path, "inkling", "native") as url:
                response = chat(url, "inkling", max_completion_tokens=1, max_tokens=None)
                self.assertEqual(response["choices"][0]["message"]["content"], "t0")
                self.assertEqual(response["choices"][0]["finish_reason"], "length")

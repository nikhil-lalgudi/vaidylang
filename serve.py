"""Serve one local VaidyLang checkpoint over HTTP."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import time
from uuid import uuid4

from checkpoint import Checkpoint
from engine import load_engine
from harness import greedy_ids
from prompt_input import PromptTokenizer, deepseek_prompt, eos_ids


class RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def chat_renderer(checkpoint):
    if checkpoint.model_type == "deepseek_v41":
        return lambda messages: deepseek_prompt(messages)[0]

    try:
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:
        raise ValueError("Install Jinja2 to render the checkpoint's chat template") from error

    template_path = checkpoint.directory / "chat_template.jinja"
    if template_path.is_file():
        source = template_path.read_text(encoding="utf-8")
    else:
        tokenizer_path = checkpoint.directory / "tokenizer_config.json"
        if not tokenizer_path.is_file():
            raise ValueError("Chat serving requires chat_template.jinja or a tokenizer_config.json chat template")
        config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        source = config.get("chat_template") if isinstance(config, dict) else None
        if isinstance(source, list):
            source = next((entry.get("template") for entry in source
                           if isinstance(entry, dict) and entry.get("name") == "default"), None)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Checkpoint requires a default chat template")

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = json.dumps
    environment.globals["raise_exception"] = raise_exception
    try:
        template = environment.from_string(source)
    except TemplateError as error:
        raise ValueError(f"Invalid chat template: {error}") from error

    def render(messages):
        try:
            return template.render(messages=messages, add_generation_prompt=True, tools=None,
                                   enable_thinking=False, reasoning_effort="none")
        except TemplateError as error:
            raise ValueError(f"Chat template rejected the messages: {error}") from error

    return render


class ChatModel:
    def __init__(self, directory, name, cpu_backend, context_limit):
        checkpoint = Checkpoint.open(directory)
        self.name = checkpoint.directory.name if name is None else name
        if not self.name.strip():
            raise ValueError("Use a nonempty model name")
        self.created = int(time())
        model_context = checkpoint.context_limit
        if type(model_context) is not int or model_context < 2:
            raise ValueError("Checkpoint must have a context limit of at least two tokens")
        self.context_limit = min(model_context, 4096) if context_limit is None else context_limit
        if type(self.context_limit) is not int or not 2 <= self.context_limit <= model_context:
            raise ValueError(f"--max-context must be between 2 and {model_context}")
        self.tokenizer = PromptTokenizer.from_checkpoint(checkpoint)
        self.render = chat_renderer(checkpoint)
        self.stop_ids = eos_ids(checkpoint)
        self.engine = load_engine(directory, cpu_backend=cpu_backend)

    def complete(self, body):
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise ValueError("Request requires a model name")
        if body["model"] != self.name:
            raise RequestError(404, f"Model {body['model']!r} is unavailable")
        supported = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream", "n"}
        if unknown := sorted(body.keys() - supported):
            raise ValueError(f"Unsupported chat fields: {', '.join(unknown)}")
        if (body.get("stream") not in (None, False) or body.get("temperature") not in (None, 0)
                or body.get("n") not in (None, 1)):
            raise ValueError("Use one non-streaming completion with greedy decoding")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str) for message in messages
        ):
            raise ValueError("Use a nonempty array of system, user or assistant messages with string content")
        messages = [{"role": message["role"], "content": message["content"]} for message in messages]

        budgets = [body[key] for key in ("max_tokens", "max_completion_tokens") if body.get(key) is not None]
        if len(budgets) > 1 or any(type(budget) is not int or budget < 1 for budget in budgets):
            raise ValueError("Specify at most one positive integer token limit")
        prompt = self.tokenizer.encode(self.render(messages))
        if not prompt:
            raise ValueError("Prompt must encode to at least one token")
        available = self.context_limit - len(prompt)
        budget = budgets[0] if budgets else min(128, available)
        if not 1 <= budget <= available:
            raise ValueError(f"Prompt plus completion exceeds the {self.context_limit}-token context limit")

        tokens = greedy_ids(self.engine, prompt, budget, self.stop_ids)
        content = self.tokenizer.decode([token for token in tokens if token not in self.stop_ids])
        return {
            "id": f"chatcmpl-{uuid4().hex}", "object": "chat.completion", "created": int(time()),
            "model": self.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop" if tokens[-1] in self.stop_ids else "length"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens),
                      "total_tokens": len(prompt) + len(tokens)},
        }


class ChatHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(30)
        super().setup()

    def send_json(self, status, value):
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_request_error(self, status, message):
        error_type = "server_error" if status >= 500 else "invalid_request_error"
        self.send_json(status, {"error": {"type": error_type, "message": message}})

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path == "/health":
            self.send_json(200, {"status": "ok"})
        elif path == "/v1/models":
            model = self.server.model
            self.send_json(200, {"object": "list", "data": [
                {"id": model.name, "object": "model", "created": model.created, "owned_by": "vaidylang"},
            ]})
        else:
            self.send_request_error(404, "Endpoint not found")

    def read_json(self):
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or self.headers.get("Transfer-Encoding")
                or not lengths[0].isascii() or not lengths[0].isdigit()):
            raise ValueError("Expected one Content-Length header for a JSON request")
        length = int(lengths[0])
        if length > 1_048_576:
            raise RequestError(413, "Request body exceeds 1 MiB")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("Incomplete request body")
        try:
            return json.loads(payload.decode("utf-8"))
        except (ValueError, RecursionError) as error:
            raise ValueError("Expected a valid UTF-8 JSON request") from error

    def do_POST(self):
        if self.path.partition("?")[0] != "/v1/chat/completions":
            self.send_request_error(404, "Endpoint not found")
            return
        try:
            response = self.server.model.complete(self.read_json())
        except RequestError as error:
            self.send_request_error(error.status, str(error))
        except ValueError as error:
            self.send_request_error(400, str(error))
        except TimeoutError:
            self.send_request_error(408, "Timed out reading the request body")
        else:
            self.send_json(200, response)


class ChatServer(HTTPServer):
    def __init__(self, address, model):
        self.model = model
        super().__init__(address, ChatHandler)


def create_server(checkpoint, *, name=None, port=8000, cpu_backend="auto", context_limit=None):
    if not 0 <= port <= 65535:
        raise ValueError("--port must be between 0 and 65535")
    model = ChatModel(checkpoint, name, cpu_backend, context_limit)
    return ChatServer(("127.0.0.1", port), model)

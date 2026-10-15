"""OllamaChat against a local HTTP server that speaks the /api/chat protocol."""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from cuad_doc_qa.generation import OllamaChat

SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}}, "required": ["found"]}
MESSAGES = [{"role": "user", "content": "Which law governs?"}]


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.delay = 0.0
        self.reply: dict[str, Any] = {
            "message": {"role": "assistant", "content": '{"found": true}'},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 5,
        }
        self.tags: dict[str, Any] = {
            "models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b", "digest": "845dbda0ea48"}]
        }


@pytest.fixture
def ollama() -> Iterator[tuple[FakeOllama, str]]:
    state = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            state.requests.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
            time.sleep(state.delay)
            body = (
                json.dumps(state.reply)
                if state.status == 200
                else json.dumps({"error": "model 'missing:7b' not found"})
            ).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps(state.tags).encode()
            self.send_response(200 if self.path == "/api/tags" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_sends_a_pinned_non_streaming_structured_request(ollama: tuple[FakeOllama, str]) -> None:
    state, url = ollama

    reply = OllamaChat("qwen2.5:7b", base_url=url + "/", num_ctx=4096).complete(MESSAGES, SCHEMA)

    assert reply == '{"found": true}'
    [request] = state.requests
    assert request["path"] == "/api/chat"
    assert request["body"] == {
        "model": "qwen2.5:7b",
        "messages": MESSAGES,
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 1024},
    }


def test_missing_model_error_names_the_pull_command(ollama: tuple[FakeOllama, str]) -> None:
    state, url = ollama
    state.status = 404

    with pytest.raises(RuntimeError, match=r"HTTP 404.*not found.*ollama pull missing:7b"):
        OllamaChat("missing:7b", base_url=url).complete(MESSAGES, SCHEMA)


def test_reply_cut_at_num_predict_is_an_error(ollama: tuple[FakeOllama, str]) -> None:
    state, url = ollama
    state.reply["done_reason"] = "length"

    with pytest.raises(RuntimeError, match="num_predict=1024"):
        OllamaChat("qwen2.5:7b", base_url=url).complete(MESSAGES, SCHEMA)


def test_unreachable_server_is_reported() -> None:
    with pytest.raises(RuntimeError, match="not reachable at http://127.0.0.1:9/api/chat"):
        OllamaChat("qwen2.5:7b", base_url="http://127.0.0.1:9", timeout=5).complete(
            MESSAGES, SCHEMA
        )


def test_slow_reply_names_the_timeout(ollama: tuple[FakeOllama, str]) -> None:
    state, url = ollama
    state.delay = 2.0

    with pytest.raises(RuntimeError, match=r"did not reply within 0.5s; raise the timeout"):
        OllamaChat("qwen2.5:7b", base_url=url, timeout=0.5).complete(MESSAGES, SCHEMA)


def test_digest_names_the_pulled_weights(ollama: tuple[FakeOllama, str]) -> None:
    _, url = ollama

    assert OllamaChat("qwen2.5:7b", base_url=url).digest() == "845dbda0ea48"
    assert OllamaChat("missing:7b", base_url=url).digest() is None


def test_options_are_the_pinned_decoding_settings() -> None:
    assert OllamaChat("m", num_ctx=4096, seed=7).options == {
        "temperature": 0,
        "seed": 7,
        "num_ctx": 4096,
        "num_predict": 1024,
    }


def test_digest_reports_an_unreachable_server() -> None:
    with pytest.raises(RuntimeError, match="not reachable at http://127.0.0.1:9"):
        OllamaChat("m", base_url="http://127.0.0.1:9").digest()

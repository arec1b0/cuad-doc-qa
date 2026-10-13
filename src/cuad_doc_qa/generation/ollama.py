"""Chat completion through a local Ollama server (POST /api/chat), standard library only.

Decoding is pinned for reproducibility: temperature 0 and a fixed seed. Output is constrained
by Ollama's structured outputs (`format` = JSON schema). `num_ctx` is always sent explicitly:
Ollama's default context is small and it truncates longer prompts silently.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        """The assistant's reply: a JSON document that follows `schema`."""
        ...


class OllamaChat:
    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        seed: int = 0,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self._base = base_url.rstrip("/")
        self._url = self._base + "/api/chat"
        self._options = {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        self._timeout = timeout

    @property
    def options(self) -> dict[str, int]:
        """Decoding options sent with every request."""
        return dict(self._options)

    def digest(self) -> str | None:
        """Digest of the local model weights (GET /api/tags), or None if the model is not pulled.

        Raises RuntimeError if Ollama is unreachable.
        """
        try:
            with urllib.request.urlopen(self._base + "/api/tags", timeout=30) as response:
                models = json.loads(response.read()).get("models", [])
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Ollama is not reachable at {self._base}: {error}") from error
        for entry in models:
            if self.model in (entry.get("name"), entry.get("model")):
                return entry.get("digest")
        return None

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        """Raises RuntimeError if Ollama is unreachable, rejects the request or cuts the reply."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": self._options,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            hint = f"; run: ollama pull {self.model}" if error.code == 404 else ""
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}{hint}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Ollama is not reachable at {self._url}: {error}") from error

        if payload.get("done_reason") == "length":
            raise RuntimeError(
                f"Ollama stopped at num_predict={self._options['num_predict']} tokens; "
                "the reply is incomplete"
            )
        logger.debug(
            "Ollama %s: %.1fs, prompt %s tokens, reply %s tokens",
            self.model,
            time.perf_counter() - started,
            payload.get("prompt_eval_count"),
            payload.get("eval_count"),
        )
        return payload["message"]["content"]

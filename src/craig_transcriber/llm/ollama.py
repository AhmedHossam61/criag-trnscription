"""A local model served by Ollama (https://ollama.com): free, private, no daily limit."""

from __future__ import annotations

import json
from typing import Iterator

from .base import LLMProvider, LLMUnavailable


def json_schema(schema: dict) -> dict:
    """Gemini's schema dialect (OBJECT, STRING, ...) as the JSON Schema Ollama takes."""
    out = {k: v for k, v in schema.items() if k not in ("type", "properties", "items")}
    out["type"] = schema["type"].lower()
    if "properties" in schema:
        out["properties"] = {k: json_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = json_schema(schema["items"])
    return out


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str = "qwen3:4b", url: str = "http://localhost:11434",
                 think: bool | None = None, timeout_s: float = 300.0):
        self.model = model
        self.url = url.rstrip("/")
        self.think = think
        self.timeout_s = timeout_s

    def describe(self) -> str:
        return f"{self.model} (local)"

    def _body(self, system: str, prompt: str, stream: bool, schema: dict | None,
              temperature: float) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if schema is not None:
            body["format"] = json_schema(schema)
        if self.think is not None:
            body["think"] = self.think
        return body

    def _error(self, exc: Exception) -> LLMUnavailable:
        import httpx
        if isinstance(exc, httpx.ConnectError):
            return LLMUnavailable(f"Ollama is not running at {self.url} (start it, or run `ollama serve`)")
        if isinstance(exc, httpx.HTTPStatusError):
            detail = exc.response.text[:300]
            if exc.response.status_code == 404:
                return LLMUnavailable(f"model {self.model} not found -- run `ollama pull {self.model}`")
            return LLMUnavailable(f"Ollama error {exc.response.status_code}: {detail}")
        return LLMUnavailable(f"{type(exc).__name__}: {str(exc)[:300]}")

    def complete(self, system: str, prompt: str, *, schema: dict | None = None,
                 temperature: float = 0.2) -> str:
        import httpx
        try:
            response = httpx.post(f"{self.url}/api/chat", timeout=self.timeout_s,
                                  json=self._body(system, prompt, False, schema, temperature))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        return response.json().get("message", {}).get("content", "")

    def stream(self, system: str, prompt: str, *, temperature: float = 0.2) -> Iterator[str]:
        import httpx
        try:
            with httpx.stream("POST", f"{self.url}/api/chat", timeout=self.timeout_s,
                              json=self._body(system, prompt, True, None, temperature)) as response:
                if response.status_code >= 400:
                    response.read()
                    response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    piece = json.loads(line).get("message", {}).get("content", "")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc

"""Google Gemini, through the google-genai SDK, with key rotation on quota errors."""

from __future__ import annotations

import time
from typing import Iterator

from .base import LLMProvider, LLMUnavailable
from .keys import KeyPool, api_keys

DEFAULT_MODEL = "gemini-2.5-flash"
# Waits before trying again when Google is overloaded ("503 UNAVAILABLE ... high demand").
RETRY_DELAYS = (2.0, 5.0)


def make_client(key: str):
    from google import genai
    from google.genai import types
    # The SDK defaults to API v1, which rejects system instructions
    # ("Developer instruction is not enabled for api version v1").
    return genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def is_transient(exc: Exception) -> bool:
    """Google's side is busy or hiccuping; the same call usually works seconds later."""
    return getattr(exc, "code", None) in (500, 502, 503, 504) or         any(s in str(exc) for s in ("UNAVAILABLE", "overloaded", "DEADLINE_EXCEEDED"))


def safety_off():
    """Players swear and fight monsters; the default filters block whole answers over it."""
    from google.genai import types
    return [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.OFF)
        for c in (types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                  types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                  types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                  types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)
    ]


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL, daily_limit: int = 20,
                 thinking_budget: int | None = None):
        self.model = model
        self.thinking_budget = thinking_budget
        self.pool = KeyPool(api_keys(), daily_limit, model)

    def describe(self) -> str:
        if not self.pool.keys:
            return f"{self.model} (no API key)"
        return f"{self.model} ({self.pool.left_today()} calls left today)"

    def _config(self, system: str, schema: dict | None, temperature: float):
        from google.genai import types
        extra = {}
        if schema is not None:
            extra.update(response_mime_type="application/json", response_schema=schema)
        if self.thinking_budget is not None:
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            safety_settings=safety_off(), **extra,
        )

    def _client(self):
        if not self.pool.keys:
            raise LLMUnavailable("no Gemini API key (set GEMINI_API in .env)")
        picked = self.pool.current()
        if picked is None:
            raise LLMUnavailable("daily call limit reached on every Gemini key")
        return picked

    def complete(self, system: str, prompt: str, *, schema: dict | None = None,
                 temperature: float = 0.2) -> str:
        waits = list(RETRY_DELAYS)
        while True:
            label, client = self._client()
            self.pool.count_call(label)
            try:
                response = client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._config(system, schema, temperature),
                )
            except Exception as exc:  # noqa: BLE001 -- SDK raises many types
                if is_quota_error(exc):
                    self.pool.mark_exhausted(label)
                    continue
                if is_transient(exc) and waits:
                    time.sleep(waits.pop(0))
                    continue
                raise LLMUnavailable(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
            if not response.text:
                reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
                raise LLMUnavailable(f"empty response ({reason})")
            return response.text

    def stream(self, system: str, prompt: str, *, temperature: float = 0.2) -> Iterator[str]:
        waits = list(RETRY_DELAYS)
        while True:
            label, client = self._client()
            self.pool.count_call(label)
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.model, contents=prompt,
                    config=self._config(system, None, temperature),
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as exc:  # noqa: BLE001
                if is_quota_error(exc) and not started:
                    self.pool.mark_exhausted(label)
                    continue
                if is_transient(exc) and not started and waits:
                    time.sleep(waits.pop(0))
                    continue
                raise LLMUnavailable(f"{type(exc).__name__}: {str(exc)[:300]}") from exc

"""
The interface every chat model provider implements, and a chain that falls
back to the next provider when one is out of quota or not running.
"""

from __future__ import annotations

from typing import Iterator


class LLMUnavailable(RuntimeError):
    """The provider can't answer right now (no key, quota used up, server down)."""


class Restart(str):
    """Yielded by ProviderChain.stream when an answer broke off and the next provider starts over."""


RESTART = Restart("")


class LLMProvider:
    name = "base"

    def describe(self) -> str:
        """Short label for the UI, e.g. 'gemini-2.5-flash (37 calls left today)'."""
        return self.name

    def complete(self, system: str, prompt: str, *, schema: dict | None = None,
                 temperature: float = 0.2) -> str:
        """Whole answer at once. With `schema`, the answer is JSON matching it."""
        raise NotImplementedError

    def stream(self, system: str, prompt: str, *, temperature: float = 0.2) -> Iterator[str]:
        """The answer as it is generated, piece by piece."""
        raise NotImplementedError


class ProviderChain(LLMProvider):
    """
    Try each provider in order. A provider that fails with LLMUnavailable
    hands over to the next one; `last_used` names the one that answered. If it
    fails partway through an answer, the stream yields RESTART first, so the
    caller drops the half answer before the next provider starts over.
    """
    name = "chain"

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("no LLM providers configured")
        self.providers = providers
        self.last_used: str | None = None

    def describe(self) -> str:
        return " → ".join(p.describe() for p in self.providers)

    def complete(self, system: str, prompt: str, *, schema: dict | None = None,
                 temperature: float = 0.2) -> str:
        errors = []
        for provider in self.providers:
            try:
                text = provider.complete(system, prompt, schema=schema, temperature=temperature)
                self.last_used = provider.describe()
                return text
            except LLMUnavailable as exc:
                errors.append(f"{getattr(provider, 'model', provider.name)}: {exc}")
        raise LLMUnavailable("; ".join(errors))

    def stream(self, system: str, prompt: str, *, temperature: float = 0.2) -> Iterator[str]:
        errors = []
        for provider in self.providers:
            started = False
            try:
                for piece in provider.stream(system, prompt, temperature=temperature):
                    if not started:
                        self.last_used = provider.describe()
                        started = True
                    yield piece
                return
            except LLMUnavailable as exc:
                errors.append(f"{getattr(provider, 'model', provider.name)}: {exc}")
                if started and provider is not self.providers[-1]:
                    yield RESTART
        raise LLMUnavailable("; ".join(errors))

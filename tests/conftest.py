"""
Shared fixtures. The real embedder, reranker and LLMs are swapped for small
deterministic fakes, so the suite runs in seconds with no GPU, no model
downloads and no API calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from craig_transcriber.llm.base import LLMProvider, LLMUnavailable
from craig_transcriber.paths import CONFIG_DIR
from craig_transcriber.rag.config import RagConfig


class FakeEmbedder:
    """Bag of hashed words: texts sharing words get similar vectors."""
    name = "fake-embedder"
    loaded = True

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.calls = 0
        self.texts_encoded = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        self.texts_encoded += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            out[i] /= max(np.linalg.norm(out[i]), 1e-9)
        return out


class FakeReranker:
    name = "fake-reranker"
    loaded = True

    def score(self, query, passages):
        q = set(re.findall(r"\w+", query.lower()))
        return np.array([len(q & set(re.findall(r"\w+", p.lower()))) / (len(q) or 1) for p in passages],
                        dtype=np.float32)


class FakeLLM(LLMProvider):
    """Plans with a fixed split, follows up once, answers citing source 1."""
    name = "fake"

    def __init__(self, plan=None, follow_up=None, fail=False):
        self.plan = plan or ["cult goals", "guardians goals"]
        self.follow_up = follow_up
        self.fail = fail
        self.prompts: list[tuple[str, str]] = []

    def describe(self):
        return "fake-llm"

    def complete(self, system, prompt, *, schema=None, temperature=0.2):
        self.prompts.append((system, prompt))
        if self.fail:
            raise LLMUnavailable("quota")
        if "complete" in json.dumps(schema or {}):
            if self.follow_up:
                queries, self.follow_up = self.follow_up, None
                return json.dumps({"complete": False, "queries": queries})
            return json.dumps({"complete": True, "queries": []})
        return json.dumps({"queries": self.plan})

    def stream(self, system, prompt, *, temperature=0.2):
        self.prompts.append((system, prompt))
        if self.fail:
            raise LLMUnavailable("quota")
        yield "The cult wants the shards "
        yield "to open The Nameless Beyond [1]."


@pytest.fixture
def archive(tmp_path: Path) -> RagConfig:
    """A tiny archive: two documents and one session transcript."""
    docs = tmp_path / "data" / "documents"
    docs.mkdir(parents=True)
    (docs / "campaign-notes.md").write_text(
        "# The Cult\n\nThe cult wants the celestial shards to open The Nameless Beyond.\n\n"
        "# The Guardians\n\nThe guardians protect the celestial shards and the altar.\n",
        encoding="utf-8")
    (docs / "rules.txt").write_text(
        "Ritual Adept lets you cast a ritual spell from your spellbook without preparing it.\n\n"
        "A long rest takes eight hours. A short rest takes one hour.", encoding="utf-8")
    session = tmp_path / "data" / "transcripts" / "ABC123"
    session.mkdir(parents=True)
    (session / "recording.json").write_text(json.dumps({"Recording": "ABC123", "Start time": "2026-08-29T20:19:26Z"}))
    (session / "transcript.json").write_text(json.dumps({"segments": [
        {"speaker": "Dungeon Master", "start_s": 10.0, "end_s": 14.0, "text": "انت شايف الـ altar قدامك"},
        {"speaker": "Arnok", "start_s": 15.0, "end_s": 18.0, "text": "I make a religion check on the altar"},
        {"speaker": "Kael", "start_s": 400.0, "end_s": 404.0, "text": "يلا نعمل long rest"},
        {"speaker": "Kael", "start_s": 410.0, "end_s": 412.0, "text": "invented", "dropped": True},
    ]}, ensure_ascii=False), encoding="utf-8")

    config = RagConfig(root=tmp_path, index_dir=tmp_path / "data" / "index")
    config.router.examples = str(CONFIG_DIR / "router_examples.jsonl")
    config.chunking.chunk_words = 40
    config.chunking.overlap_words = 5
    return config

"""Settings for the archive, read from config/rag.toml over built-in defaults."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from ..paths import CONFIG_DIR, INDEX_DIR, ROOT


@dataclass
class PartyConfig:
    name: str = ""             # the players' party; shown in the UI and told to the LLM
    tagline: str = "Lore of the party"
    emblem: str = "assets/emblem"
    roster: str = "config/party.toml"  # the characters: a page each, and who is who for the LLM
    comics: str = "assets/comics"      # a folder per comic, pages in name order
    glossary: str = "config/glossary.txt"  # campaign names: background for answers, other spellings for search


@dataclass
class SourcesConfig:
    documents: str = "data/documents"
    transcripts: str = "data/transcripts"


@dataclass
class ChunkingConfig:
    chunk_words: int = 220
    overlap_words: int = 40
    transcript_window_s: float = 120.0
    transcript_words: int = 220


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-m3"
    pooling: str = "cls"
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = 1024
    batch_size: int = 16
    device: str = "auto"


@dataclass
class RetrievalConfig:
    dense_k: int = 40
    keyword_k: int = 40
    candidates: int = 30
    final_k: int = 6


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    batch_size: int = 8


@dataclass
class RouterConfig:
    mode: str = "classifier"
    threshold: float = 0.5
    examples: str = "config/router_examples.jsonl"


@dataclass
class MultistepConfig:
    max_subqueries: int = 4
    max_hops: int = 2
    per_query_k: int = 4
    max_context: int = 10


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    fallback_models: list[str] = field(default_factory=list)  # tried in order when `model` is down
    daily_limit: int = 20
    thinking_budget: int | None = None


@dataclass
class OllamaConfig:
    model: str = "qwen3:4b"
    url: str = "http://localhost:11434"
    think: bool | None = None


@dataclass
class LLMConfig:
    providers: list[str] = field(default_factory=lambda: ["gemini", "ollama"])
    temperature: float = 0.2
    history_turns: int = 3
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 50
    users: str = "config/users.toml"   # logins; none -> the site is open (this computer only)
    session_days: int = 30             # how long a login lasts


@dataclass
class RagConfig:
    party: PartyConfig = field(default_factory=PartyConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    multistep: MultistepConfig = field(default_factory=MultistepConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    index_dir: Path = INDEX_DIR
    root: Path = ROOT

    def path(self, value: str) -> Path | None:
        """A configured path, resolved against the project root ("" -> None)."""
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p


DEFAULT_CONFIG = CONFIG_DIR / "rag.toml"


def _merge(target, values: dict, where: str) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"{where}: unknown setting '{key}'")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{where}.{key} must be a table")
            _merge(current, value, f"{where}.{key}")
        else:
            setattr(target, key, value)


def load_config(path: Path | None = None) -> RagConfig:
    config = RagConfig()
    path = path or DEFAULT_CONFIG
    if path.exists():
        with path.open("rb") as fh:
            _merge(config, tomllib.load(fh), path.name)
    return config

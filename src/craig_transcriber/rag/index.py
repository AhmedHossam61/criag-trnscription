"""
The index on disk (data/index/) and keeping it in step with the files.

    manifest.json   one entry per file: its hash, title, kind, passage count
    passages.jsonl  every passage, in the same order as the vectors
    vectors.npy     passage embeddings, float16

Re-indexing only embeds files that are new or changed; passages of unchanged
files keep their vectors. Changing the embedding model or the chunking
settings rebuilds everything, since old and new vectors don't mix.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import Passage, chunk_document
from .config import RagConfig
from .documents import SourceFile, read_sections, scan_sources
from .keyword import BM25
from .models import Embedder

INDEX_VERSION = 1


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    passages: int = 0
    seconds: float = 0.0
    rebuilt: bool = False

    def summary(self) -> str:
        parts = [f"{self.passages} passages"]
        for label, items in (("added", self.added), ("updated", self.updated), ("removed", self.removed)):
            if items:
                parts.append(f"{len(items)} {label}")
        if self.unchanged:
            parts.append(f"{self.unchanged} unchanged")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts) + f" ({self.seconds:.1f}s)"


class Index:
    """Passages, their vectors and a keyword index, loaded in memory."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.manifest: dict = {"version": INDEX_VERSION, "docs": {}}
        self.passages: list[Passage] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.by_id: dict[str, int] = {}
        self.keyword = BM25([])

    # -- disk ---------------------------------------------------------------

    @classmethod
    def load(cls, directory: Path) -> "Index":
        index = cls(directory)
        manifest = directory / "manifest.json"
        if not manifest.exists():
            return index
        index.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        with (directory / "passages.jsonl").open(encoding="utf-8") as fh:
            index.passages = [Passage(**json.loads(line)) for line in fh if line.strip()]
        index.vectors = np.load(directory / "vectors.npy").astype(np.float32)
        if len(index.vectors) != len(index.passages):
            raise RuntimeError(f"{directory} is inconsistent; run `scribe rag ingest --rebuild`")
        index._refresh()
        return index

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = {name: self.dir / f"{name}.tmp" for name in ("manifest", "passages", "vectors")}
        with tmp["passages"].open("w", encoding="utf-8") as fh:
            for p in self.passages:
                fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        with tmp["vectors"].open("wb") as fh:
            np.save(fh, self.vectors.astype(np.float16))
        tmp["manifest"].write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        # Manifest last: a crash mid-save leaves the old manifest pointing at
        # files that are either both old or both new.
        tmp["passages"].replace(self.dir / "passages.jsonl")
        tmp["vectors"].replace(self.dir / "vectors.npy")
        tmp["manifest"].replace(self.dir / "manifest.json")

    def _refresh(self) -> None:
        self.by_id = {p.id: i for i, p in enumerate(self.passages)}
        self.keyword = BM25([p.embed_text() for p in self.passages])

    # -- queries ------------------------------------------------------------

    @property
    def docs(self) -> dict[str, dict]:
        return self.manifest.get("docs", {})

    def dense_top(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.passages):
            return []
        sims = self.vectors @ query_vec
        k = min(k, len(sims))
        best = np.argpartition(-sims, k - 1)[:k]
        best = best[np.argsort(-sims[best])]
        return [(int(i), float(sims[i])) for i in best]

    def neighbours(self, passage_id: str) -> tuple[Passage | None, Passage | None]:
        """The passages just before and after, in the same file."""
        i = self.by_id.get(passage_id)
        if i is None:
            return None, None
        p = self.passages[i]
        before = self.passages[i - 1] if i > 0 and self.passages[i - 1].doc_id == p.doc_id else None
        after = (self.passages[i + 1] if i + 1 < len(self.passages)
                 and self.passages[i + 1].doc_id == p.doc_id else None)
        return before, after


def settings_signature(config: RagConfig) -> dict:
    """Whatever, if changed, makes existing vectors unusable."""
    return {"embedding": asdict(config.embedding) | {"batch_size": None, "device": None},
            "chunking": asdict(config.chunking)}


def ingest(config: RagConfig, embedder: Embedder, *, rebuild: bool = False,
           progress: Callable[[str], None] = print) -> tuple[Index, IngestReport]:
    t0 = time.time()
    report = IngestReport()
    old = Index.load(config.index_dir) if not rebuild else Index(config.index_dir)
    signature = settings_signature(config)
    if old.manifest.get("settings") not in (None, signature) or old.manifest.get("version") != INDEX_VERSION:
        progress("Embedding or chunking settings changed: rebuilding the whole index.")
        old = Index(config.index_dir)
        report.rebuilt = True
    report.rebuilt |= rebuild

    sources = scan_sources(config)
    old_docs = old.docs
    old_rows: dict[str, list[int]] = {}
    for i, p in enumerate(old.passages):
        old_rows.setdefault(p.doc_id, []).append(i)

    new_docs: dict[str, dict] = {}
    passages: list[Passage] = []
    vectors: list[np.ndarray] = []
    to_embed: list[tuple[SourceFile, str]] = []

    for src in sources:
        digest = file_hash(src.path)
        prev = old_docs.get(src.doc_id)
        if prev and prev.get("sha256") == digest and src.doc_id in old_rows:
            rows = old_rows[src.doc_id]
            passages += [old.passages[i] for i in rows]
            vectors.append(old.vectors[rows])
            new_docs[src.doc_id] = prev | {"title": src.title}
            report.unchanged += 1
        else:
            to_embed.append((src, digest))

    for n, (src, digest) in enumerate(to_embed, start=1):
        progress(f"  [{n}/{len(to_embed)}] {src.doc_id}")
        try:
            doc_passages = chunk_document(src, read_sections(src), config.chunking)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the rest
            report.failed[src.doc_id] = f"{type(exc).__name__}: {exc}"
            progress(f"      failed: {report.failed[src.doc_id]}")
            if src.doc_id in old_rows and src.doc_id in old_docs:  # keep the last good version
                rows = old_rows[src.doc_id]
                passages += [old.passages[i] for i in rows]
                vectors.append(old.vectors[rows])
                new_docs[src.doc_id] = old_docs[src.doc_id]
            continue
        if doc_passages:
            vecs = embedder.encode([p.embed_text() for p in doc_passages])
            passages += doc_passages
            vectors.append(vecs)
        new_docs[src.doc_id] = {
            "path": src.doc_id, "title": src.title, "kind": src.kind, "sha256": digest,
            "size": src.path.stat().st_size, "passages": len(doc_passages),
            "indexed": time.strftime("%Y-%m-%d %H:%M"),
        }
        (report.updated if src.doc_id in old_docs else report.added).append(src.doc_id)

    report.removed = [d for d in old_docs if d not in new_docs and d not in report.failed]

    index = Index(config.index_dir)
    index.passages = passages
    dims = [v.shape[1] for v in vectors if v.size]
    index.vectors = (np.concatenate([v for v in vectors if v.size]).astype(np.float32)
                     if dims else np.zeros((0, 0), dtype=np.float32))
    index.manifest = {"version": INDEX_VERSION, "settings": signature,
                      "embedding_model": getattr(embedder, "name", config.embedding.model), "docs": new_docs}
    index._refresh()
    index.save()
    report.passages = len(passages)
    report.seconds = time.time() - t0
    return index, report

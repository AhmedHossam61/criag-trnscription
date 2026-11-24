"""
One search: meaning (embeddings) and exact words (BM25), merged by
reciprocal rank fusion, then reordered by the cross-encoder reranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .chunking import Passage
from .config import RetrievalConfig
from .index import Index
from .models import Embedder, Reranker

RRF_K = 60  # the usual constant: rank 1 scores 1/61, rank 10 scores 1/70


@dataclass
class Hit:
    passage: Passage
    score: float                 # what the ranking used: rerank score, else fused score
    dense: float | None = None   # cosine similarity
    keyword: float | None = None # BM25
    rerank: float | None = None  # 0..1
    query: str = ""

    def as_source(self, n: int) -> dict:
        p = self.passage
        return {
            "n": n, "id": p.id, "doc_id": p.doc_id, "title": p.title, "kind": p.kind,
            "heading": p.heading, "location": p.location, "text": p.text,
            "score": round(self.score, 4),
            "dense": None if self.dense is None else round(self.dense, 4),
            "keyword": None if self.keyword is None else round(self.keyword, 3),
            "rerank": None if self.rerank is None else round(self.rerank, 4),
            "query": self.query,
        }


class Retriever:
    def __init__(self, cfg: RetrievalConfig, index: Index, embedder: Embedder,
                 reranker: Reranker | None, expand: Callable[[str], str] | None = None):
        self.cfg = cfg
        self.expand = expand      # extra words for the keyword search (other spellings of names)
        self.index = index
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, k: int, query_vec: np.ndarray | None = None) -> list[Hit]:
        if not self.index.passages:
            return []
        if query_vec is None:
            query_vec = self.embedder.encode([query], query=True)[0]
        dense = self.index.dense_top(query_vec, self.cfg.dense_k)
        keyword = self.index.keyword.top(self.expand(query) if self.expand else query, self.cfg.keyword_k)

        fused: dict[int, Hit] = {}
        for rank, (i, sim) in enumerate(dense, start=1):
            fused[i] = Hit(self.index.passages[i], 1 / (RRF_K + rank), dense=sim, query=query)
        for rank, (i, s) in enumerate(keyword, start=1):
            hit = fused.setdefault(i, Hit(self.index.passages[i], 0.0, query=query))
            hit.keyword = s
            hit.score += 1 / (RRF_K + rank)
        shortlist = sorted(fused.values(), key=lambda h: h.score, reverse=True)[:self.cfg.candidates]

        if self.reranker is not None and shortlist:
            scores = self.reranker.score(query, [h.passage.embed_text() for h in shortlist])
            for hit, s in zip(shortlist, scores):
                hit.rerank = float(s)
                hit.score = float(s)
            shortlist.sort(key=lambda h: h.score, reverse=True)
        return shortlist[:k]

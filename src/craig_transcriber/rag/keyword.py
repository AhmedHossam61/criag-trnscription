"""
BM25 keyword search. Embeddings find passages about the same thing; this
finds passages with the same words, which matters for names ("Arnok"),
spells, item names and numbers that an embedder blurs together.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..transcription.filters import normalize_arabic

STOPWORDS = set("""
a an and are as at be but by can could did do does for from had has have he her
him his how i if in into is it its me my no not of on or our she so than that the
their them then there these they this to us was we were what when where which who
whom why will with would you your about after before also any been being both
each just more most other over some such through up very
""".split())

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(normalize_arabic(text).lower()):
        if tok in STOPWORDS or (len(tok) == 1 and not tok.isdigit()):
            continue
        # Light plural folding so "guardians" finds "guardian".
        if len(tok) > 4 and tok.isascii() and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.lengths = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.postings = {
            term: (np.array([d for d, _ in p], dtype=np.int32), np.array([t for _, t in p], dtype=np.float32))
            for term, p in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_len, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            out[docs] += idf * tf * (self.k1 + 1) / (tf + norm[docs])
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        hits = np.flatnonzero(s > 0)
        if not len(hits):
            return []
        best = hits[np.argsort(-s[hits])[:k]]
        return [(int(i), float(s[i])) for i in best]

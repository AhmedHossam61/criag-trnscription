"""
Deciding whether a question needs one search or several.

"What does Ritual Adept do?" is answered by one passage. "Compare what the
cult and the guardians want from the celestial shards" needs passages about
each, found by separate searches. Sending every question down the multi-step
path would cost two extra LLM calls each; sending none would miss half the
answer on the hard ones.

The classifier is logistic regression over the question's embedding (already
computed for the search, so it costs nothing) plus a few counts of telltale
words. It trains in well under a second on config/router_examples.jsonl and is
cached in data/index/router.npz until the examples or the embedder change.
Add your own questions to the examples file to teach it your style.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Embedder

LABELS = ("single", "multi")

# Words that tend to mean a question has several parts. Arabic forms cover
# how the group writes ("الفرق بين", "وكمان", "قارن").
_MULTI_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference|differences|differ|between|both|each of)\b",
    r"\b(and also|as well as|in addition|together with|relationship|related|connect(ed|ion)?)\b",
    r"\b(before and after|over time|across sessions|every time|all the|list all|summari[sz]e)\b",
    r"\b(why did .* after|how did .* change|what led to|because of what)\b",
    r"(الفرق|قارن|مقارنة|وكمان|بين .* و|ازاي اتغير|كل مرة|كل الجلسات|لخص)",
]
_MULTI_RE = [re.compile(p, re.IGNORECASE) for p in _MULTI_PATTERNS]
_WH = re.compile(r"\b(who|what|when|where|why|how|which)\b|(مين|ايه|امتى|فين|ليه|ازاي)", re.IGNORECASE)


def heuristic_features(question: str) -> np.ndarray:
    q = question.strip()
    return np.array([
        sum(bool(r.search(q)) for r in _MULTI_RE),
        max(0, len(_WH.findall(q)) - 1),           # extra wh-words: "who ... and why ..."
        max(0, q.count("?") - 1),
        len(re.findall(r"\band\b| و", q)),
        min(len(q.split()), 40) / 10.0,
    ], dtype=np.float32)


def heuristic_probability(question: str) -> float:
    f = heuristic_features(question)
    z = -2.0 + 2.2 * f[0] + 1.2 * f[1] + 1.2 * f[2] + 0.35 * f[3] + 0.25 * f[4]
    return float(1 / (1 + np.exp(-z)))


@dataclass
class Route:
    mode: str           # "single" | "multi"
    probability: float  # P(multi)
    by: str             # classifier | heuristic | forced

    def as_dict(self) -> dict:
        return {"mode": self.mode, "probability": round(self.probability, 3), "by": self.by}


def load_examples(path: Path) -> list[tuple[str, int]]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        examples.append((row["question"], LABELS.index(row["label"])))
    return examples


class LogisticModel:
    """Plain L2-regularised logistic regression on standardised features."""

    def __init__(self, w=None, b=0.0, mean=None, std=None):
        self.w, self.b, self.mean, self.std = w, b, mean, std

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 0.05, steps: int = 600,
            lr: float = 0.5) -> "LogisticModel":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Z = (X - self.mean) / self.std
        w = np.zeros(Z.shape[1])
        b = 0.0
        # Balance the classes so a lopsided example file doesn't bias the router.
        pos = y.mean() if len(y) else 0.5
        weight = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = weight * (p - y)
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        self.w, self.b = w, b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))


class QueryRouter:
    def __init__(self, mode: str, threshold: float, examples_path: Path | None,
                 embedder: Embedder, cache_path: Path):
        self.mode = mode
        self.threshold = threshold
        self.examples_path = examples_path
        self.embedder = embedder
        self.cache_path = cache_path
        self.model: LogisticModel | None = None
        self.n_examples = 0

    # -- training -------------------------------------------------------------

    def _signature(self) -> str:
        text = self.examples_path.read_bytes() if self.examples_path and self.examples_path.exists() else b""
        return hashlib.sha256(text + self.embedder.name.encode()).hexdigest()[:16]

    def features(self, questions: list[str], vectors: np.ndarray | None = None) -> np.ndarray:
        if vectors is None:
            vectors = self.embedder.encode(questions, query=True)
        return np.hstack([vectors, np.stack([heuristic_features(q) for q in questions])])

    def ensure_model(self) -> LogisticModel | None:
        """Load the cached classifier, or train it if the examples changed."""
        if self.mode != "classifier":
            return None
        if self.model is not None:
            return self.model
        if not self.examples_path or not self.examples_path.exists():
            return None
        sig = self._signature()
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            if str(data["signature"]) == sig:
                self.model = LogisticModel(data["w"], float(data["b"]), data["mean"], data["std"])
                self.n_examples = int(data["n"])
                return self.model
        examples = load_examples(self.examples_path)
        X = self.features([q for q, _ in examples])
        y = np.array([label for _, label in examples], dtype=np.float64)
        self.model = LogisticModel().fit(X, y)
        self.n_examples = len(examples)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache_path, signature=np.array(sig), w=self.model.w, b=np.array(self.model.b),
                 mean=self.model.mean, std=self.model.std, n=np.array(len(examples)))
        return self.model

    # -- deciding -------------------------------------------------------------

    def decide(self, question: str, query_vec: np.ndarray | None = None) -> Route:
        if self.mode in LABELS:
            return Route(self.mode, 1.0 if self.mode == "multi" else 0.0, "forced")
        model = self.ensure_model()
        if model is None:
            p = heuristic_probability(question)
            by = "heuristic"
        else:
            vec = None if query_vec is None else query_vec[None, :]
            p = float(model.predict(self.features([question], vec))[0])
            by = "classifier"
        return Route("multi" if p >= self.threshold else "single", p, by)

    def evaluate(self, folds: int = 5) -> dict:
        """Cross-validated accuracy of the classifier vs the keyword rules."""
        if not self.examples_path or not self.examples_path.exists():
            raise FileNotFoundError(f"no examples at {self.examples_path}")
        examples = load_examples(self.examples_path)
        questions = [q for q, _ in examples]
        y = np.array([label for _, label in examples])
        X = self.features(questions)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(y))
        predicted = np.zeros(len(y))
        for f in range(folds):
            test = order[f::folds]
            train = np.setdiff1d(order, test)
            model = LogisticModel().fit(X[train], y[train].astype(np.float64))
            predicted[test] = model.predict(X[test])
        clf = (predicted >= self.threshold).astype(int)
        heur = np.array([heuristic_probability(q) >= self.threshold for q in questions], dtype=int)
        mistakes = [(questions[i], LABELS[y[i]], round(float(predicted[i]), 2))
                    for i in range(len(y)) if clf[i] != y[i]]
        return {
            "examples": len(y),
            "multi_share": float(y.mean()),
            "classifier_accuracy": float((clf == y).mean()),
            "heuristic_accuracy": float((heur == y).mean()),
            "mistakes": mistakes,
        }

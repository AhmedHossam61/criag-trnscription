"""
The two local models: an embedder (text -> vector, for search by meaning) and
a cross-encoder reranker (question + passage -> relevance score).

Both run through plain `transformers`, already installed for the speech
models, and load lazily on first use. In fp16 on the GPU, bge-m3 and
bge-reranker-v2-m3 take about 1.1 GB of VRAM each.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from .config import EmbeddingConfig, RerankerConfig


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """L2-normalised float32 vectors, one row per text."""


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance in [0, 1] per passage."""


def pick_device(setting: str) -> str:
    import torch
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


class TransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.name = cfg.model
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            self.device = pick_device(self.cfg.device)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model)
            self._model = AutoModel.from_pretrained(self.cfg.model, dtype=dtype).to(self.device).eval()
            self._torch = torch

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _pool(self, hidden, mask):
        if self.cfg.pooling == "cls":
            return hidden[:, 0]
        if self.cfg.pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        if self.cfg.pooling == "last":
            if self._tok.padding_side == "left":
                return hidden[:, -1]
            last = mask.sum(1) - 1
            return hidden[self._torch.arange(hidden.shape[0], device=hidden.device), last]
        raise ValueError(f"unknown pooling '{self.cfg.pooling}' (cls, mean or last)")

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        self.load()
        prefix = self.cfg.query_prefix if query else self.cfg.passage_prefix
        texts = [prefix + t for t in texts]
        out = np.zeros((len(texts), 0), dtype=np.float32)
        if not texts:
            return out
        # Longest first so each batch pads to similar lengths.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        rows: dict[int, np.ndarray] = {}
        torch = self._torch
        with self._lock, torch.inference_mode():
            for start in range(0, len(order), self.cfg.batch_size):
                idx = order[start:start + self.cfg.batch_size]
                enc = self._tok([texts[i] for i in idx], padding=True, truncation=True,
                                max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
                hidden = self._model(**enc).last_hidden_state
                vec = torch.nn.functional.normalize(self._pool(hidden, enc["attention_mask"]).float(), dim=-1)
                for i, v in zip(idx, vec.cpu().numpy()):
                    rows[i] = v
        return np.stack([rows[i] for i in range(len(texts))]).astype(np.float32)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankerConfig, device: str = "auto"):
        self.cfg = cfg
        self.name = cfg.model
        self.device_setting = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            self.device = pick_device(self.device_setting)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.model, dtype=dtype).to(self.device).eval()
            self._torch = torch

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        self.load()
        if not passages:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        scores: list[float] = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(passages), self.cfg.batch_size):
                batch = passages[start:start + self.cfg.batch_size]
                enc = self._tok([query] * len(batch), batch, padding=True, truncation=True,
                                max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
                logits = self._model(**enc).logits.view(-1).float()
                scores += torch.sigmoid(logits).cpu().tolist()
        return np.asarray(scores, dtype=np.float32)

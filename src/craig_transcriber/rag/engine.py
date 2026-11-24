"""
The archive's question answering, end to end.

    question ──► router ──► one-step:  search ─────────────────────────┐
                        └─► multi-step: plan sub-queries ─► search each │
                                        ─► follow-up search if gaps ────┤
                                                                        ▼
                                               sources ──► LLM ──► answer with [n] citations

`RagEngine.ask()` yields events as it goes (route, plan, sources, tokens,
done), which the chat UI streams and the terminal prints.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable, Iterator

from ..llm.base import RESTART, LLMProvider, LLMUnavailable, ProviderChain
from . import prompts
from .config import RagConfig
from .glossary import GlossaryFile
from .index import Index, IngestReport, ingest
from .models import CrossEncoderReranker, Embedder, Reranker, TransformerEmbedder
from .party import Party
from .retriever import Hit, Retriever
from .router import QueryRouter, Route

# Words that mean the question leans on the previous one ("what about him?").
_FOLLOW_UP = re.compile(
    r"\b(he|she|it|they|him|her|them|his|its|their|that|this|those|these|there|then|"
    r"what about|and the|same)\b|(هو|هي|هم|ده|دي|دول|كده|بتاعه|بتاعها)",
    re.IGNORECASE,
)


def build_llm(config: RagConfig) -> LLMProvider:
    providers: list[LLMProvider] = []
    for name in config.llm.providers:
        if name == "gemini":
            from ..llm.gemini import GeminiProvider
            g = config.llm.gemini
            for model in [g.model, *g.fallback_models]:
                providers.append(GeminiProvider(model, g.daily_limit, g.thinking_budget))
        elif name == "ollama":
            from ..llm.ollama import OllamaProvider
            o = config.llm.ollama
            providers.append(OllamaProvider(o.model, o.url, o.think))
        else:
            raise ValueError(f"unknown LLM provider '{name}' in rag.toml (gemini or ollama)")
    return ProviderChain(providers)


def heuristic_split(question: str, limit: int) -> list[str]:
    """Without an LLM to plan: split on question marks and 'and'/'compare' joints."""
    parts = re.split(r"\?|\band (?:also |then )?(?=what|who|how|why|when|where|which)|،| وكمان ", question)
    parts = [p.strip(" ,.;") for p in parts if len(p.strip().split()) >= 2]
    return parts[:limit] or [question]


class RagEngine:
    def __init__(self, config: RagConfig, *, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, llm: LLMProvider | None = None,
                 use_reranker: bool | None = None):
        self.config = config
        self.embedder = embedder or TransformerEmbedder(config.embedding)
        enabled = config.reranker.enabled if use_reranker is None else use_reranker
        if reranker is None and enabled:
            reranker = CrossEncoderReranker(config.reranker, config.embedding.device)
        self.reranker = reranker if enabled else None
        self.llm = llm or build_llm(config)
        self.router = QueryRouter(
            config.router.mode, config.router.threshold,
            config.path(config.router.examples), self.embedder,
            config.index_dir / "router.npz",
        )
        self.party = Party(config)
        self.glossary = GlossaryFile(config.path(config.party.glossary))
        self._lock = threading.RLock()   # guards index swaps
        self._index: Index | None = None
        self.ingesting = False
        self.last_ingest: IngestReport | None = None

    # -- index ----------------------------------------------------------------

    @property
    def index(self) -> Index:
        with self._lock:
            if self._index is None:
                self._index = Index.load(self.config.index_dir)
            return self._index

    def warm_up(self) -> None:
        """Load the index and models now, so the first question is fast."""
        _ = self.index
        for model in (self.embedder, self.reranker):
            if model is not None and hasattr(model, "load"):
                model.load()
        self.router.ensure_model()

    def reindex(self, *, rebuild: bool = False, progress: Callable[[str], None] = print) -> IngestReport:
        with self._lock:
            if self.ingesting:
                raise RuntimeError("indexing is already running")
            self.ingesting = True
        try:
            index, report = ingest(self.config, self.embedder, rebuild=rebuild, progress=progress)
            with self._lock:
                self._index = index
                self.last_ingest = report
            return report
        finally:
            self.ingesting = False

    def retriever(self) -> Retriever:
        return Retriever(self.config.retrieval, self.index, self.embedder, self.reranker, self._expand)

    def status(self) -> dict:
        index = self.index
        kinds: dict[str, int] = {}
        for doc in index.docs.values():
            kinds[doc["kind"]] = kinds.get(doc["kind"], 0) + 1
        indexed_model = index.manifest.get("embedding_model")
        return {
            "party": {"name": self.config.party.name, "tagline": self.config.party.tagline},
            "documents": len(index.docs),
            "passages": len(index.passages),
            "kinds": kinds,
            "embedding_model": self.embedder.name,
            "index_model": indexed_model,
            "index_stale": bool(indexed_model and indexed_model != self.embedder.name),
            "reranker": self.reranker.name if self.reranker else None,
            "router": self.config.router.mode,
            "llm": self.llm.describe(),
            "ingesting": self.ingesting,
            "models_loaded": all(getattr(m, "loaded", True) for m in (self.embedder, self.reranker) if m),
        }

    def documents(self) -> list[dict]:
        return sorted(self.index.docs.values(), key=lambda d: (d["kind"] == "transcript", d["path"]))

    def passage(self, passage_id: str) -> dict | None:
        index = self.index
        i = index.by_id.get(passage_id)
        if i is None:
            return None
        p = index.passages[i]
        before, after = index.neighbours(passage_id)
        return {
            "id": p.id, "doc_id": p.doc_id, "title": p.title, "kind": p.kind, "heading": p.heading,
            "location": p.location, "text": p.text,
            "before": before.text if before else None, "after": after.text if after else None,
        }

    # -- asking ---------------------------------------------------------------

    def classify(self, question: str) -> Route:
        vec = self.embedder.encode([question], query=True)[0]
        return self.router.decide(question, vec)

    def _expand(self, query: str) -> str:
        """A keyword search gets every spelling of the names it uses: the roster's, then the glossary's."""
        for expand in (self.party.expand_query, lambda q: self.glossary.current.expand(q)):
            try:
                query = expand(query)
            except (OSError, ValueError):   # a broken roster or glossary must not break search
                pass
        return query

    def _system(self, system: str) -> str:
        try:
            roster = self.party.describe()
        except (OSError, ValueError):
            roster = ""
        try:
            glossary = self.glossary.current.text
        except (OSError, ValueError):
            glossary = ""
        return prompts.for_party(system, self.config.party.name, roster, glossary)

    def search_query(self, question: str, history: list[dict]) -> str:
        """Fold the previous question in when this one only makes sense after it."""
        last_user = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        if last_user and (len(question.split()) <= 6 or _FOLLOW_UP.search(question)):
            return f"{last_user.strip()} {question.strip()}"
        return question

    def ask(self, question: str, history: list[dict] | None = None,
            mode: str = "auto") -> Iterator[dict]:
        history = history or []
        t0 = time.time()
        question = question.strip()
        if not question:
            yield {"type": "error", "message": "Ask a question first."}
            return
        index = self.index
        if not index.passages:
            yield {"type": "error",
                   "message": "The archive is empty. Add files to data/documents and run Re-index."}
            return

        query = self.search_query(question, history)
        query_vec = self.embedder.encode([query], query=True)[0]
        if mode in ("single", "multi"):
            route = Route(mode, 1.0 if mode == "multi" else 0.0, "you")
        else:
            route = self.router.decide(question, query_vec)
        yield {"type": "route", **route.as_dict(), "query": query}

        retriever = Retriever(self.config.retrieval, index, self.embedder, self.reranker, self._expand)
        t_search = time.time()
        if route.mode == "single":
            hits = retriever.search(query, self.config.retrieval.final_k, query_vec)
            yield {"type": "plan", "hop": 1, "queries": [query], "planner": "none"}
        else:
            hits = yield from self._multi_step(question, query, query_vec, retriever)
        search_s = time.time() - t_search

        sources = [h.as_source(n) for n, h in enumerate(hits, start=1)]
        yield {"type": "sources", "sources": sources}

        prompt = prompts.answer_prompt(question, sources, history, self.config.llm.history_turns)
        t_answer = time.time()
        answer = []
        try:
            for piece in self.llm.stream(self._system(prompts.ANSWER_SYSTEM), prompt,
                                         temperature=self.config.llm.temperature):
                if piece is RESTART:        # that model broke off; the next one starts over
                    answer.clear()
                    yield {"type": "restart"}
                    continue
                answer.append(piece)
                yield {"type": "token", "text": piece}
        except LLMUnavailable as exc:
            yield {"type": "error",
                   "message": f"No language model could answer: {exc}. The sources above are what the search found."}
            return
        yield {
            "type": "done", "answer": "".join(answer),
            "model": getattr(self.llm, "last_used", None) or self.llm.describe(),
            "timing": {"search_s": round(search_s, 2), "answer_s": round(time.time() - t_answer, 2),
                       "total_s": round(time.time() - t0, 2)},
        }

    def _plan(self, question: str) -> tuple[list[str], str]:
        limit = self.config.multistep.max_subqueries
        try:
            raw = self.llm.complete(self._system(prompts.PLAN_SYSTEM.format(max_q=limit)), question,
                                    schema=prompts.PLAN_SCHEMA, temperature=0.0)
            queries = [q.strip() for q in json.loads(raw).get("queries", []) if q.strip()]
            if queries:
                return queries[:limit], "llm"
        except (LLMUnavailable, json.JSONDecodeError, AttributeError):
            pass
        return heuristic_split(question, limit), "rules"

    def _follow_up(self, question: str, tried: list[str], hits: list[Hit]) -> list[str]:
        limit = self.config.multistep.max_subqueries
        sample = [h.as_source(0) for h in hits[:12]]
        try:
            raw = self.llm.complete(self._system(prompts.FOLLOWUP_SYSTEM.format(max_q=limit)),
                                    prompts.followup_prompt(question, tried, sample),
                                    schema=prompts.FOLLOWUP_SCHEMA, temperature=0.0)
            data = json.loads(raw)
        except (LLMUnavailable, json.JSONDecodeError):
            return []
        if data.get("complete"):
            return []
        seen = {q.lower() for q in tried}
        return [q.strip() for q in data.get("queries", []) if q.strip() and q.strip().lower() not in seen][:limit]

    def _multi_step(self, question: str, query: str, query_vec, retriever: Retriever):
        ms = self.config.multistep
        subqueries, planner = self._plan(question)
        queries = [query] + [q for q in subqueries if q.lower() != query.lower()]
        yield {"type": "plan", "hop": 1, "queries": queries, "planner": planner}

        per_query: list[list[Hit]] = []
        tried: list[str] = []
        for hop in range(1, ms.max_hops + 1):
            for q in queries:
                vec = query_vec if q == query else None
                per_query.append(retriever.search(q, ms.per_query_k, vec))
                tried.append(q)
            if hop == ms.max_hops or planner != "llm":
                break
            found = [h for hits in per_query for h in hits]
            queries = self._follow_up(question, tried, found)
            if not queries:
                break
            yield {"type": "plan", "hop": hop + 1, "queries": queries, "planner": "llm"}

        # Round-robin over the searches so every sub-question gets its best
        # passages in before any gets its third.
        picked: dict[str, Hit] = {}
        depth = 0
        while len(picked) < ms.max_context and any(depth < len(h) for h in per_query):
            for hits in per_query:
                if depth < len(hits) and hits[depth].passage.id not in picked:
                    picked[hits[depth].passage.id] = hits[depth]
                    if len(picked) >= ms.max_context:
                        break
            depth += 1
        return list(picked.values())

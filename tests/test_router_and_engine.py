import pytest

from craig_transcriber.llm.base import RESTART, LLMUnavailable, ProviderChain
from craig_transcriber.rag.engine import RagEngine, heuristic_split
from craig_transcriber.rag.router import QueryRouter, heuristic_probability, load_examples
from craig_transcriber.paths import CONFIG_DIR

from conftest import FakeEmbedder, FakeLLM, FakeReranker


def engine_for(archive, llm=None):
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm or FakeLLM())
    engine.reindex(progress=lambda _: None)
    return engine


def test_example_file_is_balanced_and_valid():
    examples = load_examples(CONFIG_DIR / "router_examples.jsonl")
    labels = [label for _, label in examples]
    assert len(examples) >= 60 and 0.3 < sum(labels) / len(labels) < 0.7


def test_heuristic_spots_comparisons():
    assert heuristic_probability("Compare the cult and the guardians") > 0.5
    assert heuristic_probability("What is a long rest?") < 0.5


def test_classifier_trains_caches_and_decides(tmp_path):
    router = QueryRouter("classifier", 0.5, CONFIG_DIR / "router_examples.jsonl", FakeEmbedder(),
                         tmp_path / "router.npz")
    route = router.decide("What is the difference between a long rest and a short rest?")
    assert route.by == "classifier" and route.mode == "multi"
    assert (tmp_path / "router.npz").exists()
    again = QueryRouter("classifier", 0.5, CONFIG_DIR / "router_examples.jsonl", FakeEmbedder(),
                        tmp_path / "router.npz")
    assert again.ensure_model() is not None and again.n_examples == router.n_examples
    result = router.evaluate()
    assert result["classifier_accuracy"] > 0.6


def test_forced_modes_skip_the_classifier(tmp_path):
    router = QueryRouter("single", 0.5, None, FakeEmbedder(), tmp_path / "r.npz")
    assert router.decide("Compare everything").mode == "single"


def test_single_step_event_flow(archive):
    engine = engine_for(archive)
    events = list(engine.ask("What does Ritual Adept do?", mode="single"))
    types = [e["type"] for e in events]
    assert types[:3] == ["route", "plan", "sources"] and types[-1] == "done"
    assert events[-1]["answer"].endswith("[1].")
    sources = events[2]["sources"]
    assert sources[0]["n"] == 1 and "Ritual Adept" in sources[0]["text"]


def test_multi_step_plans_follows_up_and_merges(archive):
    llm = FakeLLM(plan=["cult celestial shards", "guardians altar"], follow_up=["religion check altar"])
    engine = engine_for(archive, llm)
    archive.multistep.max_hops = 2
    events = list(engine.ask("Compare what the cult and the guardians want", mode="multi"))
    plans = [e for e in events if e["type"] == "plan"]
    assert plans[0]["planner"] == "llm" and len(plans[0]["queries"]) == 3  # question + 2 sub-queries
    assert plans[1]["hop"] == 2 and plans[1]["queries"] == ["religion check altar"]
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert len({s["id"] for s in sources}) == len(sources)  # no duplicates
    kinds = {s["kind"] for s in sources}
    assert "transcript" in kinds and "md" in kinds


def test_without_llm_sources_still_arrive(archive):
    engine = engine_for(archive, FakeLLM(fail=True))
    events = list(engine.ask("Compare the cult and the guardians and the altar", mode="multi"))
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["planner"] == "rules"
    assert any(e["type"] == "sources" and e["sources"] for e in events)
    assert events[-1]["type"] == "error"


def test_follow_up_question_uses_history(archive):
    engine = engine_for(archive)
    history = [{"role": "user", "content": "Who are the guardians?"}, {"role": "assistant", "content": "..."}]
    assert engine.search_query("What do they protect?", history).startswith("Who are the guardians?")
    assert engine.search_query("What does Ritual Adept let a wizard cast from a spellbook?", history) \
        == "What does Ritual Adept let a wizard cast from a spellbook?"


def test_empty_archive_says_so(tmp_path):
    from craig_transcriber.rag.config import RagConfig
    engine = RagEngine(RagConfig(root=tmp_path, index_dir=tmp_path / "index"), embedder=FakeEmbedder(),
                       reranker=None, llm=FakeLLM(), use_reranker=False)
    events = list(engine.ask("anything"))
    assert events == [{"type": "error", "message": events[0]["message"]}]
    assert "empty" in events[0]["message"]


def test_provider_chain_falls_back():
    class Down(FakeLLM):
        name = "down"

        def stream(self, *a, **k):
            raise LLMUnavailable("no key")
            yield

    chain = ProviderChain([Down(), FakeLLM()])
    assert "".join(chain.stream("s", "p")).startswith("The cult")
    assert chain.last_used == "fake-llm"


def test_chain_starts_over_when_an_answer_breaks_off():
    class BreaksOff(FakeLLM):
        name = "flaky"
        model = "gemini-flash-lite-latest"

        def stream(self, *a, **k):
            yield "The cult wa"
            raise LLMUnavailable("503 UNAVAILABLE")

    pieces = list(ProviderChain([BreaksOff(), FakeLLM()]).stream("s", "p"))
    assert pieces[:2] == ["The cult wa", RESTART] and "".join(pieces[2:]).startswith("The cult wants")

    with pytest.raises(LLMUnavailable, match="gemini-flash-lite-latest: 503"):
        list(ProviderChain([BreaksOff(), FakeLLM(fail=True)]).stream("s", "p"))


def test_gemini_retries_when_google_is_overloaded(monkeypatch):
    from craig_transcriber.llm import gemini

    class Busy(Exception):
        code = 503

    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(kw["model"])
            if len(calls) < 3:
                raise Busy("503 UNAVAILABLE. This model is currently experiencing high demand.")
            return type("R", (), {"text": "answer", "candidates": []})()

    class Pool:
        keys = ["k"]
        def current(self): return "key 1", type("C", (), {"models": Models()})()
        def count_call(self, label): pass
        def mark_exhausted(self, label): raise AssertionError("a busy server is not a used-up quota")

    monkeypatch.setattr(gemini, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(gemini.GeminiProvider, "_config", lambda *a: None)
    provider = gemini.GeminiProvider.__new__(gemini.GeminiProvider)
    provider.model, provider.thinking_budget, provider.pool = "m", None, Pool()  # type: ignore[assignment]
    assert provider.complete("s", "p") == "answer" and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(gemini, "RETRY_DELAYS", (0,))
    with pytest.raises(LLMUnavailable, match="high demand"):
        provider.complete("s", "p")


def test_heuristic_split():
    parts = heuristic_split("Who leads the cult? And what do the guardians want?", 4)
    assert len(parts) == 2

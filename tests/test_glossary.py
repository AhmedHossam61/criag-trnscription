from craig_transcriber.rag.engine import RagEngine
from craig_transcriber.rag.glossary import parse_glossary

from conftest import FakeEmbedder, FakeLLM, FakeReranker

GLOSSARY = """\
# Glossary for llm_correct.py -- notes for whoever edits it.

## Players and their characters
Arnok (player abdoo_) -- often misheard as ارضك / الارض

## Campaign names
master Hareedy هريدي, the most famous trader in Danafor

## Other spellings
# the first spelling is the one to use
Hareedy / Haridy / هريدي
Abo el Wafa / أبو الوفا -- Hareedy's front desk man
Raven / Riven
"""


def test_glossary_keeps_names_and_reads_spellings():
    g = parse_glossary(GLOSSARY)
    assert "notes for whoever" not in g.text and "the first spelling" not in g.text
    assert "## Campaign names" in g.text and "master Hareedy" in g.text
    # only the spellings section counts: "misheard as ارضك / الارض" is not a set of spellings
    assert g.spellings == [["Hareedy", "Haridy", "هريدي"], ["Abo el Wafa", "أبو الوفا"], ["Raven", "Riven"]]


def test_a_question_searches_every_spelling():
    g = parse_glossary(GLOSSARY)
    assert g.expand("Who is Haridy?") == "Who is Haridy? Hareedy هريدي"
    assert g.expand("مين هريدي ده") == "مين هريدي ده Hareedy Haridy"
    assert g.expand("what did abo el wafa say") == "what did abo el wafa say أبو الوفا"
    assert g.expand("the ravenous wolves") == "the ravenous wolves"      # whole words only
    assert g.expand("What is a long rest?") == "What is a long rest?"


def test_engine_searches_and_answers_with_the_glossary(archive):
    path = archive.root / "config" / "glossary.txt"
    path.parent.mkdir()
    path.write_text(GLOSSARY, encoding="utf-8")
    archive.party.glossary = "config/glossary.txt"
    llm = FakeLLM()
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)
    engine.reindex(progress=lambda _: None)
    assert engine._expand("Who is Haridy?").endswith("Hareedy هريدي")
    list(engine.ask("Who is Haridy?", mode="single"))
    system = llm.prompts[-1][0]
    assert "glossary of campaign names" in system and "master Hareedy" in system

    path.unlink()                                   # no glossary: nothing breaks, nothing is added
    assert engine._expand("Who is Haridy?") == "Who is Haridy?"
    list(engine.ask("Who is Haridy?", mode="single"))
    assert "glossary of campaign names" not in llm.prompts[-1][0]

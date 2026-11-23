from craig_transcriber.rag.index import Index, ingest
from craig_transcriber.rag.keyword import BM25, tokenize
from craig_transcriber.rag.retriever import Retriever

from conftest import FakeEmbedder, FakeReranker


def test_tokenize_folds_plurals_and_arabic_letters():
    assert tokenize("The Guardians protect shards") == ["guardian", "protect", "shard"]
    assert tokenize("أرنوك") == tokenize("ارنوك")


def test_bm25_prefers_exact_names():
    bm = BM25(["Arnok swings his axe", "the party rests", "Kael casts a spell at Arnok's foe"])
    top = bm.top("Arnok", 3)
    assert [i for i, _ in top][:2] == [0, 2] and all(s > 0 for _, s in top)
    assert bm.top("nothing matches", 3) == []


def test_ingest_is_incremental(archive):
    embedder = FakeEmbedder()
    index, report = ingest(archive, embedder, progress=lambda _: None)
    assert len(report.added) == 3 and report.passages == len(index.passages) > 0
    assert Index.load(archive.index_dir).vectors.shape == index.vectors.shape

    # nothing changed: nothing is embedded again
    embedder.texts_encoded = 0
    _, report = ingest(archive, embedder, progress=lambda _: None)
    assert report.unchanged == 3 and embedder.texts_encoded == 0

    # one file changed, one removed
    rules = archive.root / "data/documents/rules.txt"
    rules.write_text("A long rest takes eight hours.", encoding="utf-8")
    (archive.root / "data/documents/campaign-notes.md").unlink()
    _, report = ingest(archive, embedder, progress=lambda _: None)
    assert report.updated == ["data/documents/rules.txt"]
    assert report.removed == ["data/documents/campaign-notes.md"]
    assert embedder.texts_encoded == 1


def test_changing_chunk_settings_rebuilds(archive):
    ingest(archive, FakeEmbedder(), progress=lambda _: None)
    archive.chunking.chunk_words = 80
    _, report = ingest(archive, FakeEmbedder(), progress=lambda _: None)
    assert report.rebuilt and len(report.added) == 3


def test_unreadable_file_is_reported_not_fatal(archive):
    (archive.root / "data/documents/broken.pdf").write_bytes(b"not a pdf")
    index, report = ingest(archive, FakeEmbedder(), progress=lambda _: None)
    assert "data/documents/broken.pdf" in report.failed
    assert len(index.docs) == 3


def test_hybrid_search_with_rerank(archive):
    embedder = FakeEmbedder()
    index, _ = ingest(archive, embedder, progress=lambda _: None)
    retriever = Retriever(archive.retrieval, index, embedder, FakeReranker())
    hits = retriever.search("What does Ritual Adept let you cast?", 3)
    assert "Ritual Adept" in hits[0].passage.text
    assert hits[0].rerank is not None and hits[0].dense is not None
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_neighbours_stay_within_a_file(archive):
    index, _ = ingest(archive, FakeEmbedder(), progress=lambda _: None)
    first = index.passages[0]
    before, _after = index.neighbours(first.id)
    assert before is None

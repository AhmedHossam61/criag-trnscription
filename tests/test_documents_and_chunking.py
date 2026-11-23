from craig_transcriber.rag.chunking import chunk_document, chunk_text
from craig_transcriber.rag.config import ChunkingConfig
from craig_transcriber.rag.documents import read_html, read_markdown, read_sections, scan_sources


def test_scan_finds_documents_and_transcripts(archive):
    sources = scan_sources(archive)
    kinds = {s.doc_id: s.kind for s in sources}
    assert kinds == {
        "data/documents/campaign-notes.md": "md",
        "data/documents/rules.txt": "txt",
        "data/transcripts/ABC123/transcript.json": "transcript",
    }
    transcript = next(s for s in sources if s.kind == "transcript")
    assert transcript.title == "Session 2026-08-29 (ABC123)"


def test_markdown_sections_keep_heading_path(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Places\n\nintro\n\n## The Temple\n\naltar here\n\n```\n# not a heading\n```\n\n# NPCs\n\nold man\n")
    sections = read_markdown(md)
    assert [(s.heading, s.text.splitlines()[0]) for s in sections] == [
        ("Places", "intro"), ("Places > The Temple", "altar here"), ("NPCs", "old man"),
    ]


def test_html_strips_scripts_and_splits_on_headings(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<h1>Shards</h1><p>Five shards.</p><script>alert(1)</script><h2>Guardians</h2><p>They watch.</p>")
    sections = read_html(page)
    assert [(s.heading, s.text) for s in sections] == [("Shards", "Five shards."), ("Guardians", "They watch.")]


def test_docx_headings_and_tables(tmp_path):
    import docx
    doc = docx.Document()
    doc.add_heading("Session 19", level=1)
    doc.add_paragraph("We reached the altar.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Kael", "Wizard"
    path = tmp_path / "log.docx"
    doc.save(path)
    from craig_transcriber.rag.documents import SourceFile
    sections = read_sections(SourceFile("log.docx", path, "docx", "log"))
    assert sections[0].heading == "Session 19" and "altar" in sections[0].text
    assert sections[-1].text == "Kael | Wizard"


def test_chunk_text_sizes_and_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), size=30, overlap=5)
    assert all(len(c.split()) <= 35 for c in chunks)
    # every word survives, and each chunk starts with the previous one's tail
    assert set(" ".join(chunks).split()) == set(words)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[:5] == prev.split()[-5:]


def test_chunk_text_keeps_short_paragraphs_together():
    chunks = chunk_text("one two\n\nthree four\n\nfive", size=50, overlap=0)
    assert chunks == ["one two\n\nthree four\n\nfive"]


def test_transcript_windows_split_on_time_and_skip_dropped(archive):
    src = next(s for s in scan_sources(archive) if s.kind == "transcript")
    passages = chunk_document(src, read_sections(src), ChunkingConfig(transcript_window_s=120))
    assert len(passages) == 2  # 10-18 s, then 400 s
    assert passages[0].location == "00:00:10–00:00:18"
    assert "[00:00:15] Arnok: I make a religion check" in passages[0].text
    assert "invented" not in " ".join(p.text for p in passages)

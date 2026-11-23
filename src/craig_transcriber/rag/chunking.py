"""
Cutting sections into passages: the unit that is embedded, searched and cited.

Documents are cut at paragraph boundaries into passages of about
`chunk_words` words, with `overlap_words` carried over so a sentence that
straddles a cut is still found. Transcripts are grouped into windows of
consecutive lines, so a passage reads like a stretch of the session.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import ChunkingConfig
from .documents import Section, SourceFile, fmt_clock


@dataclass
class Passage:
    id: str
    doc_id: str
    ord: int
    text: str
    title: str
    kind: str
    heading: str = ""
    location: str = ""
    start_s: float | None = None
    end_s: float | None = None

    def embed_text(self) -> str:
        """What the embedder and the keyword index see: the text with its context."""
        head = " > ".join(p for p in (self.title, self.heading) if p)
        return f"{head}\n{self.text}" if head else self.text


def passage_id(doc_id: str, ord: int) -> str:
    return f"{hashlib.sha1(doc_id.encode()).hexdigest()[:10]}-{ord:04d}"


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Paragraph-aware windows of about `size` words. Whole paragraphs are packed
    together; one longer than `size` is cut into `size`-word pieces. Each new
    window starts with the last `overlap` words of the one before.
    """
    units: list[list[str]] = []
    for para in re.split(r"\n\s*\n", text):
        words = para.split()
        units += [words[i:i + size] for i in range(0, len(words), size)]

    windows: list[list[list[str]]] = []
    current: list[list[str]] = []
    count = 0
    for unit in units:
        if current and count + len(unit) > size:
            windows.append(current)
            tail = [w for u in current for w in u][-overlap:] if overlap else []
            current, count = ([tail], len(tail)) if tail else ([], 0)
        current.append(unit)
        count += len(unit)
    if current:
        windows.append(current)
    return ["\n\n".join(" ".join(u) for u in window) for window in windows]


def chunk_document(src: SourceFile, sections: list[Section], cfg: ChunkingConfig) -> list[Passage]:
    if src.kind == "transcript":
        return chunk_transcript(src, sections, cfg)
    passages: list[Passage] = []
    for section in sections:
        for text in chunk_text(section.text, cfg.chunk_words, cfg.overlap_words):
            passages.append(Passage(
                id=passage_id(src.doc_id, len(passages)), doc_id=src.doc_id, ord=len(passages),
                text=text, title=src.title, kind=src.kind,
                heading=section.heading, location=section.location,
            ))
    return passages


def chunk_transcript(src: SourceFile, lines: list[Section], cfg: ChunkingConfig) -> list[Passage]:
    passages: list[Passage] = []
    window: list[Section] = []
    words = 0

    def flush() -> None:
        nonlocal window, words
        if not window:
            return
        start, end = window[0].start_s or 0.0, window[-1].end_s or 0.0
        text = "\n".join(f"[{s.location}] {s.speaker}: {s.text.strip()}" for s in window)
        passages.append(Passage(
            id=passage_id(src.doc_id, len(passages)), doc_id=src.doc_id, ord=len(passages),
            text=text, title=src.title, kind=src.kind,
            location=f"{fmt_clock(start)}–{fmt_clock(end)}", start_s=start, end_s=end,
        ))
        window, words = [], 0

    for line in lines:
        n = len(line.text.split())
        if window and (words + n > cfg.transcript_words
                       or (line.end_s or 0) - (window[0].start_s or 0) > cfg.transcript_window_s):
            flush()
        window.append(line)
        words += n
    flush()
    return passages

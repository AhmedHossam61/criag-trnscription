"""
Finding and reading the archive's files.

Every file becomes a list of Sections: a run of text with the heading it sits
under and where it is (a page, a timestamp), so an answer can cite "PHB p. 211"
or "session v3owGZBtlLQA 02:14:33" rather than just a file name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .config import RagConfig

DOCUMENT_TYPES = {".md": "md", ".markdown": "md", ".txt": "txt", ".pdf": "pdf",
                  ".docx": "docx", ".html": "html", ".htm": "html"}


@dataclass
class SourceFile:
    doc_id: str        # path relative to the project root, with forward slashes
    path: Path
    kind: str          # md | txt | pdf | docx | html | transcript
    title: str


@dataclass
class Section:
    text: str
    heading: str = ""                # "Chapter 10 > Spellcasting"
    location: str = ""               # "p. 12", "00:14:03"
    start_s: float | None = None     # transcripts only
    end_s: float | None = None
    speaker: str = ""


def _doc_id(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _title_from_name(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip() or path.name


def scan_sources(config: RagConfig) -> list[SourceFile]:
    """Every file the index should contain, in a stable order."""
    found: list[SourceFile] = []
    docs_dir = config.path(config.sources.documents)
    if docs_dir and docs_dir.exists():
        for path in sorted(docs_dir.rglob("*")):
            kind = DOCUMENT_TYPES.get(path.suffix.lower())
            if kind and path.is_file() and not path.name.startswith((".", "~$")):
                found.append(SourceFile(_doc_id(path, config.root), path, kind, _title_from_name(path)))
    transcripts_dir = config.path(config.sources.transcripts)
    if transcripts_dir and transcripts_dir.exists():
        for path in sorted(transcripts_dir.glob("*/transcript.json")):
            found.append(SourceFile(_doc_id(path, config.root), path, "transcript",
                                    transcript_title(path)))
    return found


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_sections(src: SourceFile) -> list[Section]:
    reader = {
        "md": read_markdown, "txt": read_text, "pdf": read_pdf, "docx": read_docx,
        "html": read_html, "transcript": read_transcript,
    }[src.kind]
    return [s for s in reader(src.path) if s.text.strip()]


def read_text(path: Path) -> list[Section]:
    return [Section(path.read_text(encoding="utf-8", errors="replace"))]


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def read_markdown(path: Path) -> list[Section]:
    """One section per heading, carrying the path of headings above it."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    in_code = False

    def flush() -> None:
        if lines:
            sections.append(Section("\n".join(lines).strip(), " > ".join(t for _, t in stack)))
            lines.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2))]
        else:
            lines.append(line)
    flush()
    return sections


def read_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("reading PDFs needs pypdf: uv pip install pypdf") from exc
    reader = PdfReader(str(path))
    return [Section(page.extract_text() or "", location=f"p. {i}")
            for i, page in enumerate(reader.pages, start=1)]


def read_docx(path: Path) -> list[Section]:
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError("reading .docx needs python-docx: uv pip install python-docx") from exc
    document = docx.Document(str(path))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []

    def flush() -> None:
        if lines:
            sections.append(Section("\n".join(lines), " > ".join(t for _, t in stack)))
            lines.clear()

    for para in document.paragraphs:
        style = (para.style.name if para.style is not None else "") or ""
        m = re.match(r"Heading (\d)", style)
        if m and para.text.strip():
            flush()
            level = int(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, para.text.strip())]
        elif para.text.strip():
            lines.append(para.text)
    flush()
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        sections.append(Section("\n".join(rows), "Table"))
    return sections


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[Section] = []
        self.heading = ""
        self.buf: list[str] = []
        self.skip = 0
        self.in_heading = False
        self.heading_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3"):
            self._flush()
            self.in_heading, self.heading_buf = True, []
        elif tag in self.BLOCK:
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self.skip = max(0, self.skip - 1)
        elif tag in ("h1", "h2", "h3") and self.in_heading:
            self.in_heading = False
            self.heading = " ".join("".join(self.heading_buf).split())

    def handle_data(self, data):
        if self.skip:
            return
        (self.heading_buf if self.in_heading else self.buf).append(data)

    def _flush(self):
        text = re.sub(r"\n\s*\n+", "\n\n", "".join(self.buf)).strip()
        if text:
            self.sections.append(Section(text, self.heading))
        self.buf = []

    def close(self):
        super().close()
        self._flush()


def read_html(path: Path) -> list[Section]:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.sections


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def transcript_title(path: Path) -> str:
    """'Session 2026-08-29 (v3owGZBtlLQA)' from the folder's recording.json."""
    folder = path.parent
    info = {}
    for name in ("recording.json",):
        p = folder / name
        if p.exists():
            try:
                info = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    rec = info.get("Recording") or folder.name
    date = (info.get("Start time") or "")[:10]
    return f"Session {date} ({rec})" if date else f"Session {rec}"


def read_transcript(path: Path) -> list[Section]:
    """One section per spoken line; chunking groups them into time windows."""
    data = json.loads(path.read_text(encoding="utf-8"))
    sections = []
    for seg in data.get("segments", []):
        if seg.get("dropped"):
            continue
        start, end = float(seg["start_s"]), float(seg["end_s"])
        sections.append(Section(seg.get("text", ""), location=fmt_clock(start),
                                start_s=start, end_s=end, speaker=seg.get("speaker", "")))
    return sections

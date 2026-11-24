"""
The campaign glossary, config/glossary.txt. `scribe correct` sends it to Gemini
so names get spelled right; the archive uses it too:

- the answering model reads it as background, so it knows who an NPC is even
  when the passages found don't say, and
- its "## Other spellings" section lists names written more than one way, one
  name per line ("Hareedy / Haridy / هريدي"): a question using any of them
  also searches for the rest, as the roster's aliases do for the party.

Lines starting with a single '#' are notes for whoever edits the file and are
left out; '## ' lines are section headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .party import add_spellings

SPELLINGS_SECTION = "other spellings"
MAX_CHARS = 8000   # a glossary is meant to be a page or two; anything past this is cut


@dataclass
class Glossary:
    text: str = ""                                               # what the LLM reads
    spellings: list[list[str]] = field(default_factory=list)     # each: one name's spellings

    def expand(self, query: str) -> str:
        return add_spellings(query, self.spellings)


def parse_glossary(raw: str) -> Glossary:
    kept: list[str] = []
    spellings: list[list[str]] = []
    section = ""
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#") and not s.startswith("##"):
            continue
        if s.startswith("##"):
            section = s.lstrip("#").strip().lower()
        elif section.startswith(SPELLINGS_SECTION):
            names = [n.strip() for n in s.split("--")[0].split("/") if n.strip()]
            if len(names) > 1:
                spellings.append(names)
        kept.append(line.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return Glossary(text[:MAX_CHARS], spellings)


class GlossaryFile:
    """The glossary, re-read whenever the file changes; empty when there is none."""

    def __init__(self, path: Path | None):
        self.path = path
        self._cache: tuple[float, Glossary] = (-1.0, Glossary())

    @property
    def current(self) -> Glossary:
        path = self.path
        mtime = path.stat().st_mtime if path and path.is_file() else 0.0
        if mtime != self._cache[0]:
            glossary = parse_glossary(path.read_text(encoding="utf-8")) if path and mtime else Glossary()
            self._cache = (mtime, glossary)
        return self._cache[1]

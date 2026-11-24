"""
The party's characters: the roster in config/party.toml, and what the lore
site shows on each character's page -- their sheet, how much they spoke in
each session, and where the others mention them.

The roster also helps the archive: questions that name a character search
for all of their spellings (Raddis finds the transcript lines saying رادش),
and the LLM is told who is who.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from ..transcription.filters import normalize_arabic
from .config import RagConfig
from .documents import DOCUMENT_TYPES, SourceFile, fmt_clock, read_sections, transcript_title

ROLES = ("main", "companion")
PORTRAIT_TYPES = (".png", ".jpg", ".jpeg", ".webp")
MAX_MENTIONS_PER_SESSION = 120


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"


@dataclass
class Character:
    name: str
    role: str = "main"
    full_name: str = ""
    player: str = ""
    title: str = ""
    summary: str = ""
    sheet: str = ""
    portrait: str = ""
    aliases: list[str] = field(default_factory=list)
    id: str = ""

    @property
    def terms(self) -> list[str]:
        """Every way the character is written, longest first."""
        seen, out = set(), []
        for t in [self.full_name, self.name, *self.aliases]:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return sorted(out, key=len, reverse=True)


def load_roster(path: Path | None) -> list[Character]:
    if path is None or not path.exists():
        return []
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Character)} - {"id"}
    characters: list[Character] = []
    for n, entry in enumerate(data.get("character", []), start=1):
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"{path.name}: character {n}: unknown setting(s) {', '.join(sorted(unknown))}")
        if not str(entry.get("name", "")).strip():
            raise ValueError(f"{path.name}: character {n} has no name")
        ch = Character(**entry)
        if ch.role not in ROLES:
            raise ValueError(f"{path.name}: {ch.name}: role must be one of {', '.join(ROLES)}")
        ch.id = slug(ch.name)
        if any(c.id == ch.id for c in characters):
            raise ValueError(f"{path.name}: two characters are called {ch.name}")
        characters.append(ch)
    return characters


_AR = "ء-ي"
# Arabic glues words onto the front of a name: و / ف / ب / ل / ك, the article, يا.
_AR_PREFIX = rf"(?<![{_AR}])(?:[وفبلك]?ال|[وفبلك]|يا)?"


def _matcher(terms: list[str]) -> re.Pattern | None:
    """
    Whole words: a Latin name between non-letters; an Arabic one with at most a
    prefix and nothing after it, so the nickname نوب finds "يا نوب" and "ونوب"
    but not نوبة (a watch shift) or جنوب (south).
    """
    parts = []
    for term in terms:
        t = re.escape(normalize_arabic(term))
        parts.append(rf"(?<![^\W\d_]){t}(?![^\W\d_])" if term.isascii() else rf"{_AR_PREFIX}{t}(?![{_AR}])")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


class Party:
    def __init__(self, config: RagConfig):
        self.config = config
        self.roster_path = config.path(config.party.roster)
        self._roster: tuple[float, list[Character]] = (-1.0, [])
        self._files: dict[Path, tuple[float, object]] = {}

    # -- roster -------------------------------------------------------------

    @property
    def characters(self) -> list[Character]:
        """The roster, re-read whenever the file changes."""
        path = self.roster_path
        mtime = path.stat().st_mtime if path and path.exists() else 0.0
        if mtime != self._roster[0]:
            self._roster = (mtime, load_roster(path))
        return self._roster[1]

    def get(self, character_id: str) -> Character | None:
        return next((c for c in self.characters if c.id == character_id), None)

    def describe(self) -> str:
        """Who is who, for the LLM."""
        lines = []
        for role, label in (("main", "Player characters"), ("companion", "Companions")):
            people = []
            for c in self.characters:
                if c.role != role:
                    continue
                bits = [b for b in (c.full_name if c.full_name != c.name else "", c.title,
                                    f"played by {c.player}" if c.player else "") if b]
                other = [t for t in c.terms if t not in (c.name, c.full_name)]
                if other:
                    bits.append("also written " + ", ".join(other))
                people.append(c.name + (f" ({'; '.join(bits)})" if bits else ""))
            if people:
                lines.append(f"{label}: " + "; ".join(people) + ".")
        return "\n".join(lines)

    def expand_query(self, query: str) -> str:
        """Add a character's other spellings to a keyword search that names them."""
        text = normalize_arabic(query)
        extra = []
        for c in self.characters:
            m = _matcher(c.terms)
            if m and m.search(text):
                extra += [t for t in c.terms if not re.search(re.escape(normalize_arabic(t)), text, re.I)]
        return f"{query} {' '.join(extra)}" if extra else query

    # -- files --------------------------------------------------------------

    def _cached(self, path: Path, load):
        mtime = path.stat().st_mtime
        hit = self._files.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        value = load(path)
        self._files[path] = (mtime, value)
        return value

    def asset_folder(self, c: Character) -> Path | None:
        """assets/<character>/: a folder named after them (any case, or one of their Latin aliases)."""
        assets = self.config.root / "assets"
        if not assets.is_dir():
            return None
        names = {slug(t) for t in c.terms if t.isascii()} | {c.id}
        return next((d for d in sorted(assets.iterdir()) if d.is_dir() and slug(d.name) in names), None)

    def portrait(self, c: Character) -> Path | None:
        """The `portrait` setting, or else assets/<character>/portrait.*, or else its first image."""
        if c.portrait:
            p = self.config.path(c.portrait)
            return p if p and p.is_file() else None
        folder = self.asset_folder(c)
        if folder is None:
            return None
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PORTRAIT_TYPES)
        return next((p for p in images if p.stem.lower() == "portrait"), images[0] if images else None)

    def sheet_path(self, c: Character) -> Path | None:
        """The `sheet` setting, or else a file in data/documents/characters named after them."""
        if c.sheet:
            p = self.config.path(c.sheet)
            return p if p and p.is_file() else None
        docs = self.config.path(self.config.sources.documents)
        folder = docs / "characters" if docs else None
        if folder is None or not folder.is_dir():
            return None
        names = {slug(t) for t in c.terms if t.isascii()} | {c.id}
        return next((p for p in sorted(folder.iterdir())
                     if p.is_file() and p.suffix.lower() in DOCUMENT_TYPES and slug(p.stem) in names), None)

    def sheet(self, c: Character) -> dict | None:
        path = self.sheet_path(c)
        if path is None:
            return None
        try:
            label = path.resolve().relative_to(self.config.root.resolve()).as_posix()
        except ValueError:
            label = path.name
        kind = DOCUMENT_TYPES.get(path.suffix.lower())
        if kind is None:
            return {"path": label, "error": f"{path.suffix} files can't be read"}

        def load(p: Path):
            return [{"heading": s.heading, "location": s.location, "text": s.text.strip()}
                    for s in read_sections(SourceFile(label, p, kind, p.stem))]
        try:
            sections = self._cached(path, load)
        except Exception as exc:  # noqa: BLE001 -- show the problem on the page instead of failing it
            return {"path": label, "error": f"{type(exc).__name__}: {exc}"}
        return {"path": label, "title": path.stem, "sections": sections}

    def _sessions(self) -> list[dict]:
        folder = self.config.path(self.config.sources.transcripts)
        if folder is None or not folder.exists():
            return []
        sessions = []
        for path in folder.glob("*/transcript.json"):
            def load(p: Path):
                data = json.loads(p.read_text(encoding="utf-8"))
                return [s for s in data.get("segments", []) if not s.get("dropped")]
            info_file = path.parent / "recording.json"
            info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
            sessions.append({"id": path.parent.name, "title": transcript_title(path),
                             "date": (info.get("Start time") or "")[:10], "segments": self._cached(path, load)})
        return sorted(sessions, key=lambda s: (s["date"], s["id"]))

    # -- what a page shows --------------------------------------------------

    def appearances(self, c: Character) -> list[dict]:
        """Per session: how much they spoke, and the lines where others name them."""
        match = _matcher(c.terms)
        own = {t.lower() for t in c.terms}   # speaker labels, e.g. "Vaylor" from before a rename
        out = []
        for session in self._sessions():
            spoken, seconds, mentions = 0, 0.0, []
            for seg in session["segments"]:
                speaker = seg.get("speaker", "")
                if speaker.lower() in own:
                    spoken += 1
                    seconds += max(0.0, float(seg["end_s"]) - float(seg["start_s"]))
                elif match and match.search(normalize_arabic(seg.get("text", ""))):
                    mentions.append({"start_s": float(seg["start_s"]), "clock": fmt_clock(float(seg["start_s"])),
                                     "speaker": speaker, "text": seg.get("text", "")})
            if spoken or mentions:
                out.append({"id": session["id"], "title": session["title"], "date": session["date"],
                            "spoken_lines": spoken, "spoken_minutes": round(seconds / 60, 1),
                            "mention_count": len(mentions), "mentions": mentions[:MAX_MENTIONS_PER_SESSION]})
        return out

    def in_documents(self, c: Character, passages) -> list[dict]:
        """Documents (not their own sheet, not transcripts) that name them."""
        match = _matcher(c.terms)
        sheet = self.sheet_path(c)
        sheet_id = sheet.resolve().relative_to(self.config.root.resolve()).as_posix() \
            if sheet and sheet.resolve().is_relative_to(self.config.root.resolve()) else None
        docs: dict[str, dict] = {}
        for p in passages:
            if p.kind == "transcript" or p.doc_id == sheet_id or match is None:
                continue
            text = normalize_arabic(p.text)
            m = match.search(text)
            if not m:
                continue
            doc = docs.setdefault(p.doc_id, {"path": p.doc_id, "title": p.title, "count": 0, "snippets": []})
            doc["count"] += 1
            if len(doc["snippets"]) < 3:
                a, b = max(0, m.start() - 140), min(len(text), m.end() + 180)
                doc["snippets"].append({"passage_id": p.id, "where": " · ".join(x for x in (p.heading, p.location) if x),
                                        "text": ("…" if a else "") + text[a:b].strip() + ("…" if b < len(text) else "")})
        return sorted(docs.values(), key=lambda d: -d["count"])

    def card(self, c: Character) -> dict:
        sessions = self.appearances(c)
        return {
            "id": c.id, "name": c.name, "full_name": c.full_name, "role": c.role, "player": c.player,
            "title": c.title, "summary": c.summary, "aliases": c.aliases,
            "portrait": f"/api/party/{c.id}/portrait" if self.portrait(c) else None,
            "has_sheet": self.sheet_path(c) is not None,
            "sessions": len(sessions), "mentions": sum(s["mention_count"] for s in sessions),
            "spoken_lines": sum(s["spoken_lines"] for s in sessions),
        }

    def page(self, c: Character, passages) -> dict:
        return self.card(c) | {"sheet": self.sheet(c), "appearances": self.appearances(c),
                               "documents": self.in_documents(c, passages)}

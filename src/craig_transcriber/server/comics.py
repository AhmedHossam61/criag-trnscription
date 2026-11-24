"""
The party's comics: one folder each in assets/comics, pages read in name order
(01.png, 02.png, ... 10.png). An optional comic.toml in the folder adds:

    title = "Two Minutes"            # else the folder name
    summary = "What happens in it."
    date = "2026-11-24"
    characters = ["Kael", "Arnok"]   # names or aliases from config/party.toml
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..rag.party import Character, slug

PAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
SETTINGS = {"title", "summary", "date", "characters"}


def natural_key(path: Path) -> list:
    """2.png before 10.png, and "page.jpeg" before "page (1).jpeg" (how WhatsApp numbers copies)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)] + [path.suffix]


@dataclass
class Comic:
    id: str
    title: str
    folder: Path
    pages: list[Path]
    summary: str = ""
    date: str = ""
    characters: list[str] = field(default_factory=list)


def load_comics(root: Path | None) -> list[Comic]:
    """Every comic folder with at least one page, oldest first. ValueError on a bad comic.toml."""
    if root is None or not root.is_dir():
        return []
    comics: list[Comic] = []
    for folder in sorted((d for d in root.iterdir() if d.is_dir()), key=natural_key):
        pages = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PAGE_TYPES),
                       key=natural_key)
        if not pages:
            continue
        info: dict = {}
        settings = folder / "comic.toml"
        if settings.exists():
            try:
                with settings.open("rb") as fh:
                    info = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"{folder.name}/comic.toml: {exc}") from exc
            unknown = set(info) - SETTINGS
            if unknown:
                raise ValueError(f"{folder.name}/comic.toml: unknown setting(s) {', '.join(sorted(unknown))}")
        title = str(info.get("title") or folder.name)
        comic_id = slug(title)
        while any(c.id == comic_id for c in comics):
            comic_id += "-2"
        comics.append(Comic(comic_id, title, folder, pages, str(info.get("summary", "")),
                            str(info.get("date", "")), [str(n) for n in info.get("characters", [])]))
    return sorted(comics, key=lambda c: (c.date or "9999", natural_key(c.folder)))


def cast(comic: Comic, roster: list[Character]) -> list[dict]:
    """The comic's characters, linked to their pages when the roster knows them."""
    found = []
    for name in comic.characters:
        key = slug(name)
        c = next((c for c in roster if key == c.id or key in {slug(t) for t in c.terms if t.isascii()}), None)
        found.append({"name": c.name if c else name, "id": c.id if c else None})
    return found

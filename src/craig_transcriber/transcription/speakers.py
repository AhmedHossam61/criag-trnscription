"""Speaker maps: Discord username (or SPEAKER_00) = display name."""

from __future__ import annotations

from pathlib import Path

from ..paths import CONFIG_DIR
from .settings import SKIP_NAMES


def default_speaker_map() -> Path | None:
    """config/speaker_map.txt, when it exists."""
    path = CONFIG_DIR / "speaker_map.txt"
    return path if path.exists() else None


def load_speaker_map(path: Path) -> dict[str, str]:
    """Parse a 'discord_username = display name' file. Case-insensitive keys."""
    mapping: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def map_speaker(name: str, speaker_map: dict[str, str]) -> str | None:
    """Look up a Discord username: exact match first, then glob patterns (jockie*, megadon*)."""
    import fnmatch
    uname = name.lower()
    renamed = speaker_map.get(uname)
    if renamed is None:
        for pat, target in speaker_map.items():
            if any(ch in pat for ch in "*?[") and fnmatch.fnmatchcase(uname, pat):
                return target
    return renamed


def is_skipped(name: str | None) -> bool:
    """True for a map target that hides the speaker (SKIP, -, IGNORE)."""
    return bool(name) and name.upper() in SKIP_NAMES

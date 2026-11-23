"""
Language handling.

Chunks are cut at natural silence boundaries within a single speaker's
track, so a chunk is usually one utterance. That lines up well with
sentence-level code-switching ("...ok so let me explain this properly").
Word-level switching inside one Arabic sentence is a different problem and
is handled by the model itself, not by tagging -- which is exactly what
QwenCleo is trained for.
"""

from __future__ import annotations

import re


_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def script_mix(text: str) -> tuple[float, float]:
    """Return (latin_fraction, arabic_fraction) over letters only."""
    latin = len(_LATIN.findall(text))
    arabic = len(_ARABIC.findall(text))
    total = latin + arabic
    if total == 0:
        return 0.0, 0.0
    return latin / total, arabic / total


def label_script(text: str) -> str:
    """Classify the output by script, independent of what tag was requested."""
    latin, arabic = script_mix(text)
    if latin == 0 and arabic == 0:
        return ""
    if latin >= 0.85:
        return "en"
    if arabic >= 0.85:
        return "ar"
    return "mixed"

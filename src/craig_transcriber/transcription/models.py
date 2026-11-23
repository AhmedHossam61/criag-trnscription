"""Data model: tracks, chunks of speech, and transcribed segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    """One speaker's audio track from a Craig recording."""
    index: int
    speaker: str
    path: Path
    offset_s: float = 0.0  # start offset within the session timeline


@dataclass
class Chunk:
    """A contiguous window of real speech from one track."""
    track_index: int
    speaker: str
    chunk_id: int
    start_s: float  # absolute, on the session timeline
    end_s: float

    @property
    def key(self) -> str:
        return f"{self.track_index}:{self.chunk_id}"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Segment:
    """A transcribed chunk."""
    speaker: str
    start_s: float
    end_s: float
    text: str
    language: str = ""
    track_index: int = 0
    chunk_id: int = 0
    dropped: bool = False
    drop_reason: str = ""
    original_text: str = ""  # set only in outputs, when llm_correct.py changed the text

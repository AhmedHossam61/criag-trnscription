"""
Audio: decoding to memory-mapped PCM, the zero-run VAD, cutting chunks,
and the level / speech checks that keep noise away from the model.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import Chunk, Track
from .settings import MAX_CHUNK_S, MIN_GAP_S, MIN_SPEECH_S, PAD_S, SR


def _run(cmd: Sequence[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(map(str, cmd))}\n"
            f"{proc.stderr.decode('utf-8', 'replace')[:2000]}"
        )
    return proc.stdout


# ---------------------------------------------------------------------------
# Decoding to a memory-mappable scratch file
# ---------------------------------------------------------------------------


def decode_to_pcm(src: Path, dest: Path) -> Path:
    """
    Decode a track to 16 kHz mono 16-bit PCM WAV.

    Eight hours at 16 kHz mono is ~920 MB on disk but only a few MB resident
    once memory-mapped, so we never hold a whole track in RAM.
    """
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-i", str(src),
        "-ac", "1", "-ar", str(SR),
        "-c:a", "pcm_s16le",
        str(dest),
    ])
    return dest


def wav_data_span(path: Path) -> tuple[int, int]:
    """
    Locate the `data` chunk by walking the RIFF structure.

    Do NOT assume a 44-byte header: ffmpeg writes a LIST/INFO chunk naming the
    encoder, which pushes audio to byte 78 on most builds. Hardcoding 44 reads
    the ASCII metadata as samples -- which the zero-run VAD then reports as a
    phantom burst of speech at t=0. Header size also varies between ffmpeg
    builds, so it must be parsed rather than guessed.
    """
    with path.open("rb") as fh:
        if fh.read(4) != b"RIFF":
            raise ValueError(f"{path} is not a RIFF file")
        fh.seek(8)
        if fh.read(4) != b"WAVE":
            raise ValueError(f"{path} is not a WAVE file")
        while True:
            header = fh.read(8)
            if len(header) < 8:
                raise ValueError(f"no data chunk found in {path}")
            chunk_id = header[:4]
            size = int.from_bytes(header[4:8], "little")
            if chunk_id == b"data":
                return fh.tell(), size
            fh.seek(size + (size & 1), 1)  # chunks are word-aligned


def open_pcm(path: Path) -> np.memmap:
    offset, n_bytes = wav_data_span(path)
    return np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(n_bytes // 2,))


def close_pcm(pcm: np.memmap | None) -> None:
    """
    Release the mapping explicitly.

    Windows keeps a memory-mapped file locked, so the scratch directory cannot
    be deleted while a mapping is live -- leaving gigabytes of PCM behind in
    %TEMP%. `del` alone is not enough if anything else still holds a reference.
    """
    if pcm is None:
        return
    try:
        pcm._mmap.close()  # noqa: SLF001 -- no public API for this
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Stage 1: free VAD from Craig's exact-zero gaps
# ---------------------------------------------------------------------------

def find_voiced_runs(pcm: np.memmap, block: int = 1 << 22) -> list[tuple[int, int]]:
    """
    Locate runs of non-silent samples, scanning block-wise so an 8-hour track
    never lands in RAM.

    Craig writes literal zeros where Discord sent no packets, so `sample != 0`
    is an exact speech mask rather than an energy heuristic. Returns a list of
    [start, end) sample indices.
    """
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for base in range(0, len(pcm), block):
        chunk = np.asarray(pcm[base:base + block])
        voiced = chunk != 0
        if not voiced.any():
            if run_start is not None:
                runs.append((run_start, base))
                run_start = None
            continue

        # Transition indices within this block.
        padded = np.concatenate(([voiced[0]], voiced))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        marks = [0, *edges.tolist(), len(voiced)]

        for i in range(len(marks) - 1):
            lo, hi = marks[i], marks[i + 1]
            if lo == hi:
                continue
            if voiced[lo]:
                if run_start is None:
                    run_start = base + lo
            else:
                if run_start is not None:
                    runs.append((run_start, base + lo))
                    run_start = None

    if run_start is not None:
        runs.append((run_start, len(pcm)))
    return runs


def merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Bridge runs separated by less than `max_gap` samples."""
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def quietest_split(pcm: np.memmap, lo: int, hi: int, frame: int = 1600) -> int:
    """
    Find the lowest-energy frame inside [lo, hi) so a long monologue is cut
    between words rather than mid-syllable.
    """
    span = np.asarray(pcm[lo:hi], dtype=np.float32)
    n = len(span) // frame
    if n < 2:
        return (lo + hi) // 2
    energy = (span[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


def build_chunks(pcm: np.memmap, track: Track, min_gap_s: float = MIN_GAP_S) -> list[Chunk]:
    """Turn one track's raw samples into transcribable speech windows."""
    runs = find_voiced_runs(pcm)
    runs = merge_runs(runs, max_gap=int(min_gap_s * SR))
    runs = [(a, b) for a, b in runs if (b - a) >= int(MIN_SPEECH_S * SR)]

    pad = int(PAD_S * SR)
    max_len = int(MAX_CHUNK_S * SR)
    chunks: list[Chunk] = []

    for start, end in runs:
        start = max(0, start - pad)
        end = min(len(pcm), end + pad)

        # Split anything over the model's comfortable window, cutting at the
        # quietest point near the limit instead of a hard slice.
        pieces: list[tuple[int, int]] = []
        cursor = start
        while end - cursor > max_len:
            search_lo = cursor + int(max_len * 0.75)
            search_hi = cursor + max_len
            cut = quietest_split(pcm, search_lo, search_hi)
            pieces.append((cursor, cut))
            cursor = cut
        pieces.append((cursor, end))

        for lo, hi in pieces:
            chunks.append(
                Chunk(
                    track_index=track.index,
                    speaker=track.speaker,
                    chunk_id=len(chunks),
                    start_s=track.offset_s + lo / SR,
                    end_s=track.offset_s + hi / SR,
                )
            )
    return chunks


def read_chunk_audio(pcm: np.memmap, track: Track, chunk: Chunk) -> np.ndarray:
    """
    Cut one chunk out of the track as float32 in [-1, 1].

    Same scaling soundfile uses when reading a 16-bit WAV, so the model sees
    exactly what it saw when chunks went through temp files.
    """
    lo = int(round((chunk.start_s - track.offset_s) * SR))
    hi = int(round((chunk.end_s - track.offset_s) * SR))
    data = np.asarray(pcm[max(0, lo):min(len(pcm), hi)], dtype=np.float32)
    return data / 32768.0


def write_wav(audio: np.ndarray, dest: Path) -> Path:
    """Write float32 [-1, 1] audio as a 16 kHz mono 16-bit WAV."""
    data = np.clip(np.round(audio * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data.tobytes())
    return dest


# ---------------------------------------------------------------------------
# Level and speech checks
# ---------------------------------------------------------------------------


def chunk_peak_db(pcm: np.memmap, track: Track, chunk: Chunk) -> float:
    """Loudest sample in the chunk, in dBFS (0 = full scale)."""
    lo = max(0, int(round((chunk.start_s - track.offset_s) * SR)))
    hi = min(len(pcm), int(round((chunk.end_s - track.offset_s) * SR)))
    peak = int(np.abs(np.asarray(pcm[lo:hi], dtype=np.int32)).max()) if hi > lo else 0
    return 20 * np.log10(max(peak, 1) / 32768.0)


class SpeechCheck:
    """
    Silero VAD (a ~2 MB speech detector, CPU) measuring how much of a clip is
    actually someone talking. Laughs, breaths and lip smacks are loud enough
    to pass the level gate, and QwenCleo writes a stock sentence for them.
    """

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad
        self.torch = torch
        self.model = load_silero_vad()

    def speech_ratio(self, audio: np.ndarray, threshold: float = 0.5) -> float:
        frame = 512  # what silero takes at 16 kHz
        n = len(audio) // frame
        if n == 0:
            return 0.0
        self.model.reset_states()
        voiced = 0
        with self.torch.no_grad():
            for i in range(n):
                x = self.torch.from_numpy(np.ascontiguousarray(audio[i * frame:(i + 1) * frame]))
                voiced += self.model(x, SR).item() >= threshold
        return voiced / n

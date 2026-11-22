#!/usr/bin/env python3
"""
craig_transcribe.py -- Arabic transcription for Craig Bot Discord recordings.

Craig records one FLAC per speaker, all aligned to a common t=0, with exact
digital silence in the gaps (Discord sends no packets when a user isn't
talking). This script exploits both facts:

  * per-speaker tracks  -> diarization for free, no pyannote needed
  * exact-zero gaps     -> a free, perfect VAD; only real speech hits the GPU

Designed for an 8 GB VRAM GPU (RTX 3050) and 8-hour recordings.

Usage:
    python craig_transcribe.py recording.zip -o out/ --backend qwencleo
    python craig_transcribe.py recording.zip -o out/ --backend cohere --limit 300
    python craig_transcribe.py out/ --list-backends

Resume is automatic: re-running skips segments already in segments.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree

import numpy as np

SR = 16000  # every ASR model here wants 16 kHz mono


def _force_utf8_console() -> None:
    """
    Craig metadata carries Arabic (guild names, channel names, usernames) and
    transcripts obviously do. Windows CMD defaults to cp1252/cp437, where
    printing Arabic raises UnicodeEncodeError and kills the run. Files are
    always written as UTF-8; this fixes stdout/stderr too.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A gap must be at least this long to count as a segment boundary. Shorter
# pauses are kept inside one chunk. 0.45 s only bridged packet jitter, which
# cut sentences at every breath ("في حد من ال" / "النائتس بيلمحك") and fed the
# model 1-2 s fragments it could only guess at. 1.5 s keeps a speaker's
# sentence together; MAX_CHUNK_S still bounds the window.
# Changing this changes chunk ids, so an output folder records the value
# and refuses to resume under a different one.
MIN_GAP_S = 1.5

# Never emit a chunk longer than this. Whisper's encoder is hard-capped at 30s;
# Qwen and Cohere handle more but accuracy and VRAM both degrade.
MAX_CHUNK_S = 28.0

# Ignore speech islands shorter than this -- almost always a cough, a mic bump,
# or a single "aha" that the model will hallucinate a sentence out of.
MIN_SPEECH_S = 0.35

# Padding kept either side of a speech island so plosives aren't clipped.
PAD_S = 0.20

# Generation budget per chunk: TOKEN_FLOOR + TOKENS_PER_S * longest chunk in the
# batch, capped at TOKEN_CEIL. Hallucination loops otherwise run to the cap on
# every bad chunk. 15 tok/s sits above what screen()'s text-length ceiling lets
# through, so anything this truncates would have been dropped anyway.
# Skip chunks whose loudest sample is below this. Discord transmits mic hiss
# and breathing as packets, so they survive the zero-run VAD, and QwenCleo
# turns them into stock sentences ("...project في الـ web development").
# Measured on a real session: hallucinated chunks peak around -45 dBFS, real
# speech from the quietest mic around -29. -40 catches most of the former
# without touching the latter.
MIN_PEAK_DBFS = -40.0

# Skip short chunks in which the speech detector (Silero VAD) hears someone
# talking for less than this share of the clip. Checked against the audio of
# one 5-minute stretch: real short lines ("RITUAL!", "ايه ده", "يا جدع") were
# 36-69% speech; lines invented on laughs and breaths were 0-32%, and 0.25
# removed 4 of 6 of them without losing a real one. Only chunks up to
# VAD_MAX_CHUNK_S are checked -- every invented line was shorter -- which keeps
# the check to a minute or two per session on the CPU.
MIN_SPEECH_RATIO = 0.25
VAD_MAX_CHUNK_S = 6.0

TOKENS_PER_S = 15
TOKEN_FLOOR = 24
TOKEN_CEIL = 256


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Craig archive parsing
# ---------------------------------------------------------------------------

def _run(cmd: Sequence[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(map(str, cmd))}\n"
            f"{proc.stderr.decode('utf-8', 'replace')[:2000]}"
        )
    return proc.stdout


def check_env(backend_name: str) -> int:
    """Report on ffmpeg, CUDA, VRAM and the selected backend's imports."""
    problems = 0
    print("Core:")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"  {tool:<10} {path or 'MISSING -- install ffmpeg'}")
        problems += path is None
    print(f"  numpy      {np.__version__}")
    print(f"  python     {sys.version.split()[0]}")

    print("\nGPU:")
    try:
        import torch
        print(f"  torch      {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            print(f"  device     {props.name} ({vram:.1f} GB, sm_{props.major}{props.minor})")
            if vram < 7.0:
                print("  ! under 7 GB free -- lower --batch-size to 1")
        else:
            print("  ! CUDA not available; this will run on CPU and be far too slow")
            problems += 1
    except ImportError:
        print("  torch      not installed")
        problems += 1

    print(f"\nBackend '{backend_name}':")
    needed = {
        "qwencleo": ["qwencleo_asr", "qwen_asr"],
        "qwen3asr": ["transformers"],
        "cohere": ["transformers"],
        "faster-whisper": ["faster_whisper"],
    }[backend_name]
    for module in needed:
        try:
            mod = __import__(module)
            print(f"  {module:<14} {getattr(mod, '__version__', 'ok')}")
        except ImportError as exc:
            print(f"  {module:<14} MISSING ({exc})")
            problems += 1

    if backend_name == "cohere":
        try:
            from huggingface_hub import whoami
            print(f"  hf auth        {whoami()['name']}")
        except Exception:  # noqa: BLE001
            print("  hf auth        not logged in -- weights are gated, run `hf auth login`")
            problems += 1

    print("\n" + ("All good." if not problems else f"{problems} problem(s) above."))
    return problems


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"error: {tool} not found on PATH. Install ffmpeg first.")


def extract_archive(src: Path, workdir: Path) -> Path:
    """Return a directory containing the unpacked Craig recording."""
    if not src.exists():
        sys.exit(f"error: no such file or directory: {src}")
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        sys.exit(f"error: expected a .zip or a directory, got {src}")
    dest = workdir / "unpacked"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        # raw.dat is Craig's undecoded packet dump (hundreds of MB) -- unused.
        members = [m for m in zf.namelist() if Path(m).name != "raw.dat"]
        zf.extractall(dest, members=members)
    return dest


def parse_sesx_offsets(root: Path) -> dict[str, float]:
    """
    Read per-clip start offsets from Craig's Adobe Audition session file.

    Craig normally pads every track with leading silence so all tracks share
    t=0, in which case every startPoint is 0. But some export modes trim the
    silence and encode the offset here instead, so we honour it.
    """
    offsets: dict[str, float] = {}
    sesx_files = list(root.glob("*.sesx"))
    if not sesx_files:
        return offsets
    try:
        tree = ElementTree.parse(sesx_files[0])
    except ElementTree.ParseError:
        return offsets

    session = tree.getroot().find("session")
    sample_rate = float(session.get("sampleRate", 48000)) if session is not None else 48000.0

    for clip in tree.getroot().iter("audioClip"):
        name = clip.get("name")
        if not name:
            continue
        try:
            offsets[name] = float(clip.get("startPoint", 0)) / sample_rate
        except (TypeError, ValueError):
            offsets[name] = 0.0
    return offsets


def discover_tracks(root: Path) -> list[Track]:
    """Find the per-speaker FLAC files and recover speaker names."""
    audio: list[Path] = []
    for pattern in ("*_data/*.flac", "*_data/*.wav", "*.flac", "*.wav"):
        audio = sorted(root.glob(pattern))
        if audio:
            break
    if not audio:
        sys.exit(f"error: no audio tracks found under {root}")

    offsets = parse_sesx_offsets(root)
    tracks: list[Track] = []
    for path in audio:
        # Craig names files "<index>-<discord username>.flac"
        m = re.match(r"^(\d+)-(.+)$", path.stem)
        if m:
            index, speaker = int(m.group(1)), m.group(2)
        else:
            index, speaker = len(tracks) + 1, path.stem
        tracks.append(
            Track(
                index=index,
                speaker=speaker,
                path=path,
                offset_s=offsets.get(path.name, 0.0),
            )
        )
    tracks.sort(key=lambda t: t.index)
    return tracks


def read_recording_info(root: Path) -> dict[str, str]:
    info_path = root / "info.txt"
    if not info_path.exists():
        return {}
    info: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
        # First line is "Recording <id>" with no colon.
        m = re.match(r"^Recording\s+(\S+)\s*$", line)
        if m:
            info["Recording"] = m.group(1)
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value:
                info[key] = value
    return info


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


def token_budget(duration_s: float) -> int:
    return int(min(TOKEN_CEIL, TOKEN_FLOOR + TOKENS_PER_S * duration_s))


def make_batches(chunks: list[Chunk], max_count: int, max_seconds: float) -> list[list[Chunk]]:
    """
    Group chunks of similar length so little of each batch is padding.

    A batch is padded to its longest chunk, so mixing a 1 s "تمام" with a 28 s
    monologue wastes most of the GPU time. Sorted longest-first, a batch grows
    until its padded size (count x longest) would pass max_seconds, or its
    count would pass max_count. Longest-first also means an out-of-memory
    batch shows up in the first minute, not hours in.
    """
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    for chunk in sorted(chunks, key=lambda c: c.duration_s, reverse=True):
        if current and (
            len(current) + 1 > max_count
            or (len(current) + 1) * current[0].duration_s > max_seconds
        ):
            batches.append(current)
            current = []
        current.append(chunk)
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# Stage 3: hallucination filtering
# ---------------------------------------------------------------------------

# Phrases these models emit when handed near-silence or noise. The zero-run
# gate removes most of the opportunity, but low-level mic hiss still slips
# through and both Whisper and Cohere are documented as hallucinating on it.
HALLUCINATION_PATTERNS = [
    r"ترجمة\s+نانسي",
    r"اشترك\s+في\s+القناة",
    r"لا\s+تنسى\s+الاشتراك",
    r"شكرا\s+(لكم\s+)?على\s+المشاهدة",
    r"subtitles?\s+by",
    r"amara\.org",
    r"thanks?\s+for\s+watching",
    # "I love you, you love me" (Arabic Barney dub). Comes out of near-silence.
    # Match both orderings but only when BOTH halves appear together, so
    # a genuine standalone "بحبك" from a player isn't flagged.
    r"بحبك[^\n]{0,30}بحبني",
    r"بحبني[^\n]{0,30}بحبك",
    # QwenCleo's stock sentences on noise -- podcast training-data leftovers.
    r"بنعمل\s+project",
    r"web\s+(development|design)",
    r"engineering\s+college",
    r"cairo\s+university",
    # Stock sentences from short clips of laughter/breath. Each came from
    # every player in one session (39x for the first), so none is real speech.
    r"مش\s+متأكد[ةه]?\s+[إا]ن\s+[أا]نا\s+بحبك",
    r"^بصراح[ةه]\s+[أا]نا\s+مش\s+عارف\s+([أا]عمل\s+[إا]يه|[إا]يه\s+اللي\s+بيحصل\s+ده)\W*$",
    r"^(طب\s+)?[أا]نا\s+بحب\s+[أا]عمل\s+(ازاي|[إا]يه\s+في\s+ال)",
    r"بحبك\s+يا\s+حبيبي",
    # Chinese/Japanese/Korean script: nobody in the session speaks it.
    r"[぀-ヿ㐀-鿿가-힯]",
    r"^\W*$",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]

# Arabic diacritics and tatweel -- strip before comparing repeated tokens.
_ARABIC_NOISE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")


def normalize_arabic(text: str) -> str:
    text = _ARABIC_NOISE.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip()


def repetition_ratio(text: str) -> float:
    """Fraction of tokens that are duplicates. Loops score near 1.0."""
    tokens = normalize_arabic(text).split()
    if len(tokens) < 6:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def loop_tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", normalize_arabic(text)).split()


def has_tandem_repeat(tokens: list[str], max_n: int = 6) -> bool:
    """
    True if a phrase of 1..max_n words is said back to back many times
    ("X Y X Y X Y X Y"): what a decoder stuck in a loop produces. Single
    words need 6 in a row, since "لا لا لا" and "ماشي ماشي ماشي" are real.
    """
    for n in range(1, max_n + 1):
        need = 6 if n == 1 else 4
        for i in range(len(tokens) - n * need + 1):
            gram = tokens[i:i + n]
            reps, j = 1, i + n
            while tokens[j:j + n] == gram:
                reps += 1
                j += n
            if reps >= need:
                return True
    return False


def has_dominant_ngram(tokens: list[str], min_cover: float = 0.7) -> bool:
    """
    True if one 2-4 word phrase, said 3+ times, makes up most of the line.
    Occurrences are counted without overlap, so "لا لا لا لا" is two "لا لا",
    not three. 0.7 keeps a real line the model stuttered on ("لا انت انت ما
    كنتش انت ما كنتش هتقول انت ما كنتش هتقول بتحورها", 0.64).
    """
    for n in (2, 3, 4):
        if len(tokens) < 3 * n:
            continue
        counts: dict[tuple[str, ...], int] = {}
        next_free: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if i < next_free.get(gram, 0):
                continue
            counts[gram] = counts.get(gram, 0) + 1
            next_free[gram] = i + n
        top = max(counts.values())
        if top >= 3 and top * n / len(tokens) >= min_cover:
            return True
    return False


def has_ngram_loop(text: str) -> bool:
    """
    Short-phrase loops that repetition_ratio misses.

    Repetition alone is not enough: Egyptian conversation repeats itself
    ("ريت شوال ريت شوال يعني ايه ... مش ريت شوال", "هل انت هل انت نمت").
    Counting any bigram said 3 times as a loop threw away 14 s of real
    dialogue in a 5-minute test, so only back-to-back runs or a phrase that
    dominates the line count.
    """
    tokens = loop_tokens(text)
    return has_tandem_repeat(tokens) or has_dominant_ngram(tokens)


def screen(text: str, chunk: Chunk) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    stripped = text.strip()
    if not stripped:
        return False, "empty"

    for pattern in _HALLUCINATION_RE:
        if pattern.search(stripped):
            return False, "boilerplate-hallucination"

    if repetition_ratio(stripped) > 0.82:
        return False, "token-loop"

    if has_ngram_loop(stripped):
        return False, "ngram-loop"

    # A model that produces far more text than the audio could carry is
    # confabulating. English runs denser per second than Arabic, so use the
    # looser of the two limits when the output contains Latin script.
    latin, _ = script_mix(stripped)
    ceiling = 30.0 if latin > 0.3 else 25.0
    if len(stripped) > ceiling * max(chunk.duration_s, 1.0) * 1.6:
        return False, "text-exceeds-audio-duration"

    return True, ""


# ---------------------------------------------------------------------------
# Language handling
#
# Chunks are cut at natural silence boundaries within a single speaker's
# track, so a chunk is usually one utterance. That lines up well with
# sentence-level code-switching ("...ok so let me explain this properly").
# Word-level switching inside one Arabic sentence is a different problem and
# is handled by the model itself, not by tagging -- which is exactly what
# QwenCleo is trained for.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 2: ASR backends
#
# Each backend loads lazily and returns (text, language_tag) per chunk.
# Only one is ever resident, which is what keeps this inside 8 GB.
#
# `language` is one of:
#   "ar"   -- force the Arabic tag. Correct when Arabic is the matrix language
#             and English appears as embedded words or short phrases.
#   "en"   -- force the English tag.
#   "auto" -- decide per chunk. Uses the model's own language ID where it has
#             one; falls back to a dual-tag scoring pass where it doesn't.
# ---------------------------------------------------------------------------

class Backend:
    name = "base"
    default_batch = 1             # max chunks per batch
    default_batch_seconds = 60.0  # max padded audio per batch (count x longest)
    supports_hotwords = False

    def __init__(self, language: str = "auto", hotwords: str | None = None):
        self.language = language
        self.hotwords = hotwords
        self.beams = 1  # beam search width; backends that support it read this in load()

    def load(self) -> None:
        raise NotImplementedError

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        """
        Return one (text, language_tag) pair per input.

        `audio` is float32 16 kHz mono in [-1, 1]; `max_new_tokens` is sized
        to the longest chunk in the batch.
        """
        raise NotImplementedError

    def unload(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class QwenCleoBackend(Backend):
    """
    mohammedaly22/QwenCleo-ASR -- Qwen3-ASR-1.7B fine-tuned on Egyptian
    podcast speech AND Arabic/English code-switching. The only model here
    trained end-to-end for mixed speech: it keeps embedded English in Latin
    script (`React`, `already`, `soft skills`) instead of transliterating it
    into Arabic letters. Best choice for a bilingual Discord call.

        pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        pip install qwencleo-asr --no-deps
        pip install "qwen-asr>=0.0.6" numpy soundfile huggingface_hub
    """
    name = "qwencleo"
    default_batch = 16
    default_batch_seconds = 120.0

    _TAGS = {"ar": "Arabic", "en": "English", "auto": None}

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwencleo_asr import DEFAULT_MODEL_ID
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        # Load the underlying Qwen3-ASR model directly rather than through the
        # QwenCleoASR wrapper: the wrapper's transcribe() loops over its inputs
        # one at a time, so it never actually batches. Same checkpoint, dtype
        # and device the wrapper would use.
        self.model = Qwen3ASRModel.from_pretrained(
            DEFAULT_MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=-1,  # batches are sized by make_batches()
        )
        if self.beams > 1:
            # qwen_asr calls generate() with max_new_tokens only; extra
            # generation options go straight through to transformers.
            import functools
            hf_model = self.model.model
            hf_model.generate = functools.partial(hf_model.generate, num_beams=self.beams)

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        self.model.max_new_tokens = max_new_tokens
        results = self.model.transcribe(
            [(a, SR) for a in audio], language=self._TAGS[self.language]
        )

        out: list[tuple[str, str]] = []
        for r in results:
            text = getattr(r, "text", str(r))
            # The model emits its own language tag; fall back to script.
            detected = getattr(r, "language", None)
            out.append((text, (detected or label_script(text) or self.language)))
        return out


class Qwen3ASRBackend(Backend):
    """
    Qwen/Qwen3-ASR-1.7B-hf -- QwenCleo's base. Genuinely multilingual with
    built-in language identification, so `--language auto` is reliable here.
    Weaker on Egyptian dialect than the fine-tune, but it is the backend to
    reach for if QwenCleo produces something odd, and it is the only one
    that accepts a hotword glossary.

        pip install "transformers>=5.13.0" torch soundfile librosa accelerate
    """
    name = "qwen3asr"
    default_batch = 16
    default_batch_seconds = 120.0
    supports_hotwords = True

    _TAGS = {"ar": "Arabic", "en": "English", "auto": None}

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        ).eval()

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        import torch
        tag = self._TAGS[self.language]

        # apply_transcription_request is fed file paths, the input form its
        # docs show; the files only live for this batch.
        with tempfile.TemporaryDirectory(prefix="craig_batch_") as tmp:
            paths = [str(write_wav(a, Path(tmp) / f"{i}.wav")) for i, a in enumerate(audio)]
            request = {"audio": paths}
            if tag is not None:
                request["language"] = [tag] * len(paths)
            if self.hotwords:
                # Biases decoding toward your jargon so it stays in Latin script.
                request["prompt"] = self.hotwords

            inputs = self.processor.apply_transcription_request(**request).to(
                self.model.device, self.model.dtype
            )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]

        parsed = self.processor.decode(generated, return_format="parsed")
        out: list[tuple[str, str]] = []
        for item in parsed:
            text = item.get("transcription", "")
            lang = (item.get("language") or "").lower()[:2] or label_script(text)
            out.append((text, lang))
        return out


class CohereArabicBackend(Backend):
    """
    CohereLabs/cohere-transcribe-arabic-07-2026 -- 2B FastConformer encoder,
    lowest WER of any open model on the Open Universal Arabic ASR
    Leaderboard (25.87 avg) and by far the fastest option here.

    Caveat that matters for a bilingual call: the model card states it has no
    automatic language detection and shows inconsistent performance on
    code-switched audio. With `--language auto` this backend therefore runs
    each chunk under BOTH tags and keeps whichever the decoder scored higher.
    That doubles its cost -- still fast, but it is a workaround for a real
    limitation, not a feature. Prefer QwenCleo if English shows up often.

    Gated: accept the terms on the model page, then `hf auth login`.

        pip install "transformers>=5.4.0" torch soundfile librosa sentencepiece protobuf accelerate
    """
    name = "cohere"
    default_batch = 16
    default_batch_seconds = 120.0

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        model_id = "CohereLabs/cohere-transcribe-arabic-07-2026"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        ).eval()

    def _decode_with_score(self, audio, tag: str, max_new_tokens: int) -> list[tuple[str, float]]:
        """Transcribe under one language tag, returning mean token log-prob."""
        import torch

        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", language=tag)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded = self.processor.decode(result.sequences, skip_special_tokens=True)
        texts = [decoded] if isinstance(decoded, str) else list(decoded)

        try:
            transition = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            )
            finite = torch.isfinite(transition)
            totals = (transition * finite).sum(dim=1)
            counts = finite.sum(dim=1).clamp(min=1)
            scores = (totals / counts).tolist()
        except Exception:  # noqa: BLE001 -- scoring is best-effort
            scores = [float("-inf")] * len(texts)

        return list(zip(texts, scores))

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        if self.language in ("ar", "en"):
            return [
                (text, self.language)
                for text, _ in self._decode_with_score(audio, self.language, max_new_tokens)
            ]

        # auto: score both tags and keep the more confident transcription.
        arabic = self._decode_with_score(audio, "ar", max_new_tokens)
        english = self._decode_with_score(audio, "en", max_new_tokens)

        out: list[tuple[str, str]] = []
        for (ar_text, ar_score), (en_text, en_score) in zip(arabic, english):
            if en_score > ar_score:
                out.append((en_text, "en"))
            else:
                out.append((ar_text, "ar"))
        return out


class FasterWhisperBackend(Backend):
    """
    Baseline only. Whisper large-v3 flattens Egyptian into MSA and is the
    worst of these on code-switching -- it tends to transliterate English
    into Arabic script rather than keeping it in Latin. Useful for
    comparison, not production.

        pip install faster-whisper
    """
    name = "faster-whisper"
    default_batch = 1

    def __init__(self, language: str = "auto", hotwords: str | None = None,
                 model_size: str = "large-v3"):
        super().__init__(language, hotwords)
        self.model_size = model_size

    def load(self) -> None:
        from faster_whisper import WhisperModel
        # int8_float16 keeps large-v3 near 3 GB, leaving headroom on an 8 GB card.
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        # Whisper decodes one input at a time and has its own length limits,
        # so max_new_tokens is not applied here.
        out: list[tuple[str, str]] = []
        for samples in audio:
            segments, info = self.model.transcribe(
                samples,
                language=None if self.language == "auto" else self.language,
                beam_size=5,
                vad_filter=False,          # our zero-run gate already did this
                condition_on_previous_text=False,  # stops hallucination cascades
                initial_prompt=self.hotwords or None,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            out.append((text, getattr(info, "language", "") or label_script(text)))
        return out


BACKENDS: dict[str, type[Backend]] = {
    "qwencleo": QwenCleoBackend,
    "qwen3asr": Qwen3ASRBackend,
    "cohere": CohereArabicBackend,
    "faster-whisper": FasterWhisperBackend,
}

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def fmt_timestamp(seconds: float, comma: bool = True) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


# A sentence at least STOCK_MIN_WORDS long that comes out word for word from
# STOCK_MIN_SPEAKERS different people, STOCK_MIN_COUNT+ times, is the model's
# stock filler, not speech. Catches new ones the patterns above don't know.
STOCK_MIN_WORDS = 5
STOCK_MIN_SPEAKERS = 3
STOCK_MIN_COUNT = 4


def drop_stock_phrases(segments: list[Segment]) -> int:
    """Mark session-wide repeated sentences as dropped; return how many."""
    groups: dict[str, list[Segment]] = {}
    for seg in segments:
        if seg.dropped:
            continue
        norm = re.sub(r"[^\w\s]", "", normalize_arabic(seg.text)).strip()
        if len(norm.split()) >= STOCK_MIN_WORDS:
            groups.setdefault(norm, []).append(seg)
    dropped = 0
    for segs in groups.values():
        if len(segs) >= STOCK_MIN_COUNT and len({s.speaker for s in segs}) >= STOCK_MIN_SPEAKERS:
            for seg in segs:
                seg.dropped, seg.drop_reason = True, "stock-phrase"
                dropped += 1
    return dropped


def refresh_drops(segments: list[Segment]) -> None:
    """
    Re-apply screen() and the stock-phrase rule, so filter changes made since
    the transcript was recorded take effect without re-running the model.
    Chunks are reconstructed from each segment's own timing.

    Text dropped by an older, stricter filter comes back too. Segments the
    level gate dropped ("too-quiet") were judged on audio, not text, so they
    stay dropped.
    """
    rescreened = restored = 0
    for seg in segments:
        if seg.dropped and seg.drop_reason in ("too-quiet", "no-speech"):
            continue
        chunk = Chunk(
            track_index=seg.track_index, speaker=seg.speaker,
            chunk_id=seg.chunk_id, start_s=seg.start_s, end_s=seg.end_s,
        )
        keep, reason = screen(seg.text, chunk)
        if not keep and not seg.dropped:
            rescreened += 1
        elif keep and seg.dropped:
            restored += 1
        seg.dropped, seg.drop_reason = not keep, reason
    if rescreened:
        print(f"  re-screen dropped {rescreened} previously-kept segment(s)")
    if restored:
        print(f"  re-screen restored {restored} segment(s) an older filter had dropped")
    stock = drop_stock_phrases(segments)
    if stock:
        print(f"  dropped {stock} segment(s) repeating a stock phrase across speakers")


CORRECTIONS_FILE = "corrections.json"


def apply_corrections(segments: list[Segment], outdir: Path) -> tuple[list[Segment], int]:
    """
    Overlay llm_correct.py's fixes, keyed "<track_index>:<chunk_id>", on copies
    of the segments. segments.jsonl is never changed, so the raw model output
    stays available and a new correction pass can always start over.
    """
    path = outdir / CORRECTIONS_FILE
    if not path.exists():
        return segments, 0
    fixes = json.loads(path.read_text(encoding="utf-8"))
    out, applied = [], 0
    for seg in segments:
        fix = fixes.get(f"{seg.track_index}:{seg.chunk_id}")
        if fix is None or seg.dropped:
            out.append(seg)
            continue
        applied += 1
        if fix.get("drop"):
            out.append(replace(seg, dropped=True, drop_reason="llm-no-speech"))
        else:
            out.append(replace(seg, text=fix["text"], original_text=seg.text))
    return out, applied


def write_transcript_txt(kept: list[Segment], path: Path, info: dict[str, str]) -> None:
    """Readable transcript with consecutive turns by one speaker merged."""
    with path.open("w", encoding="utf-8") as fh:
        if info:
            for key, value in info.items():
                fh.write(f"# {key}: {value}\n")
            fh.write("\n")
        last_speaker = None
        for seg in kept:
            if seg.speaker != last_speaker:
                fh.write(f"\n[{fmt_timestamp(seg.start_s, comma=False)}] {seg.speaker}:\n")
                last_speaker = seg.speaker
            # Mark non-Arabic turns so you can find the code-switches by eye.
            script = label_script(seg.text)
            tag = f"[{script}] " if script in ("en", "mixed") else ""
            fh.write(f"  {tag}{seg.text}\n")


def write_outputs(segments: list[Segment], outdir: Path, info: dict[str, str]) -> None:
    refresh_drops(segments)
    raw_kept = sorted((s for s in segments if not s.dropped), key=lambda s: (s.start_s, s.speaker))
    segments, corrected = apply_corrections(segments, outdir)
    if corrected:
        print(f"  applied {corrected} correction(s) from {CORRECTIONS_FILE}")
        write_transcript_txt(raw_kept, outdir / "transcript-uncorrected.txt", info)

    kept = sorted((s for s in segments if not s.dropped), key=lambda s: (s.start_s, s.speaker))

    (outdir / "transcript.json").write_text(
        json.dumps(
            {"recording": info, "segments": [asdict(s) for s in kept]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (outdir / "transcript.srt").open("w", encoding="utf-8") as fh:
        for i, seg in enumerate(kept, start=1):
            fh.write(f"{i}\n")
            fh.write(f"{fmt_timestamp(seg.start_s)} --> {fmt_timestamp(seg.end_s)}\n")
            fh.write(f"{seg.speaker}: {seg.text}\n\n")

    write_transcript_txt(kept, outdir / "transcript.txt", info)

    dropped = [s for s in segments if s.dropped]
    if dropped:
        with (outdir / "dropped.txt").open("w", encoding="utf-8") as fh:
            for seg in sorted(dropped, key=lambda s: s.start_s):
                fh.write(
                    f"[{fmt_timestamp(seg.start_s, comma=False)}] {seg.speaker} "
                    f"({seg.drop_reason}): {seg.text}\n"
                )


def language_report(segments: list[Segment]) -> str:
    """Summarise the Arabic/English split so you can sanity-check the tagging."""
    kept = [s for s in segments if not s.dropped]
    if not kept:
        return ""
    counts = {"ar": 0, "en": 0, "mixed": 0, "?": 0}
    duration = {"ar": 0.0, "en": 0.0, "mixed": 0.0, "?": 0.0}
    for seg in kept:
        key = label_script(seg.text) or "?"
        counts[key] += 1
        duration[key] += seg.end_s - seg.start_s
    total = sum(duration.values()) or 1.0
    parts = [
        f"{key}: {counts[key]} segs / {duration[key] / 60:.1f} min ({100 * duration[key] / total:.0f}%)"
        for key in ("ar", "en", "mixed", "?")
        if counts[key]
    ]
    return "  " + "   ".join(parts)


def transcribe_with_fallback(backend: Backend, batch: list[Chunk],
                             audio: list[np.ndarray]) -> list[tuple[str, str] | None]:
    """
    Transcribe one batch; on failure (usually CUDA out-of-memory) retry it in
    halves. A chunk that still fails on its own comes back as None.
    """
    try:
        return backend.transcribe_batch(audio, token_budget(max(c.duration_s for c in batch)))
    except Exception as exc:  # noqa: BLE001 -- never lose 8h to one bad chunk
        backend.unload()  # release VRAM in case this was an OOM
        if len(batch) == 1:
            print(f"\n  ! chunk {batch[0].key} failed ({exc}); will retry on re-run")
            return [None]
        mid = len(batch) // 2
        print(f"\n  ! batch of {len(batch)} failed ({type(exc).__name__}); retrying in halves")
        return (transcribe_with_fallback(backend, batch[:mid], audio[:mid])
                + transcribe_with_fallback(backend, batch[mid:], audio[mid:]))


def run_batches(backend: Backend, batches: list[list[Chunk]], load_batch,
                jsonl: Path, segments: list[Segment]) -> int:
    """
    Transcribe every batch, appending each result to `segments` and to
    segments.jsonl as it lands. `load_batch(batch)` returns the batch's audio;
    it runs on a worker thread so the next batch is cut while the GPU works.
    Returns the number of chunks that failed (left out of the jsonl, so a
    re-run retries them).
    """
    pending_speech = sum(c.duration_s for b in batches for c in b)
    processed_audio = 0.0
    failed = 0
    start = time.time()

    with jsonl.open("a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=1) as prefetch:
        upcoming = prefetch.submit(load_batch, batches[0]) if batches else None
        for i, batch in enumerate(batches):
            audio = upcoming.result()
            # Cut the next batch's audio on the CPU while the GPU works on this one.
            if i + 1 < len(batches):
                upcoming = prefetch.submit(load_batch, batches[i + 1])

            results = transcribe_with_fallback(backend, batch, audio)

            for chunk, result in zip(batch, results):
                if result is None:
                    failed += 1
                    continue
                text, lang = result
                keep, reason = screen(text, chunk)
                seg = Segment(
                    speaker=chunk.speaker,
                    start_s=round(chunk.start_s, 3),
                    end_s=round(chunk.end_s, 3),
                    text=text.strip(),
                    language=lang or label_script(text),
                    track_index=chunk.track_index,
                    chunk_id=chunk.chunk_id,
                    dropped=not keep,
                    drop_reason=reason,
                )
                segments.append(seg)
                sink.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")
            sink.flush()

            processed_audio += sum(c.duration_s for c in batch)
            elapsed = time.time() - start
            rtf = processed_audio / elapsed if elapsed else 0
            remaining = max(0.0, pending_speech - processed_audio) / rtf if rtf else 0
            pct = 100 * processed_audio / pending_speech if pending_speech else 100
            print(
                f"  {pct:5.1f}%  {processed_audio / 60:7.1f}/{pending_speech / 60:.1f} min  "
                f"{rtf:5.1f}x realtime  eta {remaining / 60:5.1f} min",
                end="\r", flush=True,
            )
        print()
    return failed


def load_done(jsonl: Path) -> dict[str, Segment]:
    """Read already-transcribed segments so a re-run resumes where it stopped."""
    done: dict[str, Segment] = {}
    if not jsonl.exists():
        return done
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from a hard kill
            seg = Segment(**record)
            done[f"{seg.track_index}:{seg.chunk_id}"] = seg
    return done


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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


# Folders written before the gap setting was recorded all used this value.
LEGACY_MIN_GAP_S = 0.45


def check_recording_match(outdir: Path, info: dict[str, str], min_gap_s: float) -> None:
    """
    Refuse to resume into an outdir that holds a different recording, or the
    same recording cut into chunks with a different --min-gap.

    Resume keys are "<track_index>:<chunk_id>", which every Craig recording
    reuses, and which point at different audio once the gap setting changes.
    Either mismatch would silently skip audio and keep text attached to the
    wrong chunks.
    """
    marker = outdir / "recording.json"
    current = info.get("Recording", "")
    has_segments = (outdir / "segments.jsonl").exists()
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        previous = saved.get("Recording", "")
        if previous and current and previous != current:
            sys.exit(
                f"error: {outdir} already holds recording {previous}, but the input is "
                f"{current}. Use a different -o directory."
            )
        saved_gap = float(saved.get("Min gap s", LEGACY_MIN_GAP_S))
        if has_segments and saved_gap != min_gap_s:
            sys.exit(
                f"error: {outdir} was transcribed with --min-gap {saved_gap:g}, but this run "
                f"uses {min_gap_s:g}, which cuts the audio into different chunks.\n"
                f"  To finish or rewrite that folder: add --min-gap {saved_gap:g}\n"
                f"  To transcribe with the new setting: use a different -o directory."
            )
    elif has_segments:
        sys.exit(
            f"error: {outdir} has a segments.jsonl from an unknown recording. "
            f"Use a different -o directory (or delete it to start over)."
        )
    marker.write_text(
        json.dumps({**info, "Min gap s": f"{min_gap_s:g}"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def plan(root: Path, scratch: Path, limit_s: float | None,
         speaker_map: dict[str, str] | None = None,
         min_peak_db: float = MIN_PEAK_DBFS,
         min_gap_s: float = MIN_GAP_S,
         start_s: float | None = None,
         min_speech_ratio: float = MIN_SPEECH_RATIO,
         ) -> tuple[list[Track], dict[int, list[Chunk]], dict[int, Path], dict[str, str]]:
    """
    Decode and segment every track. Chunks quieter than min_peak_db, and short
    chunks the speech detector hears little speech in, are left out of the
    plan and returned separately as {key: reason}, so chunk ids (and resume)
    are unaffected by either threshold.
    """
    tracks = discover_tracks(root)
    if speaker_map:
        kept: list[Track] = []
        for t in tracks:
            renamed = map_speaker(t.speaker, speaker_map)
            if renamed is None:
                kept.append(t)
                continue
            if renamed.upper() in {"SKIP", "-", "IGNORE"}:
                print(f"  skipping track {t.index} ({t.speaker}) per speaker map")
                continue
            t.speaker = renamed
            kept.append(t)
        tracks = kept
    pcm_paths: dict[int, Path] = {}
    plans: dict[int, list[Chunk]] = {}
    skipped: dict[str, str] = {}
    vad = SpeechCheck() if min_speech_ratio > 0 else None

    for track in tracks:
        pcm_path = scratch / f"track{track.index}.wav"
        decode_to_pcm(track.path, pcm_path)
        pcm_paths[track.index] = pcm_path

        pcm = open_pcm(pcm_path)
        try:
            chunks = build_chunks(pcm, track, min_gap_s)
            # Filtered after build_chunks, so chunk ids match a full run.
            if limit_s is not None:
                chunks = [c for c in chunks if c.end_s <= limit_s]
            if start_s is not None:
                chunks = [c for c in chunks if c.start_s >= start_s]
            loud = [c for c in chunks if chunk_peak_db(pcm, track, c) >= min_peak_db]
            voiced = loud
            if vad is not None:
                voiced = [
                    c for c in loud
                    if c.duration_s > VAD_MAX_CHUNK_S
                    or vad.speech_ratio(read_chunk_audio(pcm, track, c)) >= min_speech_ratio
                ]
        finally:
            close_pcm(pcm)
        loud_keys = {c.key for c in loud}
        voiced_keys = {c.key for c in voiced}
        for c in chunks:
            if c.key not in loud_keys:
                skipped[c.key] = "too-quiet"
            elif c.key not in voiced_keys:
                skipped[c.key] = "no-speech"
        plans[track.index] = voiced

        speech = sum(c.duration_s for c in voiced)
        total = len(pcm) / SR
        pct = (100 * speech / total) if total else 0.0
        notes = []
        if len(loud) < len(chunks):
            notes.append(f"{len(chunks) - len(loud)} too quiet")
        if len(voiced) < len(loud):
            notes.append(f"{len(loud) - len(voiced)} without speech")
        print(
            f"  track {track.index:>2}  {track.speaker:<24} "
            f"{total / 3600:6.2f}h audio -> {speech / 60:8.2f} min speech "
            f"({pct:5.1f}%) in {len(voiced)} chunks"
            + (", " + ", ".join(notes) if notes else "")
        )
    return tracks, plans, pcm_paths, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe Craig Bot Discord recordings (Arabic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        help="Craig .zip or an already-extracted directory",
    )
    parser.add_argument("-o", "--outdir", type=Path, default=Path("transcripts"))
    parser.add_argument(
        "--backend", default="qwencleo", choices=sorted(BACKENDS),
        help="ASR model to use (default: qwencleo)",
    )
    parser.add_argument(
        "--language", default="auto", choices=["ar", "en", "auto"],
        help=(
            "ar: force the Arabic tag -- right when Arabic is the matrix language "
            "and English appears only as embedded words. en: force English. "
            "auto (default): decide per chunk, for recordings with whole English "
            "sentences mixed in."
        ),
    )
    parser.add_argument(
        "--hotwords", type=Path, default=None, metavar="FILE",
        help="text file of domain vocabulary to bias decoding (qwen3asr, faster-whisper)",
    )
    parser.add_argument(
        "--speaker-map", type=Path, default=None, metavar="FILE",
        help=(
            "text file mapping Discord usernames to display names, one "
            "'username = display name' per line. Blank lines and '#' comments "
            "are ignored. Match is case-insensitive on the Discord username."
        ),
    )
    parser.add_argument(
        "--min-gap", type=float, default=MIN_GAP_S, metavar="SECONDS",
        help=(
            f"a pause must be at least this long to split a speaker's speech "
            f"into separate chunks (default {MIN_GAP_S:g}). Folders made before "
            f"this option existed used {LEGACY_MIN_GAP_S:g}; pass that to resume them."
        ),
    )
    parser.add_argument(
        "--min-peak-db", type=float, default=MIN_PEAK_DBFS, metavar="DBFS",
        help=(
            f"skip chunks whose loudest sample is below this level "
            f"(default {MIN_PEAK_DBFS:g}). These are mic hiss and breathing that "
            f"the model turns into invented sentences. Raise toward -35 for more "
            f"aggressive filtering; -120 disables it."
        ),
    )
    parser.add_argument(
        "--min-speech-ratio", type=float, default=MIN_SPEECH_RATIO, metavar="R",
        help=(
            f"skip chunks up to {VAD_MAX_CHUNK_S:g} s in which the speech detector hears "
            f"talking for less than this share of the clip (default {MIN_SPEECH_RATIO:g}). "
            f"Catches laughs and breaths the model writes sentences for. "
            f"0 disables it; around 0.35 is stricter but may cut real one-word replies."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="max chunks per batch (default: backend's, 16 for qwencleo)",
    )
    parser.add_argument(
        "--batch-seconds", type=float, default=None, metavar="SECONDS",
        help=(
            "max padded audio per batch, i.e. chunk count x longest chunk "
            "(default: backend's, 120 for qwencleo). Lower it on out-of-memory "
            "errors, raise it if VRAM is to spare."
        ),
    )
    parser.add_argument(
        "--limit", type=float, default=None, metavar="SECONDS",
        help=(
            "only transcribe up to N seconds into the recording -- use this to "
            "benchmark backends. With --start, the end of the window."
        ),
    )
    parser.add_argument(
        "--start", type=float, default=None, metavar="SECONDS",
        help=(
            "only transcribe from N seconds into the recording, e.g. --start 11700 "
            "--limit 12000 for 03:15:00-03:20:00. For comparing settings on a short "
            "stretch; use a separate -o folder for each setting."
        ),
    )
    parser.add_argument(
        "--beams", type=int, default=1, metavar="N",
        help=(
            "beam search width (qwencleo only). 1 (default) = greedy, fastest. "
            "4 considers several candidate sentences and keeps the most likely, "
            "usually a bit more accurate, roughly 2-3x slower. Batch defaults "
            "shrink automatically to fit VRAM."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="segment and report only; load no model",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, default=None, metavar="DIR",
        help=(
            "where to put decoded PCM (default: system temp). Budget ~950 MB "
            "per speaker per 8 hours -- point this at a roomy drive, e.g. "
            "D:\\craig-scratch, if your system drive is tight."
        ),
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--check-env", action="store_true",
        help="verify ffmpeg, CUDA, VRAM and backend imports, then exit",
    )
    parser.add_argument(
        "--rewrite-outputs", action="store_true",
        help=(
            "re-apply the hallucination filter to an existing "
            "out/segments.jsonl and regenerate transcript.txt/.srt/.json. "
            "No zip, no model, no GPU. Use after updating filter patterns."
        ),
    )
    args = parser.parse_args()

    if args.check_env:
        sys.exit(1 if check_env(args.backend) else 0)
    if args.rewrite_outputs:
        jsonl = args.outdir / "segments.jsonl"
        if not jsonl.exists():
            sys.exit(f"error: {jsonl} not found; nothing to rewrite")
        done = load_done(jsonl)
        print(f"Loaded {len(done)} segments from {jsonl}")
        if args.speaker_map:
            # Rename segments transcribed before the map had an entry for them.
            speaker_map = load_speaker_map(args.speaker_map)
            for key, seg in list(done.items()):
                renamed = map_speaker(seg.speaker, speaker_map)
                if renamed and renamed.upper() in {"SKIP", "-", "IGNORE"}:
                    del done[key]  # hidden from the transcript; segments.jsonl keeps it
                elif renamed:
                    seg.speaker = renamed
        marker = args.outdir / "recording.json"
        info = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        write_outputs(list(done.values()), args.outdir, info)
        print(f"Wrote transcript.txt / .srt / .json under {args.outdir}")
        return
    if args.input is None:
        parser.error("the following arguments are required: input")

    require_ffmpeg()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="craig_", dir=args.scratch_dir))

    try:
        root = extract_archive(args.input, scratch)
        info = read_recording_info(root)
        if info:
            print("Recording:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        if not args.dry_run:
            check_recording_match(args.outdir, info, args.min_gap)

        speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else None
        if speaker_map:
            print(f"\nSpeaker map: {len(speaker_map)} rename(s) loaded from {args.speaker_map}")

        print("\nSegmenting tracks (zero-run VAD):")
        t0 = time.time()
        tracks, plans, pcm_paths, skipped = plan(
            root, scratch, args.limit, speaker_map=speaker_map,
            min_peak_db=args.min_peak_db, min_gap_s=args.min_gap, start_s=args.start,
            min_speech_ratio=args.min_speech_ratio,
        )
        total_chunks = sum(len(c) for c in plans.values())
        total_speech = sum(c.duration_s for cs in plans.values() for c in cs)
        print(
            f"\n  {total_chunks} chunks / {total_speech / 60:.1f} min of speech "
            f"(segmented in {time.time() - t0:.1f}s)"
        )

        if args.dry_run:
            print("\nDry run -- no model loaded.")
            return

        hotwords = None
        if args.hotwords:
            hotwords = args.hotwords.read_text(encoding="utf-8").strip()

        backend = BACKENDS[args.backend](language=args.language, hotwords=hotwords)
        backend.beams = args.beams
        if hotwords and not backend.supports_hotwords:
            print(f"  note: backend '{backend.name}' ignores --hotwords")
        if args.beams > 1 and backend.name != "qwencleo":
            print(f"  note: --beams is only used by qwencleo; '{backend.name}' ignores it")
        # Each beam is another copy of the batch on the GPU.
        batch_size = args.batch_size or max(1, backend.default_batch // args.beams)
        batch_seconds = args.batch_seconds or backend.default_batch_seconds / args.beams

        jsonl = args.outdir / "segments.jsonl"
        done = load_done(jsonl)
        if done:
            print(f"\nResuming: {len(done)} segments already transcribed.")
        # Segments transcribed before the level gate or speech check existed
        # (or under looser thresholds) are hidden from the outputs;
        # segments.jsonl keeps them.
        for key, seg in done.items():
            if key in skipped and not seg.dropped:
                seg.dropped, seg.drop_reason = True, skipped[key]

        segments: list[Segment] = list(done.values())
        pending = [c for cs in plans.values() for c in cs if c.key not in done]
        if pending:
            print(
                f"\nLoading backend '{backend.name}' (batch up to {batch_size} chunks / "
                f"{batch_seconds:g}s padded audio, language={args.language})..."
            )
            backend.load()
        batches = make_batches(pending, batch_size, batch_seconds)
        print(f"{len(pending)} chunks to transcribe in {len(batches)} batches (longest first).")

        track_by_index = {t.index: t for t in tracks}
        pcms = {
            t.index: open_pcm(pcm_paths[t.index])
            for t in tracks if any(c.key not in done for c in plans[t.index])
        }

        def load_batch(batch: list[Chunk]) -> list[np.ndarray]:
            return [
                read_chunk_audio(pcms[c.track_index], track_by_index[c.track_index], c)
                for c in batch
            ]

        try:
            failed = run_batches(backend, batches, load_batch, jsonl, segments)
        finally:
            for pcm in pcms.values():
                close_pcm(pcm)

        backend.unload()
        write_outputs(segments, args.outdir, info)

        kept = sum(1 for s in segments if not s.dropped)
        n_quiet = sum(1 for r in skipped.values() if r == "too-quiet")
        print(
            f"\nDone. {kept} segments kept, {len(segments) - kept} filtered; skipped "
            f"{n_quiet} chunks as too quiet (below {args.min_peak_db:g} dBFS) and "
            f"{len(skipped) - n_quiet} with too little speech."
        )
        if failed:
            print(f"  {failed} chunk(s) failed -- re-run the same command to retry them.")
        report = language_report(segments)
        if report:
            print("Script mix (by output, not by requested tag):")
            print(report)
        print(f"Wrote transcript.txt / .srt / .json to {args.outdir}/")

    finally:
        if args.keep_scratch:
            print(f"Scratch retained at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            if scratch.exists():
                # Windows refuses to unlink a file that is still mapped.
                print(
                    f"warning: could not fully remove scratch at {scratch} -- "
                    f"delete it manually to reclaim the disk space."
                )


if __name__ == "__main__":
    main()

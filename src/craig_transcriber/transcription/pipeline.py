"""
The transcription driver shared by the Craig and phone pipelines: planning
chunks, batching, running the backend with resume, and output-folder checks.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import (
    SpeechCheck, build_chunks, chunk_peak_db, close_pcm, decode_to_pcm, open_pcm,
    read_chunk_audio,
)
from .backends import Backend
from .craig import discover_tracks
from .filters import screen
from .language import label_script
from .models import Chunk, Segment, Track
from .settings import (
    MIN_GAP_S, MIN_PEAK_DBFS, MIN_SPEECH_RATIO, SR, TOKEN_CEIL, TOKEN_FLOOR,
    TOKENS_PER_S, VAD_MAX_CHUNK_S,
)
from .speakers import is_skipped, map_speaker


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
            if is_skipped(renamed):
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

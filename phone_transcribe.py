#!/usr/bin/env python3
"""
phone_transcribe.py -- Arabic transcription for ONE mixed recording of the
whole table (a phone on the table, a Zoom/Meet export, ...).

Unlike Craig, a phone gives one track with everyone on it, so this script
has to work out who spoke when (diarization) before transcribing:

  1. diarize   -- pyannote (best; needs a free Hugging Face token) or a
                  built-in fallback (silero VAD + voice embeddings + clustering)
  2. name      -- match each anonymous speaker against voice profiles built
                  from your Craig recordings, so SPEAKER_02 becomes "Arnok"
  3. transcribe -- the same QwenCleo batching, filters and resume as
                  craig_transcribe.py, one chunk per speaker turn

Usage:
    # once: learn everyone's voice from Craig recordings you already have
    python phone_transcribe.py --enroll recording.zip other.sesx.zip --speaker-map speaker_map.txt

    # every session
    python phone_transcribe.py session.m4a -o out-phone --num-speakers 5

    # rename speakers without re-transcribing
    python phone_transcribe.py --rewrite-outputs -o out-phone --speaker-map phone_names.txt

Resume is automatic, as in craig_transcribe.py. The diarization is cached in
the output folder, so a re-run goes straight to transcription.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

from craig_transcribe import (
    BACKENDS, MAX_CHUNK_S, MIN_GAP_S, MIN_SPEECH_S, PAD_S, SR,
    Chunk, Segment, Track,
    chunk_peak_db, close_pcm, decode_to_pcm, discover_tracks, build_chunks,
    extract_archive, language_report, load_done, load_speaker_map, make_batches,
    map_speaker, open_pcm, quietest_split, read_chunk_audio, read_recording_info,
    require_ffmpeg, run_batches, write_outputs, write_wav,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"  # not gated

# Length of the audio window each voice embedding is computed on. Profiles
# and speaker centroids use the longer window (more reliable); the fallback
# diarizer slides the shorter one so it can follow quick back-and-forth.
PROFILE_WIN_S = 3.0
DIARIZE_WIN_S = 1.5
DIARIZE_STEP_S = 0.75

# A diarized speaker is given a profile's name only if their voices are at
# least this similar (cosine, -1..1). Below it they stay "Speaker N".
# Measured on a simulated phone mix vs profiles from other sessions: true
# matches 0.52-0.85, the closest wrong pair 0.46.
MATCH_MIN_SIM = 0.45

# With voice profiles, the built-in diarizer splits into this many extra
# groups and lets the profile match merge them back. A DM doing NPC voices
# otherwise drags a quiet player's cluster toward him: 90% -> 96% of speech
# time correctly attributed in testing.
EXTRA_CLUSTERS = 2

# Phone audio has room noise, never Craig's digital silence, and the turns
# already come from a voice detector, so the level gate is much laxer here.
PHONE_MIN_PEAK_DBFS = -50.0

DEFAULT_VOICES = Path("voices.npz")
SKIP_NAMES = {"SKIP", "-", "IGNORE"}


@dataclass
class Turn:
    start_s: float
    end_s: float
    label: str  # SPEAKER_00, SPEAKER_01, ... ordered by talk time


# ---------------------------------------------------------------------------
# Voice embeddings
# ---------------------------------------------------------------------------

class Embedder:
    """WeSpeaker ResNet34 speaker embeddings, batched on the GPU."""

    def __init__(self) -> None:
        import torch
        from pyannote.audio import Model
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model.from_pretrained(EMBEDDING_MODEL).to(self.device).eval()

    def embed(self, clips: list[np.ndarray], batch: int = 64) -> np.ndarray:
        """L2-normalised embedding per clip. All clips must be the same length."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(clips), batch):
                x = self.torch.from_numpy(np.stack(clips[i:i + batch])).to(self.device)
                out.append(self.model(x[:, None, :]).float().cpu().numpy())
        if not out:
            return np.zeros((0, 256), dtype=np.float32)
        emb = np.concatenate(out)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    def unload(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Centre-crop to n samples, or tile a short clip up to n."""
    if len(audio) >= n:
        lo = (len(audio) - n) // 2
        return audio[lo:lo + n]
    return np.resize(audio, n) if len(audio) else np.zeros(n, dtype=np.float32)


def read_span(pcm: np.memmap, start_s: float, end_s: float) -> np.ndarray:
    lo = max(0, int(start_s * SR))
    hi = min(len(pcm), int(end_s * SR))
    return np.asarray(pcm[lo:hi], dtype=np.float32) / 32768.0


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


# ---------------------------------------------------------------------------
# Voice profiles (--enroll)
# ---------------------------------------------------------------------------

def load_voices(path: Path) -> dict[str, tuple[np.ndarray, int]]:
    """name -> (sum of embeddings, count). Sums let new sessions add evidence."""
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {
        str(name): (data["sums"][i], int(data["counts"][i]))
        for i, name in enumerate(data["names"])
    }


def save_voices(path: Path, voices: dict[str, tuple[np.ndarray, int]]) -> None:
    names = sorted(voices)
    np.savez(
        path,
        names=np.array(names),
        sums=np.stack([voices[n][0] for n in names]),
        counts=np.array([voices[n][1] for n in names]),
    )


def enroll(inputs: list[Path], voices_path: Path, speaker_map: dict[str, str] | None,
           minutes: float, scratch_root: Path | None) -> None:
    """
    Build a voice profile per speaker from Craig recordings, where every
    track is one known person. Clean, loud chunks are spread across the
    whole session so one noisy stretch doesn't dominate.
    """
    voices = load_voices(voices_path)
    if voices:
        print(f"Adding to {len(voices)} existing profile(s) in {voices_path}")
    embedder = Embedder()
    win = int(PROFILE_WIN_S * SR)
    per_track = max(1, int(minutes * 60 / PROFILE_WIN_S))

    for src in inputs:
        scratch = Path(tempfile.mkdtemp(prefix="enroll_", dir=scratch_root))
        try:
            root = extract_archive(src, scratch)
            rec = read_recording_info(root).get("Recording", src.name)
            print(f"\n{src.name} (recording {rec}):")
            for track in discover_tracks(root):
                name = track.speaker
                if speaker_map:
                    name = map_speaker(track.speaker, speaker_map) or track.speaker
                if name.upper() in SKIP_NAMES:
                    print(f"  skip  {track.speaker}")
                    continue
                pcm_path = decode_to_pcm(track.path, scratch / f"track{track.index}.wav")
                pcm = open_pcm(pcm_path)
                try:
                    # Short gap: we want tight islands of speech, not sentences.
                    chunks = [
                        c for c in build_chunks(pcm, track, min_gap_s=0.45)
                        if c.duration_s >= 1.5 and chunk_peak_db(pcm, track, c) >= -35.0
                    ]
                    windows = []
                    for c in chunks:
                        audio = read_chunk_audio(pcm, track, c)
                        if len(audio) < win:
                            windows.append(fit_length(audio, win))
                        else:
                            windows.extend(audio[i:i + win] for i in range(0, len(audio) - win + 1, win))
                    if len(windows) > per_track:
                        picks = np.linspace(0, len(windows) - 1, per_track).round().astype(int)
                        windows = [windows[i] for i in picks]
                    if len(windows) < 10:
                        print(f"  skip  {track.speaker:<24} only {len(windows)} usable window(s)")
                        continue
                    emb = embedder.embed(windows)
                finally:
                    close_pcm(pcm)
                total, count = voices.get(name, (np.zeros(emb.shape[1], dtype=np.float64), 0))
                voices[name] = (total + emb.sum(axis=0), count + len(emb))
                print(f"  {track.speaker:<24} -> {name:<18} {len(emb) * PROFILE_WIN_S / 60:5.1f} min of voice")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    embedder.unload()
    if not voices:
        sys.exit("error: no voices enrolled")
    save_voices(voices_path, voices)
    print(f"\nSaved {len(voices)} voice profile(s) to {voices_path}: {', '.join(sorted(voices))}")

    # Two profiles that look alike will get confused on the phone too.
    names = sorted(voices)
    profiles = np.stack([normalize(voices[n][0]) for n in names])
    sim = profiles @ profiles.T
    np.fill_diagonal(sim, -1)
    i, j = np.unravel_index(np.argmax(sim), sim.shape)
    print(f"Most similar pair: {names[i]} / {names[j]} ({sim[i, j]:.2f}; under ~0.5 is easy to tell apart)")


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def label_by_talk_time(raw: list[tuple[float, float, object]]) -> list[Turn]:
    """Rename arbitrary cluster ids to SPEAKER_00.. by descending talk time."""
    talk: Counter = Counter()
    for start, end, lab in raw:
        talk[lab] += end - start
    names = {lab: f"SPEAKER_{i:02d}" for i, (lab, _) in enumerate(talk.most_common())}
    return sorted((Turn(round(s, 3), round(e, 3), names[lab]) for s, e, lab in raw),
                  key=lambda t: t.start_s)


def diarize_pyannote(pcm: np.memmap, token: str | None, num_speakers: int | None,
                     min_speakers: int | None, max_speakers: int | None) -> list[Turn]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=token)
    if pipeline is None:
        raise RuntimeError("pipeline could not be loaded")
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # In memory: pyannote's own file loading needs FFmpeg DLLs on Windows.
    waveform = torch.from_numpy(np.asarray(pcm, dtype=np.float32) / 32768.0)[None]
    kwargs = dict(num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            output = pipeline({"waveform": waveform, "sample_rate": SR}, hook=hook, **kwargs)
    except ImportError:
        output = pipeline({"waveform": waveform, "sample_rate": SR}, **kwargs)
    # "exclusive" has no overlapping turns -- one voice per chunk is what ASR wants.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    raw = [(seg.start, seg.end, lab) for seg, _, lab in annotation.itertracks(yield_label=True)]
    del pipeline, waveform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return label_by_talk_time(raw)


def speech_regions(pcm: np.memmap, block_s: float = 1800.0) -> list[tuple[float, float]]:
    """Silero VAD over the recording, half an hour at a time to bound RAM."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    block = int(block_s * SR)
    regions: list[tuple[float, float]] = []
    for base in range(0, len(pcm), block):
        audio = torch.from_numpy(np.asarray(pcm[base:base + block], dtype=np.float32) / 32768.0)
        for ts in get_speech_timestamps(
            audio, model, sampling_rate=SR,
            min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=60,
        ):
            regions.append(((base + ts["start"]) / SR, (base + ts["end"]) / SR))
        print(f"  voice detection {min(100, 100 * (base + block) / len(pcm)):5.1f}%", end="\r", flush=True)
    print()
    return regions


def cluster(emb: np.ndarray, k: int, max_fit: int = 3000) -> np.ndarray:
    """
    Spectral clustering on cosine affinity, fitted on an even subsample (it is
    O(n^2)), then refined over all windows by nearest centroid.
    """
    from sklearn.cluster import SpectralClustering

    fit_idx = np.linspace(0, len(emb) - 1, min(len(emb), max_fit)).round().astype(int)
    affinity = np.clip(emb[fit_idx] @ emb[fit_idx].T, 0, None)
    labels_fit = SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=0, assign_labels="cluster_qr",
    ).fit_predict(affinity)
    centroids = np.stack([normalize(emb[fit_idx][labels_fit == c].mean(axis=0)) for c in range(k)])
    for _ in range(3):
        labels = np.argmax(emb @ centroids.T, axis=1)
        centroids = np.stack([
            normalize(emb[labels == c].mean(axis=0)) if np.any(labels == c) else centroids[c]
            for c in range(k)
        ])
    return np.argmax(emb @ centroids.T, axis=1)


def diarize_basic(pcm: np.memmap, num_speakers: int, embedder: Embedder) -> list[Turn]:
    """
    Token-free fallback: find speech, slide a short window over it, embed each
    window, cluster the windows into num_speakers voices, and cut a turn
    wherever the voice changes. Weaker than pyannote on overlapping speech.
    """
    regions = speech_regions(pcm)
    win, step = int(DIARIZE_WIN_S * SR), DIARIZE_STEP_S
    windows: list[tuple[int, float]] = []  # (region index, window centre in s)
    for r, (a, b) in enumerate(regions):
        if b - a <= DIARIZE_WIN_S:
            windows.append((r, (a + b) / 2))
            continue
        n = int((b - a - DIARIZE_WIN_S) // step) + 1
        centres = [a + DIARIZE_WIN_S / 2 + i * step for i in range(n)]
        if b - DIARIZE_WIN_S / 2 - centres[-1] > step / 2:
            centres.append(b - DIARIZE_WIN_S / 2)
        windows.extend((r, c) for c in centres)
    if not windows:
        return []

    print(f"  embedding {len(windows)} windows...")
    clips = []
    emb_parts = []
    for r, c in windows:
        clips.append(fit_length(read_span(pcm, c - DIARIZE_WIN_S / 2, c + DIARIZE_WIN_S / 2), win))
        if len(clips) == 2048:
            emb_parts.append(embedder.embed(clips))
            clips = []
    if clips:
        emb_parts.append(embedder.embed(clips))
    emb = np.concatenate(emb_parts)

    print(f"  clustering into {num_speakers} speakers...")
    labels = cluster(emb, num_speakers)

    raw: list[tuple[float, float, int]] = []
    i = 0
    while i < len(windows):
        r = windows[i][0]
        j = i
        while j < len(windows) and windows[j][0] == r:
            j += 1
        seq = labels[i:j].tolist()
        # Majority of each window and its neighbours: one odd window inside
        # someone's sentence is noise, not a speaker change.
        if len(seq) >= 3:
            seq = [Counter(seq[max(0, k - 1):k + 2]).most_common(1)[0][0] for k in range(len(seq))]
        a, b = regions[r]
        start = a
        for k in range(1, len(seq) + 1):
            if k == len(seq) or seq[k] != seq[k - 1]:
                end = b if k == len(seq) else (windows[i + k - 1][1] + windows[i + k][1]) / 2
                raw.append((start, end, seq[k - 1]))
                start = end
        i = j
    return label_by_talk_time(raw)


# ---------------------------------------------------------------------------
# Naming speakers from voice profiles
# ---------------------------------------------------------------------------

def speaker_centroids(pcm: np.memmap, turns: list[Turn], embedder: Embedder,
                      per_speaker: int = 40) -> dict[str, np.ndarray]:
    """One embedding per diarized speaker, from their longest turns."""
    win = int(PROFILE_WIN_S * SR)
    by_label: dict[str, list[Turn]] = {}
    for t in turns:
        by_label.setdefault(t.label, []).append(t)
    out = {}
    for label, ts in by_label.items():
        longest = sorted(ts, key=lambda t: t.end_s - t.start_s, reverse=True)[:per_speaker]
        clips = [fit_length(read_span(pcm, t.start_s, t.end_s), win)
                 for t in longest if t.end_s - t.start_s >= 1.0]
        if clips:
            out[label] = normalize(embedder.embed(clips).mean(axis=0))
    return out


def match_speakers(centroids: dict[str, np.ndarray],
                   voices: dict[str, tuple[np.ndarray, int]]) -> dict[str, dict]:
    """
    Give each diarized speaker the profile name it is closest to, if close
    enough. Several speakers may get the same name: diarization often splits
    one person (the DM's narration vs his NPC voices) into two groups, and
    this merges them back. Forcing one-to-one instead pinned leftover names
    on whatever group remained.
    """
    labels = sorted(centroids)
    names = sorted(voices)
    result = {lab: {"name": None, "similarity": None, "best": None} for lab in labels}
    if not labels or not names:
        return result
    profiles = np.stack([normalize(voices[n][0]) for n in names])
    sim = np.stack([centroids[lab] for lab in labels]) @ profiles.T
    for r, lab in enumerate(labels):
        best = int(np.argmax(sim[r]))
        result[lab] = {
            "name": names[best] if sim[r, best] >= MATCH_MIN_SIM else None,
            "similarity": round(float(sim[r, best]), 3),
            "best": names[best],
            "scores": {n: round(float(sim[r, c]), 3) for c, n in enumerate(names)},
        }
    return result


def merge_same_voice(turns: list[Turn], matched: dict[str, dict]) -> list[Turn]:
    """Give every group matched to the same profile one shared label."""
    by_name: dict[str, str] = {}
    canon: dict[str, str] = {}
    for label in sorted(matched):  # SPEAKER_00 talks most, so it becomes the canonical one
        name = matched[label]["name"]
        canon[label] = by_name.setdefault(name, label) if name else label
    raw = [(t.start_s, t.end_s, canon.get(t.label, t.label)) for t in turns]
    return label_by_talk_time(raw)


def display_names(speakers: dict[str, dict], user_map: dict[str, str] | None) -> dict[str, str]:
    """
    label -> name shown in the transcript. A --speaker-map entry wins (keyed
    by SPEAKER_00 or by the auto name), then the voice match, then "Speaker N".
    """
    names = {}
    for label, meta in speakers.items():
        auto = meta.get("name") or f"Speaker {int(label.split('_')[-1]) + 1}"
        chosen = auto
        if user_map:
            chosen = user_map.get(label.lower()) or user_map.get(auto.lower()) or auto
        names[label] = chosen
    return names


def write_samples(pcm: np.memmap, turns: list[Turn], outdir: Path, seconds: float = 12.0) -> None:
    """A short clip of each speaker, so you can check who's who by ear."""
    sample_dir = outdir / "speaker_samples"
    sample_dir.mkdir(exist_ok=True)
    by_label: dict[str, list[Turn]] = {}
    for t in turns:
        by_label.setdefault(t.label, []).append(t)
    for label, ts in by_label.items():
        pieces, total = [], 0.0
        for t in sorted(ts, key=lambda t: t.end_s - t.start_s, reverse=True)[:6]:
            piece = read_span(pcm, t.start_s, min(t.end_s, t.start_s + 6.0))
            pieces += [piece, np.zeros(int(0.4 * SR), dtype=np.float32)]
            total += len(piece) / SR
            if total >= seconds:
                break
        if pieces:
            write_wav(np.concatenate(pieces), sample_dir / f"{label}.wav")


# ---------------------------------------------------------------------------
# Turns -> ASR chunks
# ---------------------------------------------------------------------------

def turns_to_chunks(pcm: np.memmap, turns: list[Turn], min_gap_s: float) -> list[Chunk]:
    """
    Join a speaker's consecutive turns across short pauses (nobody else
    spoke in between) into sentence-sized chunks, split anything over
    MAX_CHUNK_S at its quietest point, and pad each edge without reaching
    into the neighbouring chunk.
    """
    groups: list[list[float | str]] = []
    for t in turns:
        g = groups[-1] if groups else None
        if (g and g[2] == t.label and t.start_s - g[1] <= min_gap_s
                and t.end_s - g[0] <= MAX_CHUNK_S):
            g[1] = t.end_s
        else:
            groups.append([t.start_s, t.end_s, t.label])

    max_len = int(MAX_CHUNK_S * SR)
    spans: list[tuple[float, float, str]] = []
    for start, end, label in groups:
        lo, hi = int(start * SR), int(end * SR)
        while hi - lo > max_len:
            cut = quietest_split(pcm, lo + int(max_len * 0.75), lo + max_len)
            spans.append((lo / SR, cut / SR, label))
            lo = cut
        spans.append((lo / SR, hi / SR, label))
    spans = [s for s in spans if s[1] - s[0] >= MIN_SPEECH_S]

    chunks = []
    total = len(pcm) / SR
    for i, (start, end, label) in enumerate(spans):
        prev_end = spans[i - 1][1] if i else 0.0
        next_start = spans[i + 1][0] if i + 1 < len(spans) else total
        chunks.append(Chunk(
            track_index=0, speaker=label, chunk_id=i,
            start_s=max(start - PAD_S, (prev_end + start) / 2, 0.0),
            end_s=min(end + PAD_S, (end + next_start) / 2, total),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Output folder bookkeeping
# ---------------------------------------------------------------------------

def check_source(outdir: Path, info: dict[str, str]) -> None:
    """Refuse to mix two recordings, or two chunkings, in one output folder."""
    marker = outdir / "recording.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        for key in ("Source", "Size bytes", "Min gap s"):
            if key in saved and saved[key] != info[key]:
                sys.exit(
                    f"error: {outdir} was made with {key} = {saved[key]}, but this run has "
                    f"{info[key]}. Use a different -o directory."
                )
        info.update({k: v for k, v in saved.items() if k not in info})
    elif (outdir / "segments.jsonl").exists():
        sys.exit(f"error: {outdir} has a segments.jsonl from an unknown recording. Use a different -o.")
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def write_named_outputs(segments: list[Segment], outdir: Path, info: dict[str, str],
                        speakers: dict[str, dict], user_map: dict[str, str] | None) -> None:
    names = display_names(speakers, user_map)
    shown = []
    for seg in segments:
        name = names.get(seg.speaker, seg.speaker)
        if name.upper() in SKIP_NAMES:
            continue
        shown.append(replace(seg, speaker=name))
    write_outputs(shown, outdir, info)


def print_speakers(speakers: dict[str, dict], turns: list[Turn], names: dict[str, str]) -> None:
    talk: Counter = Counter()
    for t in turns:
        talk[t.label] += t.end_s - t.start_s
    print("\nSpeakers:")
    for label in sorted(speakers):
        meta = speakers[label]
        line = f"  {label}  {talk[label] / 60:6.1f} min  -> {names[label]}"
        if meta.get("best"):
            line += f"   (closest voice: {meta['best']} {meta['similarity']:.2f})"
        print(line)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one mixed recording of several speakers (Arabic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("inputs", type=Path, nargs="*",
                        help="the recording (any format ffmpeg reads); with --enroll, Craig zips")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("transcripts-phone"))
    parser.add_argument("--enroll", action="store_true",
                        help="build voice profiles from Craig recordings instead of transcribing")
    parser.add_argument("--enroll-minutes", type=float, default=5.0, metavar="MIN",
                        help="voice to sample per speaker per Craig recording (default 5)")
    parser.add_argument("--voices", type=Path, default=DEFAULT_VOICES, metavar="FILE",
                        help=f"voice profile file (default {DEFAULT_VOICES})")
    parser.add_argument("--speaker-map", type=Path, default=None, metavar="FILE",
                        help=("with --enroll: Discord username = name, as in craig_transcribe.py. "
                              "Otherwise: SPEAKER_00 = name, to fix or set names by hand "
                              "(SKIP hides a speaker)."))
    parser.add_argument("--num-speakers", type=int, default=None, metavar="N",
                        help="how many people talk in the recording -- improves diarization a lot")
    parser.add_argument("--min-speakers", type=int, default=None, metavar="N")
    parser.add_argument("--max-speakers", type=int, default=None, metavar="N")
    parser.add_argument("--diarizer", default="auto", choices=["auto", "pyannote", "basic"],
                        help=("pyannote: best, needs a Hugging Face token. basic: built-in, no "
                              "token, needs --num-speakers or voice profiles. auto (default): "
                              "pyannote if it loads, else basic."))
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token (default: $HF_TOKEN, or a saved `hf auth login`)")
    parser.add_argument("--backend", default="qwencleo", choices=sorted(BACKENDS))
    parser.add_argument("--language", default="auto", choices=["ar", "en", "auto"])
    parser.add_argument("--min-gap", type=float, default=MIN_GAP_S, metavar="SECONDS",
                        help=f"join one speaker's turns across pauses up to this long (default {MIN_GAP_S:g})")
    parser.add_argument("--min-peak-db", type=float, default=PHONE_MIN_PEAK_DBFS, metavar="DBFS",
                        help=f"skip chunks whose loudest sample is below this (default {PHONE_MIN_PEAK_DBFS:g})")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-seconds", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--limit", type=float, default=None, metavar="SECONDS",
                        help="only process the first N seconds (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="diarize and name speakers, but don't transcribe")
    parser.add_argument("--scratch-dir", type=Path, default=None, metavar="DIR")
    parser.add_argument("--rewrite-outputs", action="store_true",
                        help="re-apply names (--speaker-map) and filters to an existing -o folder; no GPU")
    args = parser.parse_args()

    user_map = None
    if args.speaker_map and not args.enroll:
        user_map = load_speaker_map(args.speaker_map)

    if args.enroll:
        if not args.inputs:
            parser.error("--enroll needs one or more Craig recordings")
        require_ffmpeg()
        speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else None
        enroll(args.inputs, args.voices, speaker_map, args.enroll_minutes, args.scratch_dir)
        return

    if args.rewrite_outputs:
        jsonl = args.outdir / "segments.jsonl"
        if not jsonl.exists():
            sys.exit(f"error: {jsonl} not found; nothing to rewrite")
        speakers = json.loads((args.outdir / "speakers.json").read_text(encoding="utf-8"))
        info = json.loads((args.outdir / "recording.json").read_text(encoding="utf-8"))
        write_named_outputs(list(load_done(jsonl).values()), args.outdir, info, speakers, user_map)
        names = display_names(speakers, user_map)
        print("\n".join(f"  {lab} -> {names[lab]}" for lab in sorted(names)))
        print(f"Wrote transcript.txt / .srt / .json under {args.outdir}")
        return

    if len(args.inputs) != 1:
        parser.error("give exactly one recording to transcribe")
    src = args.inputs[0]
    if not src.exists():
        sys.exit(f"error: {src} not found")
    require_ffmpeg()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="phone_", dir=args.scratch_dir))
    pcm = None

    try:
        info = {
            "Source": src.name,
            "Size bytes": str(src.stat().st_size),
            "Min gap s": f"{args.min_gap:g}",
        }
        if args.limit is not None:
            info["Limit s"] = f"{args.limit:g}"
        check_source(args.outdir, info)

        print(f"Decoding {src.name}...")
        pcm_path = decode_to_pcm(src, scratch / "mix.wav")
        if args.limit is not None:
            trimmed = open_pcm(pcm_path)
            write_wav(np.asarray(trimmed[:int(args.limit * SR)], dtype=np.float32) / 32768.0,
                      scratch / "mix_limit.wav")
            close_pcm(trimmed)
            pcm_path = scratch / "mix_limit.wav"
        pcm = open_pcm(pcm_path)
        print(f"  {len(pcm) / SR / 60:.1f} min of audio")

        voices = load_voices(args.voices)
        if voices:
            print(f"Voice profiles: {', '.join(sorted(voices))} (from {args.voices})")
        else:
            print(f"No voice profiles at {args.voices}: speakers will be named Speaker 1, 2, ... "
                  f"(run --enroll first to name them automatically)")

        # --- 1. diarization (cached) ---------------------------------------
        diar_path = args.outdir / "diarization.json"
        embedder = None
        if diar_path.exists():
            saved = json.loads(diar_path.read_text(encoding="utf-8"))
            turns = [Turn(**t) for t in saved["turns"]]
            info["Diarizer"] = saved["diarizer"]
            print(f"\nUsing saved diarization ({saved['diarizer']}, {len(turns)} turns) "
                  f"from {diar_path}; delete it and segments.jsonl to redo.")
        else:
            print("\nDiarizing (who spoke when)...")
            t0 = time.time()
            turns, used = None, args.diarizer
            if args.diarizer in ("auto", "pyannote"):
                token = args.hf_token or os.environ.get("HF_TOKEN")
                try:
                    turns = diarize_pyannote(pcm, token, args.num_speakers,
                                             args.min_speakers, args.max_speakers)
                    used = "pyannote"
                except Exception as exc:  # noqa: BLE001 -- gated model, no token, ...
                    msg = (
                        f"  pyannote unavailable ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})\n"
                        f"  To enable it: create a free token at https://hf.co/settings/tokens, accept\n"
                        f"  the terms at https://hf.co/{PYANNOTE_MODEL}, then pass --hf-token or set HF_TOKEN."
                    )
                    if args.diarizer == "pyannote":
                        sys.exit(msg)
                    print(msg + "\n  Falling back to the built-in diarizer.")
            if turns is None:
                used = "basic"
                k = args.num_speakers or len(voices)
                if not k:
                    sys.exit("error: the built-in diarizer needs --num-speakers (or voice profiles)")
                if voices:
                    k += EXTRA_CLUSTERS  # merged back by the profile match below
                embedder = Embedder()
                turns = diarize_basic(pcm, k, embedder)
            info["Diarizer"] = used
            n_labels = len({t.label for t in turns})
            print(f"  {len(turns)} turns, {n_labels} speakers ({used}, {time.time() - t0:.0f}s)")
            if voices:
                # Groups that match the same voice are one person: merge them
                # now so their turns join into whole sentences. Saved with the
                # diarization, so chunk ids stay fixed for resume even if the
                # profiles change later.
                embedder = embedder or Embedder()
                matched = match_speakers(speaker_centroids(pcm, turns, embedder), voices)
                turns = merge_same_voice(turns, matched)
                if len({t.label for t in turns}) < n_labels:
                    print(f"  merged into {len({t.label for t in turns})} speakers by voice match")
            diar_path.write_text(json.dumps(
                {"diarizer": used, "turns": [asdict(t) for t in turns]}, indent=1), encoding="utf-8")

        # --- 2. naming -----------------------------------------------------
        speakers_path = args.outdir / "speakers.json"
        labels = sorted({t.label for t in turns})
        if voices:
            embedder = embedder or Embedder()
            speakers = match_speakers(speaker_centroids(pcm, turns, embedder), voices)
        elif speakers_path.exists():
            speakers = json.loads(speakers_path.read_text(encoding="utf-8"))
        else:
            speakers = {}
        for lab in labels:
            speakers.setdefault(lab, {"name": None, "similarity": None, "best": None})
        if embedder:
            embedder.unload()
        speakers_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
        names = display_names(speakers, user_map)
        print_speakers(speakers, turns, names)
        write_samples(pcm, turns, args.outdir)
        print(f"  listen to {args.outdir / 'speaker_samples'} to check; fix names with --speaker-map")

        # --- 3. transcription ----------------------------------------------
        mix = Track(index=0, speaker="mix", path=pcm_path)
        chunks = turns_to_chunks(pcm, turns, args.min_gap)
        loud = [c for c in chunks if chunk_peak_db(pcm, mix, c) >= args.min_peak_db]
        quiet = len(chunks) - len(loud)
        print(f"\n{len(loud)} chunks / {sum(c.duration_s for c in loud) / 60:.1f} min of speech"
              + (f" ({quiet} too quiet, skipped)" if quiet else ""))
        if args.dry_run:
            print("Dry run -- not transcribing.")
            return

        jsonl = args.outdir / "segments.jsonl"
        done = load_done(jsonl)
        if done:
            print(f"Resuming: {len(done)} chunks already transcribed.")
        segments = list(done.values())
        pending = [c for c in loud if c.key not in done]
        failed = 0
        if pending:
            backend = BACKENDS[args.backend](language=args.language)
            batches = make_batches(pending, args.batch_size or backend.default_batch,
                                   args.batch_seconds or backend.default_batch_seconds)
            print(f"Loading backend '{backend.name}'; {len(pending)} chunks in {len(batches)} batches.")
            backend.load()
            failed = run_batches(backend, batches,
                                 lambda batch: [read_chunk_audio(pcm, mix, c) for c in batch],
                                 jsonl, segments)
            backend.unload()

        write_named_outputs(segments, args.outdir, info, speakers, user_map)
        kept = sum(1 for s in segments if not s.dropped)
        print(f"\nDone. {kept} segments kept, {len(segments) - kept} filtered.")
        if failed:
            print(f"  {failed} chunk(s) failed -- re-run the same command to retry them.")
        report = language_report(segments)
        if report:
            print(report)
        print(f"Wrote transcript.txt / .srt / .json to {args.outdir}/")

    finally:
        close_pcm(pcm)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()

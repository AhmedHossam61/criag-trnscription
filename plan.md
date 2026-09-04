# Craig Transcriber — Arabic transcription for Discord recordings

Handoff notes for working on this locally. Read this before changing anything
in `craig_transcribe.py`.

---

## Goal

Take a Craig Bot recording of a Discord voice call (Egyptian Arabic, with
English mixed in) and produce a speaker-labelled transcript. Recordings are
typically **8 hours** long. Everything runs locally.

## Environment

| | |
|---|---|
| GPU | RTX 3050, **8 GB VRAM** (Ampere, sm_86) |
| OS / shell | Windows, **cmd.exe** — not bash, not PowerShell |
| Python | 3.12 preferred (3.13 has thin ML wheel coverage) |
| External | `ffmpeg` on PATH — not a pip package |

The 8 GB VRAM ceiling is the binding constraint on model choice. Only one
model is ever resident.

---

## The two facts the whole design rests on

Both verified against the 20-second sample.

**1. Craig writes one FLAC per speaker, aligned to a shared t=0.**
Confirmed in the `.sesx` session file: every `audioClip` had `startPoint="0"`.
So diarization is free — no pyannote, no clustering, no VRAM spent on it, and
overlapping speech is handled correctly because each speaker is on their own
track. The sample contains a real interruption at ~6.0s that a single mixdown
would have garbled.

**2. Gaps between utterances are exact digital zeros.**
Discord sends no packets when a user isn't talking, so Craig writes literal
`0.0`. Measured on the sample: 71.5% and 50.9% of each track. That gives a
free, exact VAD — `sample != 0` rather than an energy heuristic.

The payoff is large. Synthetic test, 8 speakers × 30 min:

```
32 track-hours of audio  ->  62 minutes of actual speech
```

Only that 62 minutes reaches the GPU. Segmenting took 5.2s at 107 MB peak RSS
(tracks are memory-mapped, never loaded whole).

---

## Pipeline

```
Craig .zip
  ├─ info.txt          guild / channel / start time  (contains Arabic)
  ├─ *.sesx            per-clip start offsets
  └─ <id>_data/
       ├─ 1-<user>.flac    48 kHz stereo, zero-padded
       └─ 2-<user>.flac

  │
  │  ffmpeg -> 16 kHz mono PCM WAV in scratch (~950 MB per speaker per 8h)
  ▼
  np.memmap                     never hold a full track in RAM
  │
  ▼
  find_voiced_runs()            zero-run VAD, block-wise scan
  merge_runs(MIN_GAP_S)         bridge jitter gaps inside a sentence
  build_chunks(MAX_CHUNK_S)     split long runs at the quietest frame
  │
  ▼
  Backend.transcribe_batch()    -> [(text, language_tag), ...]
  │
  ▼
  screen()                      hallucination filter
  │
  ▼
  segments.jsonl (append, flushed per batch)   <- resume point
  │
  ▼
  transcript.txt / .srt / .json  + dropped.txt
```

### Tunables (top of the file)

| Constant | Value | Why |
|---|---|---|
| `MIN_GAP_S` | 0.45 | Shorter zero-runs are packet jitter mid-sentence. Splitting there cuts words in half. |
| `MAX_CHUNK_S` | 28.0 | Whisper's encoder is hard-capped at 30s. Qwen/Cohere take more but degrade. |
| `MIN_SPEECH_S` | 0.35 | Below this it's a cough or mic bump, and models hallucinate sentences out of it. |
| `PAD_S` | 0.20 | Keeps plosives from being clipped at boundaries. |

---

## Model choice

**Recommended: `mohammedaly22/QwenCleo-ASR`** — Qwen3-ASR-1.7B fine-tuned on
Egyptian speech *and* Arabic↔English code-switching. Reports 19.85% WER on the
author's Egyptian + code-switching set vs ~40% for the Qwen3-ASR base. It keeps
embedded English in Latin script (`React`, `download`) instead of
transliterating into Arabic letters — which matters because this is a call
among developers.

Caveats worth keeping in mind: the benchmark is self-reported on the author's
own test set, and the repo has near-zero community usage.

| Backend | Params | VRAM (fp16) | Use when |
|---|---|---|---|
| `qwencleo` | 1.7B | ~4.5 GB | Default. Egyptian + code-switching. |
| `qwen3asr` | 2B | ~4.5 GB | QwenCleo's base. Native language ID; only backend with real hotword support. |
| `cohere` | 2B | ~5 GB | Fastest, best MSA/dialect WER (25.87 on the Open Universal Arabic ASR Leaderboard). **See warning below.** |
| `faster-whisper` | 1.5B | ~3 GB | Baseline only. Installs without torch. |

**Cohere warning.** Its model card states it has no automatic language
detection and shows *inconsistent performance on code-switched audio* — which
contradicts the launch blog. Under `--language auto` this backend runs each
chunk under both `ar` and `en` tags and keeps whichever scored higher on mean
token log-prob. That doubles its cost and is a workaround for a stated
limitation, not a feature.

**Rejected:** Whisper turbo (worse on Arabic than large-v3), MMS, SeamlessM4T,
OmniASR-7B (won't fit 8 GB), Egyptian-only Whisper fine-tunes (break on
English code-switching).

---

## Code-switching

Two distinct problems, often conflated:

- **English words inside an Arabic sentence** — `انا عملت ال deployment على ال cluster`.
  Language tagging cannot help. The model either handles it or transliterates.
  This is what QwenCleo was fine-tuned for.
- **Whole English sentences between Arabic ones** — this *is* a tagging
  problem, and `--language auto` addresses it. Chunks are cut at silence
  boundaries within one speaker's track, so a chunk is usually one utterance,
  which lines up well with sentence-level switching.

`--language ar` forces an Arabic matrix tag; more stable if English turns out
to be only scattered words. `label_script()` classifies output by actual
characters produced, independent of the requested tag — so if `auto` misfires
and starts transliterating, the `en`/`mixed` share in the final report
collapses toward zero. That's the signal to switch to `ar`.

---

## Status

### Verified locally

- Craig zip parsing, `.sesx` offsets, speaker-name recovery
- Zero-run VAD and chunk boundaries against the real 20s sample
- Scale + memory behaviour (8 × 30 min synthetic)
- Hallucination filter (boilerplate, token loops, over-long output)
- Output writers, resume roundtrip, truncated-JSONL tolerance
- `--dry-run`, `--limit`, `--check-env`, error paths

### NOT verified — no GPU or model weights available when this was written

**Every one of the four backend classes is written against documentation and
has never been executed.** Treat `Backend.transcribe_batch` implementations as
first drafts. Specifically unproven:

- `QwenCleoASR.transcribe()` return shape — code reads `.text` and probes
  `.language` via `getattr`, both guessed from the README
- `processor.decode(..., return_format="parsed")` for Qwen3-ASR
- Cohere's dual-tag log-prob scoring (`compute_transition_scores` path)
- Whether `--hotwords` measurably helps
- Real throughput on a 3050 — the estimates below are extrapolation

**Rough throughput guesses** for ~8h of speech: Cohere 20–40 min, QwenCleo
several hours (autoregressive), faster-whisper in between. Benchmark with
`--limit 300` before committing to a full run.

### Bugs already found and fixed — don't reintroduce

- **WAV header offset.** ffmpeg writes a `LIST/INFO` chunk, so audio starts at
  byte **78**, not 44. Hardcoding 44 read ASCII metadata as samples, which the
  zero-run VAD reported as a phantom speech burst at t=0. `wav_data_span()`
  now walks the RIFF chunks. Header size varies between ffmpeg builds — never
  assume it.
- **UTF-8 console.** The recording metadata is Arabic (`Channel: المطبخ`).
  CMD defaults to cp1252/cp437 where printing that raises `UnicodeEncodeError`
  and kills the run. `_force_utf8_console()` runs at import.
- **Windows file locking.** `del` on an `np.memmap` doesn't release the
  mapping, so scratch cleanup silently failed and stranded gigabytes in
  `%TEMP%`. Use `close_pcm()` in a `finally`.
- **cu121 is retired.** Current PyTorch CUDA indexes are cu126, cu128, cu129,
  cu130. torch is deliberately unpinned — `--no-deps` means QwenCleo's
  suggested 2.5.1 was never enforced anyway, and it has no Python 3.13 wheels.

---

## Open questions

1. **Open-mic speakers break the zero-run VAD.** If someone uses
   always-transmit instead of voice activation, Craig records their track
   continuously and it collapses to one giant chunk. `--dry-run` shows this
   immediately as ~100% speech for that track. Fallback would be Silero VAD as
   a second stage on affected tracks only — not yet implemented.
2. **Which backend actually wins** on this audio. Unresolved until benchmarked.
3. **Overlap handling in output.** Segments are currently sorted by start time,
   so an interruption appears as two adjacent turns. Fine for reading, possibly
   wrong for downstream summarisation.
4. **Post-processing.** An LLM pass to clean up the merged transcript
   (punctuation, speaker attribution near overlaps) is unexplored.

---

## Commands

```cmd
REM one venv per backend; deactivate first if one is active
setup.cmd qwencleo
set PYTAG=-3.12 && setup.cmd qwencleo
set CUDA=cu128 && setup.cmd qwencleo

REM verify environment without running anything
python craig_transcribe.py --check-env --backend qwencleo

REM segment only, no model, no GPU — run this on the real 8h file first
python craig_transcribe.py recording.zip -o out --dry-run

REM benchmark a backend on 5 minutes
python craig_transcribe.py recording.zip -o bench --backend qwencleo --limit 300

REM full run; scratch on a roomy drive
python craig_transcribe.py recording.zip -o out --backend qwencleo ^
    --language auto --scratch-dir D:\craig-scratch --hotwords hotwords.txt
```

Resume is automatic — re-running skips anything already in `out/segments.jsonl`.

CMD does not expand wildcards, so type the full zip filename.

---

## Files

```
craig_transcribe.py           the pipeline (single file, no package)
setup.cmd                     per-backend venv creation
requirements.txt              core only — numpy. --dry-run works with just this
requirements-qwencleo.txt     read the header; needs --no-deps ordering
requirements-qwen3asr.txt
requirements-cohere.txt       gated weights, needs `hf auth login`
requirements-whisper.txt      no torch at all (CTranslate2)
plan.md                       this file
```

Everything heavy is imported lazily inside the backend classes, so the core
pipeline depends on numpy alone.

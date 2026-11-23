# Craig Transcriber — session notes (D&D use case)

Living log of decisions, current state, and future enhancements. Add to it
after each session; keep `plan.md` for the "why" and this file for the "what
we tried and what to try next".

---

## Current state

### Hardware / environment
- RTX 3050, 8 GB VRAM.
- Windows, cmd.exe (some commands run in PowerShell — prefix `.\`).
- Python 3.12 via uv-managed venv `.venv-qwencleo`.
- Setup: `setup.cmd qwencleo` (uv-based, verifies `torch.cuda.is_available()`).

### Backend in use
- `qwencleo` (mohammedaly22/QwenCleo-ASR, 1.7B, bfloat16 on load).
- Wrapper loads in bf16 on its own — no manual dtype coercion needed.
- **Does not accept hotwords.** The `--hotwords` flag is silently ignored for
  this backend. Character/NPC names will be inconsistently spelled.

### Best-known run parameters
- `--batch-size 8` — sweet spot. Batch 4 is safer, 12 is slower (past the
  plateau; QwenCleo's wrapper likely does not truly batch internally).
- `--language ar` — Arabic matrix, some English mixed in.
- `--hotwords hotwords.txt` (kept for future backends).
- `--speaker-map speaker_map.txt`.

### Throughput observed
| Config | ×realtime | ETA for 8h session |
|---|---|---|
| batch 2, no bot filter | 0.3× | ~33h |
| batch 8, no bot filter | 0.9× | ~11h |
| batch 12, bot filter | 0.5× (regression) | ~16h |
| batch 8, bot filter | ~0.9× target | ~8.5h |

Throughput is content-dependent — long chunks (~28s) take proportionally
longer than short ones. Early ×realtime is pessimistic because it includes
model-load time.

---

## What we added this session

### 1. `--speaker-map FILE` (with glob support)
- Format: `discord_username = display name`, one per line.
- Case-insensitive on the left; supports `fnmatch` globs (`jockie*`, `megadon*`).
- Set the right side to `SKIP` / `-` / `IGNORE` to drop a track entirely.
- Craig appends numeric discriminators (`megadoon` → `megadon7338`), so
  patterns MUST use `*` for any bot name.

Current `speaker_map.txt`:
```
ahmed hossam  = Kael
ahmedhoss61   = Kael
seifeldien    = Vaylor
geniusgamerm  = Dungeon Master
noob          = Arnok
arnok         = Arnok
abdoo_        = Arnok
hazemonsif    = Thalion
megadon*      = SKIP
megadoon*     = SKIP
jockie*       = SKIP
jockie_music* = SKIP
```

Bots skipped this session: `Jockie_Music_8158` (108 min of "music-as-speech"
was being sent to the GPU), `megadon7338`. Removing them saved ~127 min of
GPU work per 8h session — ~22% of total.

### 2. `hotwords.txt`
D&D 5.5e vocabulary: party names, rules terms (initiative, DC, AC, HP, saving
throw, advantage, etc.), classes, species, common spells, common creatures,
common gear. NPC section is empty — add each NPC as you meet them in play.

Currently a no-op for QwenCleo. Will bias decoding under `qwen3asr` or
`faster-whisper` backends.

### 3. Hallucination filter (`screen()`) improvements
Added:
- **Barney lyric pattern**: `بحبك ... بحبني` (both orderings, within 30 chars).
  Model emits this Arabic dub of "I love you, you love me" on near-silence.
- **n-gram loop detector** (`has_ngram_loop`): flags a chunk if any bigram or
  trigram appears ≥3 times. Catches the `X Y X Y X Y` pattern that the
  existing unigram `repetition_ratio` misses.
- **Re-screen at write time**: `write_outputs()` re-applies `screen()` to
  previously-kept segments. Adding a new hallucination pattern later moves
  those old segments into `dropped.txt` without re-transcribing.

### 4. `--rewrite-outputs` flag
Re-run the writer on an existing `out/segments.jsonl` without touching the
GPU or the model. Use after adding filter patterns:
```cmd
python craig_transcribe.py --rewrite-outputs -o out
```

### 5. `setup.cmd` — rewritten to use uv
- `uv venv` auto-fetches Python 3.12 if not installed.
- `uv pip install --python <venv>` (5–10× faster than pip).
- Hard fail on CPU-only torch: after install, runs
  `python -c "assert torch.cuda.is_available()"` and jumps to `:failed_cuda`
  with fix-it steps if False. Prevents silent CPU fallback.

---

## Known issues / open items

### Speed
- **QwenCleo does not truly batch.** Sublinear scaling from batch 2 → 8
  suggests the wrapper iterates internally. Batch 8 is the ceiling — no more
  gains without switching backends.
- **Content-dependent throughput.** Long chunks slow throughput; expect the
  ×realtime to fluctuate through the run, not stay flat.
- **VRAM ceiling.** ~4.5 GB model + batch 8 ≈ 6.5–7.0 GB used on 8 GB card.
  No room to go bigger even if the wrapper batched properly.

### Accuracy
- **No hotword biasing under QwenCleo.** Character/NPC names get mangled and
  inconsistent across the transcript. Fixing this = switching backends or
  post-processing (see below).
- **DM voices multiple NPCs on one track.** Diarization is correct (one
  track = one person, not one character). Transcript cannot distinguish DM
  narration from NPC dialogue at the ASR layer.
- **In-character vs out-of-character.** No acoustic marker — needs an LLM
  pass to split.

### Bots / speakers
- `bodaa` in the sample zip had 0 min of speech — probably a player who
  didn't talk. Currently unmapped; costs zero GPU time. Add a real mapping if
  they turn out to be someone real.

---

## Future enhancements (ranked)

### High-value, medium-effort

1. **Try `qwen3asr` backend for hotword-biased runs.**
   - Same 1.7B base as QwenCleo, minus the Egyptian fine-tune.
   - Loses some dialect WER, gains proper decoder biasing on character/NPC
     names — likely a net win for D&D given how much proper-noun jargon we
     have.
   - Requires `setup.cmd qwen3asr` in a separate venv.
   - Benchmark side-by-side on the same 5-minute slice (`--limit 300`)
     before committing.

2. **LLM post-processing pass** on the merged transcript:
   - Normalize character/NPC name spellings against `hotwords.txt`.
   - Split IC vs OOC lines (heuristics: rules jargon and meta-talk = OOC;
     NPC dialogue and description = IC when from DM track).
   - Attribute NPC lines on the DM track to specific NPCs from surrounding
     context.
   - Per-scene / per-session recap summarisation. Highest-leverage
     downstream step.

3. **Overlap handling.** Currently segments are sorted by start time, so
   interruptions read as adjacent turns. Fine for reading but wrong for
   downstream summarisation. Revisit if the merged transcript reads poorly.

### Speed-focused (accuracy trade-off)

4. **`faster-whisper large-v3`** as a "rough transcript in 1 hour" fallback.
   - RTX 3050 with `int8_float16` hits 8–15× realtime.
   - Egyptian dialect WER is worse than QwenCleo — real quality drop.
   - Good for "did anything interesting happen this session" skims.

5. **Cohere backend** as the fastest option (~20–40 min for 8h).
   - FastConformer encoder, not autoregressive.
   - No language ID → runs each chunk under both `ar` and `en` tags, keeps
     the higher-scoring one (doubles cost, still fastest overall).
   - Model card warns of inconsistent behavior on code-switched audio —
     English jargon like "Fireball" may get transliterated.
   - Gated weights; needs `hf auth login`.

### Robustness

6. **Silero-VAD fallback for open-mic speakers.** If a track's speech ratio
   is >90%, the zero-run VAD is being bypassed and every chunk hits
   `MAX_CHUNK_S`. DMs are the most likely offenders. Not implemented; not
   urgent unless a speaker actually shows this pattern.

7. **Extend `HALLUCINATION_PATTERNS`.** Watch `dropped.txt` and
   `transcript.txt` for new stock phrases and add them. Common families
   besides "بحبك/بحبني":
   - YouTube captions boilerplate (already covered).
   - Repeated religious phrases fired on silence ("الحمد لله" ×N).
   - Repeated short English ("okay okay okay", "yeah yeah yeah") — usually
     the n-gram detector catches these.

---

## Command reference (current)

Setup (once per backend):
```cmd
setup.cmd qwencleo
```

Environment check:
```cmd
python craig_transcribe.py --check-env --backend qwencleo
```

Dry-run (segmentation only, no GPU):
```cmd
python craig_transcribe.py recording.zip -o out --dry-run --speaker-map speaker_map.txt
```

Benchmark (5 min):
```cmd
python craig_transcribe.py recording.zip -o bench --backend qwencleo --language ar ^
    --hotwords hotwords.txt --speaker-map speaker_map.txt --batch-size 8 --limit 300
```

Full session:
```cmd
python craig_transcribe.py recording.zip -o out --backend qwencleo --language ar ^
    --hotwords hotwords.txt --speaker-map speaker_map.txt --batch-size 8 ^
    --scratch-dir D:\craig-scratch
```

Rewrite outputs after adding filter patterns (no GPU):
```cmd
python craig_transcribe.py --rewrite-outputs -o out
```

Watch GPU during a run (second cmd window):
```cmd
nvidia-smi -l 2
```
Expected: GPU-Util 95–100%, memory ~6.5–7.0 GB, temp under 80°C.

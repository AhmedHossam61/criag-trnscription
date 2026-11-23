# How to run the transcriber

Two scripts, one for each kind of recording:

| You have | Use | Speakers come from |
|---|---|---|
| A Craig zip from Discord (one track per player) | `craig_transcribe.py` | Discord usernames, via `speaker_map.txt` |
| One mixed recording (a phone on the table, a Zoom export, ...) | `phone_transcribe.py` | Voice matching against profiles built from Craig recordings |

All commands below are for **PowerShell**, run from the project folder
`G:\AI_ML\Projects\Criag_transcriber`.

---

## 0. One-time setup

### 0.1 Requirements
- An NVIDIA GPU with CUDA (tested on an RTX 3050 8 GB).
- `ffmpeg` on the PATH. Check with `ffmpeg -version`. If it's missing: `winget install Gyan.FFmpeg`.
- `uv`. Check with `uv --version`. If it's missing: `winget install --id=astral-sh.uv -e`.

### 0.2 Create the environment
The `.venv-qwencleo` environment already exists on this machine. To rebuild it from scratch (from **cmd**, not PowerShell):

```cmd
setup.cmd qwencleo
```

Then add the packages the phone script needs. They don't change torch:

```powershell
uv pip install --python .\.venv-qwencleo\Scripts\python.exe pyannote.audio silero-vad
```

### 0.3 Activate it (every new terminal)

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1
```

The prompt should start with `(.venv-qwencleo)`. If you see `ModuleNotFoundError: No module named 'qwencleo_asr'`, you are in the wrong environment (for example `.venv`). Run `deactivate`, then activate `.venv-qwencleo`.

### 0.4 Check everything works

```powershell
python craig_transcribe.py --check-env
```

---

## Part A: Craig recordings (`craig_transcribe.py`)

### A.1 Transcribe a session
Pass the Craig `.zip` directly; there's no need to extract it. Use a **new output folder per recording**:

```powershell
python craig_transcribe.py "lYTb4e_RvE3iMkK4werllBBrVxIcps.sesx.zip" -o out-5oTKajQzwGEm --speaker-map speaker_map.txt
```

The output folder (`-o`) will contain:

| File | What it is |
|---|---|
| `transcript.txt` | Readable transcript, grouped by speaker. **Start here.** |
| `transcript.srt` | Subtitles with timestamps (opens in VLC next to the audio). |
| `transcript.json` | Everything, for scripts. |
| `dropped.txt` | Lines the filters threw away as likely hallucinations. Check it if something seems missing. |
| `segments.jsonl` | Raw progress log used for resume. Don't edit it. |
| `recording.json` | Which recording and settings this folder belongs to. |

### A.2 Stop and resume
Press `Ctrl+C` any time. To continue, run **the exact same command** again; finished parts are skipped. The script refuses to mix a different recording into the same folder.

### A.3 Speaker names: `speaker_map.txt`
Craig names tracks by Discord username. `speaker_map.txt` renames them:

```
ahmedhoss61  = Kael
geniusgamerm = Dungeon Master
jockie*      = SKIP      # music bot: not transcribed at all
```

- The left side ignores case, and `*` matches anything.
- `SKIP` drops that track.
- If a run shows a raw username, add a line for it and use A.5 to rename without re-transcribing.

### A.4 Try it on a few minutes first

```powershell
# Only the first 5 minutes
python craig_transcribe.py "session.zip" -o test-out --speaker-map speaker_map.txt --limit 300

# Just show how much speech there is per player; no GPU
python craig_transcribe.py "session.zip" -o test-out --speaker-map speaker_map.txt --dry-run

# A window in the middle: 03:15:00 to 03:20:00 (seconds = h*3600 + m*60)
python craig_transcribe.py "session.zip" -o test-window --speaker-map speaker_map.txt --start 11700 --limit 12000
```

To compare settings, run the same window into a separate folder per setting and read the two `transcript.txt` files side by side.

### A.5 Re-write the transcript without re-transcribing
Use this after editing `speaker_map.txt`, or after the hallucination filters are updated:

```powershell
python craig_transcribe.py --rewrite-outputs -o out-5oTKajQzwGEm --speaker-map speaker_map.txt
```

It takes seconds and doesn't use the GPU.

### A.6 Older output folders (`--min-gap`)
Folders made before the "whole sentences" change (`out`, `out-AqliOggnsXbo`, `out-5oTKajQzwGEm`) used the older 0.45 s setting. The script stops and tells you if you mix settings.

- **To finish or rewrite an old folder as-is**, add `--min-gap 0.45`.
- **For the better, whole-sentence version**, transcribe again into a **new** folder, e.g. `-o out-AqliOggnsXbo-v2`.

### A.7 Useful options

| Option | Default | When to change it |
|---|---|---|
| `--min-peak-db` | `-40` | Skips near-silent chunks (mic hiss) that the model turns into invented sentences. Use `-35` if you still see junk such as "web development"; use `-45` if a quiet player loses real lines. |
| `--min-gap` | `1.5` | Pause length (seconds) that splits a player's speech. Higher gives longer lines. |
| `--batch-seconds` | `120` | Lower it (e.g. `60`) if you get out-of-memory errors or the run is very slow. |
| `--batch-size` | `16` | Max chunks per GPU batch. |
| `--beams` | `1` | Beam search. Tested with `4` on 5 minutes checked against the audio: no real gain (12 lines better, 11 worse) and 2 more invented lines, at 2–3× the time. Leave at `1`. |
| `--start` | none | Start N seconds into the recording, for testing a window (see A.4). |
| `--min-speech-ratio` | `0.25` | Clips up to 6 s where a speech detector hears talking in less than 25% of the clip are skipped (laughs, breaths, lip smacks the model writes sentences for). `0` turns it off; `0.35` is stricter but may cut real one-word replies. |
| `--language` | `auto` | `ar` forces Arabic output; `en` forces English. |
| `--scratch-dir` | system temp | Temporary audio needs about 1 GB per player per 8 hours. Point it at a roomy drive if C: is tight. |

Speed: about 4× realtime. One hour of *speech* (not recording length) takes about 15 minutes.

---

## Part B: one mixed recording (`phone_transcribe.py`)

The script does three things:

1. **Who spoke when.** Splits the recording into speaker turns (diarization).
2. **Who is who.** Names each speaker by comparing their voice with saved voice profiles.
3. **Transcribe.** Uses the same model, filters and resume as Part A.

### B.1 Build the voice profiles (once)
The profiles are learned from your Craig recordings, where each track is one known person:

```powershell
python phone_transcribe.py --enroll recording.zip "lYTb4e_RvE3iMkK4werllBBrVxIcps.sesx.zip" "craig-AqliOggnsXbo-Kxc5cerDJMktKvezsVw_n-MIqBn-6z.sesx.zip" --speaker-map speaker_map.txt
```

- This writes `voices.npz` and takes about 2 minutes per recording. **It is already done** for these three.
- Each new Craig session you enroll **adds** to the profiles, which makes matching more reliable. To add one:
  ```powershell
  python phone_transcribe.py --enroll "new-session.zip" --speaker-map speaker_map.txt
  ```
- To start over, delete `voices.npz`.
- `voices.npz` contains your friends' voice prints, so it is kept out of git on purpose.

### B.2 Transcribe the recording
Any audio or video format ffmpeg reads works (m4a, mp3, wav, ogg, mp4, ...):

```powershell
python phone_transcribe.py "session.m4a" -o out-phone-session1 --num-speakers 5
```

Always pass `--num-speakers` with the number of people who talk; it matters a lot for accuracy.

The console shows the speakers it found and how sure it is about each:

```
Speakers:
  SPEAKER_00     8.4 min  -> Dungeon Master   (closest voice: Dungeon Master 0.84)
  SPEAKER_04     0.7 min  -> Arnok            (closest voice: Arnok 0.59)
```

- Scores above about **0.5** are confident.
- A speaker scoring under 0.45 isn't named and shows up as `Speaker N`.

Besides the files from A.1, the output folder gets:

| File | What it is |
|---|---|
| `speaker_samples\SPEAKER_00.wav` ... | A ~12-second clip of each speaker. **Listen to these to check the names.** |
| `speakers.json` | Speaker → name, with the similarity scores. |
| `diarization.json` | The saved "who spoke when". Re-runs reuse it instead of redoing it. |

Resume works like A.2: run the same command again.

### B.3 Fix or set names (no re-transcribing)
Make a text file, e.g. `phone_names.txt`:

```
SPEAKER_03 = Thalion        # rename by speaker id
Speaker 6  = Guest          # or by the name currently shown
Vaylor     = SKIP           # hide someone from the transcript
```

Then run:

```powershell
python phone_transcribe.py --rewrite-outputs -o out-phone-session1 --speaker-map phone_names.txt
```

### B.4 Better speaker detection with pyannote (optional, recommended)
Without setup, the script uses its **built-in** detector, which scored 96% correct in testing. **pyannote** is usually more accurate, especially when people talk over each other. It is free but needs a Hugging Face account:

1. Create a token (type "Read") at https://hf.co/settings/tokens.
2. Open https://hf.co/pyannote/speaker-diarization-community-1 and accept the conditions.
3. Give the token to the script, in either of two ways:
   ```powershell
   # Option 1: for this terminal only
   $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"

   # Option 2: save it once for good
   hf auth login
   ```

After that, `phone_transcribe.py` uses pyannote automatically (`--diarizer auto`). To force one or the other:

```powershell
--diarizer pyannote   # error if it can't load (shows why)
--diarizer basic      # always the built-in one
```

### B.5 Redo the speaker detection
The "who spoke when" result is saved. To redo it, for example with a different `--num-speakers` or after setting up pyannote, use a new `-o` folder. Alternatively, delete `diarization.json` **and** `segments.jsonl` from the old folder.

### B.6 Try it on a few minutes first

```powershell
# Speakers and names only, no transcription (about 1 minute)
python phone_transcribe.py "session.m4a" -o test-phone --num-speakers 5 --dry-run

# First 5 minutes, fully transcribed
python phone_transcribe.py "session.m4a" -o test-phone5 --num-speakers 5 --limit 300
```

### B.7 Useful options

| Option | Default | Meaning |
|---|---|---|
| `--num-speakers N` | none | Number of people talking. **Always set it.** |
| `--min-speakers` / `--max-speakers` | none | Use these instead when you're not sure (pyannote only). |
| `--voices FILE` | `voices.npz` | Which profile file to use. |
| `--diarizer` | `auto` | `auto`, `pyannote` or `basic` (see B.4). |
| `--hf-token` | `$HF_TOKEN` | Token for pyannote, passed on the command line. |
| `--min-gap` | `1.5` | Joins one person's turns across pauses up to this many seconds. |
| `--min-peak-db` | `-50` | Skips near-silent chunks (laxer than Craig because phones always have room noise). |
| `--limit`, `--dry-run`, `--batch-seconds`, `--language`, `--scratch-dir` | | Same as Part A. |

Speed: about 15 minutes per hour of recording on the RTX 3050.

### B.8 Tips for recording with a phone
- Put the phone **in the middle of the table**, not next to one person.
- Use a recorder app that saves **WAV**, or m4a at **128 kbps or higher**.
- Turn off "voice note" noise reduction or "speech enhancement" if the app has it.
- Fewer people talking at once means better speaker names and better text.
- Quiet speakers (e.g. Arnok) are the hardest; sitting closer to the phone helps.

---

## Part C: fix misheard words with Gemini (`llm_correct.py`)

The speech model gets names, D&D terms and fast slang wrong: "religion check" comes out as "الريجيت شيك" and "Arnok" as "ارضك". After transcribing (Part A or B), this sends Gemini **the audio** of each window together with the transcript lines and your glossary. Gemini fixes what it hears is wrong, and marks lines where nobody actually spoke.

### C.1 Setup (once)
- **API keys:** put them in `.env` next to the scripts:
  ```
  GEMINI_API=your-first-key
  GEMINI_API_2=your-second-key
  ```
  Get keys at https://aistudio.google.com/apikey. When one key runs out of quota, the next takes over. `.env` is kept out of git.
- **Glossary:** edit `glossary.txt` with your campaign's names and words (NPCs, places, items, how names get misheard). It is already seeded with what we know. Names matter most; Gemini knows the D&D rules already.

### C.2 Run it

```powershell
# Craig session: pass the same zip and speaker map
python llm_correct.py -o out-v3owGZBtlLQA --audio "craig-v3owGZBtlLQA-kLUAHXQ8yB5gQu8PPAhxF3gfZdDmYF.sesx.zip" --speaker-map speaker_map.txt

# Phone session: pass the recording
python llm_correct.py -o out-phone-session1 --audio "session.m4a"
```

- **One call per 30 minutes of session.** A 6-hour session needs 12 calls; the free tier gives about 20 per key per day.
- **Progress is saved after every call.** If the daily limit is reached, run the same command the next day and it continues.
- **Try one call first:** add `--max-calls 1`. To see the plan without spending calls: `--dry-run`.

### C.3 What you get
| File | What it is |
|---|---|
| `transcript.txt` / `.srt` / `.json` | Now the **corrected** version. In `.json`, changed lines keep the old text in `original_text`. |
| `transcript-uncorrected.txt` | The transcript before correction, to compare. |
| `corrections.json` | Every change Gemini made (`text`, or `drop` for "nobody spoke"), with the old text in `was`. |
| `dropped.txt` | Includes the lines Gemini marked as no speech (`llm-no-speech`). |

`segments.jsonl` is never changed, so nothing is lost:
- `--rewrite-outputs` (Part A/B) keeps the corrections.
- `python llm_correct.py -o FOLDER --undo` removes them all.
- `--redo` throws them away and corrects again, e.g. after improving the glossary.

### C.4 Options
| Option | Default | Meaning |
|---|---|---|
| `--glossary` | `glossary.txt` | Your names and terms. |
| `--window-min` | `30` | Minutes of session per call. Larger means fewer calls, but Gemini may skim. |
| `--model` | `gemini-2.5-flash` | Which Gemini model. `--list-models` shows what your key can use. |
| `--daily-limit` | `20` | Calls per key per day; the script stops before Google does. |
| `--max-calls` | none | Stop after N calls in this run. |
| Without `--audio` | | Text-only mode: Gemini only fixes obvious things (glossary names, English spelled in Arabic letters). Much weaker; use the audio. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'qwencleo_asr'` / `'pyannote'` | Wrong environment. Run `deactivate`, then `.\.venv-qwencleo\Scripts\Activate.ps1`. |
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, then open a new terminal. |
| `error: ... already holds recording X` | That folder belongs to another recording. Use a different `-o`. |
| `error: ... was transcribed with --min-gap 0.45` | Old folder; see A.6. |
| Very slow, GPU memory full | Windows spills VRAM into system RAM instead of crashing. Close other GPU apps (browser, games), or use `--batch-seconds 60`. |
| `pyannote unavailable (GatedRepoError ...)` | No token, or the model's conditions weren't accepted; see B.4. The script falls back to the built-in detector meanwhile. |
| Invented lines ("web development", "بحبك ... بحبني") | Raise `--min-peak-db` (e.g. `-35`) in a new folder. New filter patterns can be applied to old folders with `--rewrite-outputs`. |
| `llm_correct.py`: "out of quota" on every key | The free daily limit is used up. Run the same command tomorrow; finished windows are kept. |
| `llm_correct.py`: a window "failed" | Usually temporary (server busy). Run the same command again; only unfinished windows are sent. |
| Wrong name on a phone speaker | Listen to `speaker_samples\`, then fix it with B.3. Enrolling more Craig sessions (B.1) improves future matches. |

---

## Cheat sheet

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1

# Craig zip
python craig_transcribe.py "SESSION.zip" -o out-SESSION --speaker-map speaker_map.txt
python craig_transcribe.py --rewrite-outputs -o out-SESSION --speaker-map speaker_map.txt

# Phone recording
python phone_transcribe.py --enroll "CRAIG.zip" --speaker-map speaker_map.txt   # add voices (optional)
python phone_transcribe.py "SESSION.m4a" -o out-phone-SESSION --num-speakers 5
python phone_transcribe.py --rewrite-outputs -o out-phone-SESSION --speaker-map phone_names.txt

# Gemini correction (after either of the above; keys in .env)
python llm_correct.py -o out-SESSION --audio "SESSION.zip" --speaker-map speaker_map.txt
```

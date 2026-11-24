# How to run everything

Everything goes through one command, `scribe`:

| You want to | Command | Part |
|---|---|---|
| Transcribe a Craig zip from Discord (one track per player) | `scribe craig` | A |
| Transcribe one mixed recording (a phone on the table, a Zoom export, ...) | `scribe phone` | B |
| Fix misheard names and terms with Gemini | `scribe correct` | C |
| Ask questions about your documents and sessions | `scribe serve` (chat UI) or `scribe rag` | D |

All commands below are for **PowerShell**, run from the project folder
`G:\AI_ML\Projects\Criag_transcriber`. Recordings go in `data\recordings\`,
transcripts land in `data\transcripts\<recording id>\`, and the documents the
archive answers from go in `data\documents\`.

---

## 0. One-time setup

### 0.1 Requirements
- An NVIDIA GPU with CUDA (tested on an RTX 3050 8 GB).
- `ffmpeg` on the PATH. Check with `ffmpeg -version`. If it's missing: `winget install Gyan.FFmpeg`.
- `uv`. Check with `uv --version`. If it's missing: `winget install --id=astral-sh.uv -e`.

### 0.2 Create the environment
The `.venv-qwencleo` environment already exists on this machine, with the project installed in it. To rebuild it from scratch (from **cmd**, not PowerShell):

```cmd
scripts\setup.cmd qwencleo
```

This installs torch, the speech model, and this project (`scribe`, the archive and the chat UI).

If you ever change `pyproject.toml` or move the folder, re-install the project into the environment:

```powershell
uv pip install --python .\.venv-qwencleo\Scripts\python.exe -e ".[rag,server,correct]"
```

### 0.3 Activate it (every new terminal)

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1
```

The prompt should start with `(.venv-qwencleo)`. If `scribe` is not found, or you see `No module named 'qwencleo_asr'`, you are in the wrong environment (for example `.venv`). Run `deactivate`, then activate `.venv-qwencleo`.

### 0.4 Check everything works

```powershell
scribe craig --check-env
```

---

## Part A: Craig recordings (`scribe craig`)

### A.1 Transcribe a session
Put the Craig `.zip` in `data\recordings\` (no need to extract it) and run:

```powershell
scribe craig "data\recordings\craig-v3owGZBtlLQA-kLUAHXQ8yB5gQu8PPAhxF3gfZdDmYF.sesx.zip"
```

- The output goes to `data\transcripts\<recording id>\`, here `data\transcripts\v3owGZBtlLQA\`. Use `-o FOLDER` to choose another.
- Speaker names come from `config\speaker_map.txt` automatically (see A.3).

The output folder will contain:

| File | What it is |
|---|---|
| `transcript.txt` | Readable transcript, grouped by speaker. **Start here.** |
| `transcript.srt` | Subtitles with timestamps (opens in VLC next to the audio). |
| `transcript.json` | Everything, for scripts. The archive (Part D) reads this. |
| `dropped.txt` | Lines the filters threw away as likely hallucinations. Check it if something seems missing. |
| `segments.jsonl` | Raw progress log used for resume. Don't edit it. |
| `recording.json` | Which recording and settings this folder belongs to. |

### A.2 Stop and resume
Press `Ctrl+C` any time. To continue, run **the exact same command** again; finished parts are skipped. The command refuses to mix a different recording into the same folder.

### A.3 Speaker names: `config\speaker_map.txt`
Craig names tracks by Discord username. `config\speaker_map.txt` renames them:

```
ahmedhoss61  = Kael
geniusgamerm = Dungeon Master
jockie*      = SKIP      # music bot: not transcribed at all
```

- The left side ignores case, and `*` matches anything.
- `SKIP` drops that track.
- If a run shows a raw username, add a line for it and use A.5 to rename without re-transcribing.
- To use a different file, pass `--speaker-map FILE`.

### A.4 Try it on a few minutes first

```powershell
# Only the first 5 minutes
scribe craig "data\recordings\session.zip" -o data\experiments\test --limit 300

# Just show how much speech there is per player; no GPU
scribe craig "data\recordings\session.zip" --dry-run

# A window in the middle: 03:15:00 to 03:20:00 (seconds = h*3600 + m*60)
scribe craig "data\recordings\session.zip" -o data\experiments\window --start 11700 --limit 12000
```

Use `data\experiments\` for tests: the archive only reads `data\transcripts\`, so trial runs don't end up in the answers. To compare settings, run the same window into a separate folder per setting and read the two `transcript.txt` files side by side.

### A.5 Re-write the transcript without re-transcribing
Use this after editing `config\speaker_map.txt`, or after the hallucination filters are updated:

```powershell
scribe craig --rewrite-outputs -o data\transcripts\v3owGZBtlLQA
```

It takes seconds and doesn't use the GPU.

### A.6 Older output folders (`--min-gap`)
Folders made before the "whole sentences" change used the older 0.45 s setting. The command stops and tells you if you mix settings.

- **To finish or rewrite an old folder as-is**, add `--min-gap 0.45`.
- **For the better, whole-sentence version**, transcribe again into a **new** folder.

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

## Part B: one mixed recording (`scribe phone`)

The command does three things:

1. **Who spoke when.** Splits the recording into speaker turns (diarization).
2. **Who is who.** Names each speaker by comparing their voice with saved voice profiles.
3. **Transcribe.** Uses the same model, filters and resume as Part A.

### B.1 Build the voice profiles (once)
The profiles are learned from your Craig recordings, where each track is one known person:

```powershell
scribe phone --enroll "data\recordings\craig-F4fL0XAStKft-NbceN5GheS_qljB8l0sFInUlcqgBPm.sesx.zip" "data\recordings\craig-v3owGZBtlLQA-kLUAHXQ8yB5gQu8PPAhxF3gfZdDmYF.sesx.zip" --speaker-map config\speaker_map.txt
```

- This writes `data\voices.npz` and takes about 2 minutes per recording. **It is already done** for three earlier sessions.
- Each new Craig session you enroll **adds** to the profiles, which makes matching more reliable.
- To start over, delete `data\voices.npz`.
- `voices.npz` contains your friends' voice prints, so it is kept out of git on purpose.

### B.2 Transcribe the recording
Any audio or video format ffmpeg reads works (m4a, mp3, wav, ogg, mp4, ...):

```powershell
scribe phone "data\recordings\session.m4a" --num-speakers 5
```

The output goes to `data\transcripts\phone-session\` (or `-o FOLDER`). Always pass `--num-speakers` with the number of people who talk; it matters a lot for accuracy.

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
Make a text file, e.g. `config\phone_names.txt`:

```
SPEAKER_03 = Thalion        # rename by speaker id
Speaker 6  = Guest          # or by the name currently shown
Vaelor     = SKIP           # hide someone from the transcript
```

Then run:

```powershell
scribe phone --rewrite-outputs -o data\transcripts\phone-session --speaker-map config\phone_names.txt
```

### B.4 Better speaker detection with pyannote (optional, recommended)
Without setup, the command uses its **built-in** detector, which scored 96% correct in testing. **pyannote** is usually more accurate, especially when people talk over each other. It is free but needs a Hugging Face account:

1. Create a token (type "Read") at https://hf.co/settings/tokens.
2. Open https://hf.co/pyannote/speaker-diarization-community-1 and accept the conditions.
3. Give the token to the command, in either of two ways:
   ```powershell
   # Option 1: for this terminal only
   $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"

   # Option 2: save it once for good
   hf auth login
   ```

After that, `scribe phone` uses pyannote automatically (`--diarizer auto`). To force one or the other:

```powershell
--diarizer pyannote   # error if it can't load (shows why)
--diarizer basic      # always the built-in one
```

### B.5 Redo the speaker detection
The "who spoke when" result is saved. To redo it, for example with a different `--num-speakers` or after setting up pyannote, use a new `-o` folder. Alternatively, delete `diarization.json` **and** `segments.jsonl` from the old folder.

### B.6 Try it on a few minutes first

```powershell
# Speakers and names only, no transcription (about 1 minute)
scribe phone "data\recordings\session.m4a" -o data\experiments\phone --num-speakers 5 --dry-run

# First 5 minutes, fully transcribed
scribe phone "data\recordings\session.m4a" -o data\experiments\phone5 --num-speakers 5 --limit 300
```

### B.7 Useful options

| Option | Default | Meaning |
|---|---|---|
| `--num-speakers N` | none | Number of people talking. **Always set it.** |
| `--min-speakers` / `--max-speakers` | none | Use these instead when you're not sure (pyannote only). |
| `--voices FILE` | `data\voices.npz` | Which profile file to use. |
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

## Part C: fix misheard words with Gemini (`scribe correct`)

The speech model gets names, D&D terms and fast slang wrong: "religion check" comes out as "الريجيت شيك" and "Arnok" as "ارضك". After transcribing (Part A or B), this sends Gemini **the audio** of each window together with the transcript lines and your glossary. Gemini fixes what it hears is wrong, and marks lines where nobody actually spoke.

### C.1 Setup (once)
- **API keys:** put them in `.env` in the project folder (see `.env.example`):
  ```
  GEMINI_API=your-first-key
  GEMINI_API_2=your-second-key
  ```
  Get keys at https://aistudio.google.com/apikey. When one key runs out of quota, the next takes over. `.env` is kept out of git.
- **Glossary:** edit `config\glossary.txt` with your campaign's names and words (NPCs, places, items, how names get misheard). Names matter most; Gemini knows the D&D rules already.

### C.2 Run it

```powershell
# Craig session: pass the same zip (config\speaker_map.txt is used for SKIP tracks)
scribe correct -o data\transcripts\v3owGZBtlLQA --audio "data\recordings\craig-v3owGZBtlLQA-kLUAHXQ8yB5gQu8PPAhxF3gfZdDmYF.sesx.zip"

# Phone session: pass the recording
scribe correct -o data\transcripts\phone-session --audio "data\recordings\session.m4a"
```

- **One call per 30 minutes of session.** A 6-hour session needs 12 calls; the free tier gives about 20 per key per day.
- **Progress is saved after every call.** If the daily limit is reached, run the same command the next day and it continues.
- **Try one call first:** add `--max-calls 1`. To see the plan without spending calls: `--dry-run`.
- The call budget is shared with the archive's answers (Part D), counted per key and per model in `data\.gemini_usage.json`.

### C.3 What you get
| File | What it is |
|---|---|
| `transcript.txt` / `.srt` / `.json` | Now the **corrected** version. In `.json`, changed lines keep the old text in `original_text`. |
| `transcript-uncorrected.txt` | The transcript before correction, to compare. |
| `corrections.json` | Every change Gemini made (`text`, or `drop` for "nobody spoke"), with the old text in `was`. |
| `dropped.txt` | Includes the lines Gemini marked as no speech (`llm-no-speech`). |

`segments.jsonl` is never changed, so nothing is lost:
- `--rewrite-outputs` (Part A/B) keeps the corrections.
- `scribe correct -o FOLDER --undo` removes them all.
- `--redo` throws them away and corrects again, e.g. after improving the glossary.

### C.4 Options
| Option | Default | Meaning |
|---|---|---|
| `--glossary` | `config\glossary.txt` | Your names and terms. |
| `--window-min` | `30` | Minutes of session per call. Larger means fewer calls, but Gemini may skim. |
| `--model` | `gemini-2.5-flash` | Which Gemini model. `--list-models` shows what your key can use. |
| `--daily-limit` | `20` | Calls per key per day; the command stops before Google does. |
| `--max-calls` | none | Stop after N calls in this run. |
| Without `--audio` | | Text-only mode: Gemini only fixes obvious things (glossary names, English spelled in Arabic letters). Much weaker; use the audio. |

After correcting, run `scribe rag ingest` (or **Re-index** in the UI) so the archive picks up the corrected text.

---

## Part D: the archive (ask questions about your files)

A small RAG system over two kinds of files:

- **Your documents** in `data\documents\`: rulebooks, campaign notes, character sheets, homebrew. PDF, Markdown, Word (`.docx`), text and HTML are read, in any subfolder. It starts with `lore\`, `characters\`, `sessions\` (hand-written recaps) and `rules\`; the folders are only for you to keep things tidy, and you can add more.

The page is named after your party, set under `[party]` in `config\rag.toml` (`name = "VAKT"`). The LLM is told the name too, so "What has VAKT done so far?" is understood as a question about your group. Put the party's emblem at `assets\emblem.png` (or `.svg`, `.webp`, `.jpg`) and restart `scribe serve`: it replaces the built-in sigil in the header and becomes the browser-tab icon.
- **Session transcripts** in `data\transcripts\*\transcript.json`, added automatically.

### D.1 How an answer is made
1. **Router.** A small classifier decides whether the question needs **one search** ("What does Ritual Adept do?") or **several** ("Compare what the cult and the guardians want"). It is trained on the examples in `config\router_examples.jsonl`.
2. **Search.** Each search combines meaning (the `bge-m3` embedding model, which matches English and Arabic to each other) with exact words (BM25, for names like "Arnok").
3. **Rerank.** A local cross-encoder (`bge-reranker-v2-m3`) reads the question with each candidate passage and keeps the most relevant.
4. **Multi-step only.** The LLM splits the question into sub-questions, each is searched, and if something is still missing it runs one more round of searches.
5. **Answer.** The LLM (Gemini, or a local Ollama model) writes the answer from the passages only, citing them as `[1]`, `[2]`.

### D.2 First run

```powershell
# 1. Add your files
copy "C:\path\to\Players Handbook.pdf" data\documents\

# 2. Build the index. The first time downloads the two models (about 4.5 GB)
scribe rag ingest

# 3. Start the chat UI, then open http://127.0.0.1:8000
scribe serve
```

Afterwards, `scribe rag ingest` only re-reads files that are new or changed, so it takes seconds. In the UI, the **Library** tab does the same: drop files on it to upload and index them, and **Re-index** after adding files by hand or finishing a transcript.

### D.3 The chat UI
- **Auto / One-step / Multi-step** under the question box: Auto lets the router decide; the other two force a mode.
- Each answer shows how it was routed (and the router's probability), and how many searches it ran.
- **Sources** (right side): the passages the answer used, numbered like the citations. Click a `[2]` in the answer to jump to that passage. **Show context** loads the passages just before and after it. The scores are the reranker's relevance (`rel`), the meaning similarity (`sim`) and the keyword score (`kw`).
- Conversations are kept in your browser. A follow-up like "and what about him?" is searched together with your previous question.
- The status panel (bottom left) shows the models, the index size, and how many Gemini calls are left today.

### D.4 From the terminal

```powershell
scribe rag status                       # what is indexed
scribe rag ask "What does Ritual Adept do?"
scribe rag ask --mode multi "Compare the cult and the guardians"
scribe rag search "celestial shards"    # retrieval only, no LLM call: check what it finds
scribe rag classify "Who has dark vision in the party?"
scribe rag router-eval                  # router accuracy on its examples
scribe rag ingest --rebuild             # re-embed everything
```

### D.5 Which LLM answers
Set in `config\rag.toml` under `[llm]`. Providers are tried in order:

```toml
providers = ["gemini", "ollama"]
```

- **Gemini** uses the keys in `.env`. Free tier: about 20 calls per key per model per day, shared with `scribe correct`. A one-step question costs **1 call**; a multi-step question costs **2–3** (plan, maybe one follow-up, answer). To save calls, set `max_hops = 1` under `[multistep]`, or point `[llm.gemini] model` at a model with a bigger free quota (`scribe correct --list-models` lists them).
- **Ollama** runs a model on your own GPU: no limit, nothing leaves the computer. Install from https://ollama.com, then:
  ```powershell
  ollama pull qwen3:4b
  ```
  `qwen3:4b` fits next to the embedder and reranker in 8 GB. When Gemini is out of calls, the archive switches to it automatically. To use only Ollama, set `providers = ["ollama"]`.

### D.6 The router
- `scribe rag router-eval` shows the classifier's cross-validated accuracy on `config\router_examples.jsonl`, next to plain keyword rules, and lists the questions it gets wrong.
- To teach it, add lines to `config\router_examples.jsonl` (`"single"` or `"multi"`), especially questions it routed wrongly. It retrains in under a second the next time it starts.
- `[router] mode` in `rag.toml`: `classifier` (default), `heuristic` (keyword rules, no training), or `single` / `multi` to always use one path.

### D.7 Trying other models
All in `config\rag.toml`:

| Setting | Default | Notes |
|---|---|---|
| `[embedding] model` | `BAAI/bge-m3` | Any Hugging Face embedding model. Set `pooling` to match it (`cls` for bge, `mean` for e5, `last` for Qwen3-Embedding), and the prefixes for models that want them (`query: ` / `passage: ` for e5). Changing it rebuilds the index. |
| `[reranker] model` | `BAAI/bge-reranker-v2-m3` | Any cross-encoder. `enabled = false`, or `scribe serve --no-rerank`, to compare answers without it. |
| `[retrieval] final_k` | `6` | Passages an answer is written from (one-step). |
| `[chunking] chunk_words` | `220` | Passage size. Changing it rebuilds the index. |

`scribe rag search "..."` is the quickest way to see how a change affects what is found.

### D.8 Good documents
- **Character sheets:** fillable PDFs (the official 5e sheet, D&D Beyond exports) work; what was typed in the form fields is read too. Name each file after the character (`characters\Kael.pdf`), since answers cite the file name. `scribe rag read FILE` prints exactly what the archive gets from a file, without loading any models.
- Text PDFs work; **scanned** PDFs (images of pages) have no text to read and come out empty. Run OCR on them first.
- Markdown headings (`#`, `##`) and Word heading styles are kept, so answers can cite "Campaign notes > The Cult".
- One topic per file is easier to manage than one giant file: deleting or updating it only touches its own passages.

### D.9 The party's pages
The **Party** tab in the site has a page for each character in `config\party.toml`: the four player characters, and a **Companions** page with a card for each companion. A page shows:
- the portrait, name, title and who plays them;
- their sheet (`sheet = "data/documents/characters/Kael.pdf"`), laid out for reading;
- on the right, every session they're in: how much they spoke, and each line where someone else names them;
- buttons that ask the archive about them.

To fill in a character, edit their entry in `config\party.toml` (title, summary, sheet, aliases) and reload the page; no restart needed. Each character has a folder in `assets\` named after them (`assets\Kael\`, `assets\Hasaneen\`, ...): their page shows `portrait.png` from it (`.jpg` and `.webp` work too), or the first image in the folder. Other art can live there as well.

**Aliases matter.** The transcripts write names in Arabic (كايل, حسنين) or misheard (رادش for Raddis). Every spelling in `aliases` is found on the page, and a question that uses one spelling also searches for the others. If a page says someone never comes up, look for how the transcripts spell them and add it.

### D.10 Older or lower-quality transcripts
- **Only the `transcript.txt` from an earlier run** (lines like `[00:00:06.299] Dungeon Master:`): import it, from wherever it is:
  ```powershell
  scribe import "C:\old\transcript.txt"
  scribe import "C:\old\session5.txt" --id session-05 --date 2026-03-14   # no '# Recording' / '# Start time' header
  ```
  It becomes `data\transcripts\<recording id>\` with a `transcript.json`, so it counts as a real session (character pages included). Speaker names go through `config\speaker_map.txt`, so Discord names and the old "Vaylor" come out right. Times inside a turn are estimated; turn starts are exact. If the same recording was already transcribed properly, the import refuses, since that version is better.
- **Folders from this tool** (with a `transcript.json` inside, e.g. an old `out-AqliOggnsXbo`): copy the folder into `data\transcripts\`, then re-index. They count like any other session, including on the character pages. If the old folder used the 0.45 s setting, refresh its speaker names and filters with `scribe craig --rewrite-outputs -o data\transcripts\FOLDER --min-gap 0.45`. If you still have the Craig zip, transcribing it again gives a much better transcript.
- **Text or Word files from elsewhere**: put them in `data\documents\sessions\`, named so you can tell them apart (`Session 05 - 2026-03-14.txt`). Answers can cite them, and character pages list them under "In your documents", but the per-session speaking counts only come from `transcript.json` folders.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `scribe` not found / `No module named 'qwencleo_asr'` | Wrong environment. Run `deactivate`, then `.\.venv-qwencleo\Scripts\Activate.ps1`. If it still isn't found, re-install the project (0.2). |
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, then open a new terminal. |
| `error: ... already holds recording X` | That folder belongs to another recording. Use a different `-o`. |
| `error: ... was transcribed with --min-gap 0.45` | Old folder; see A.6. |
| Very slow, GPU memory full | Windows spills VRAM into system RAM instead of crashing. Close other GPU apps (browser, games, `scribe serve`), or use `--batch-seconds 60`. |
| `pyannote unavailable (GatedRepoError ...)` | No token, or the model's conditions weren't accepted; see B.4. The command falls back to the built-in detector meanwhile. |
| Invented lines ("web development", "بحبك ... بحبني") | Raise `--min-peak-db` (e.g. `-35`) in a new folder. New filter patterns can be applied to old folders with `--rewrite-outputs`. |
| `scribe correct`: "out of quota" on every key | The free daily limit is used up. Run the same command tomorrow; finished windows are kept. |
| `scribe correct`: a window "failed" | Usually temporary (server busy). Run the same command again; only unfinished windows are sent. |
| Wrong name on a phone speaker | Listen to `speaker_samples\`, then fix it with B.3. Enrolling more Craig sessions (B.1) improves future matches. |
| UI says "No language model could answer" | Gemini is out of calls and Ollama isn't running. Start Ollama (D.5), or wait for the daily reset. The sources are still shown. |
| UI says "The archive is empty" | Add files to `data\documents\` and press **Re-index** (or `scribe rag ingest`). |
| A PDF gives 0 passages | It's scanned; see D.8. |
| Answers miss something you know is there | `scribe rag search "..."` to see what retrieval finds. Try the other mode (One-step / Multi-step), or name the thing exactly as it's written. |
| Transcribing while `scribe serve` runs is slow | The archive's models use about 2.5 GB of VRAM. Stop the server while transcribing. |

---

## Cheat sheet

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1

# Craig zip (put it in data\recordings first)
scribe craig "data\recordings\SESSION.zip"
scribe craig --rewrite-outputs -o data\transcripts\SESSION_ID

# Phone recording
scribe phone --enroll "data\recordings\CRAIG.zip" --speaker-map config\speaker_map.txt   # add voices (optional)
scribe phone "data\recordings\SESSION.m4a" --num-speakers 5
scribe phone --rewrite-outputs -o data\transcripts\phone-SESSION --speaker-map config\phone_names.txt

# Gemini correction (after either of the above; keys in .env)
scribe correct -o data\transcripts\SESSION_ID --audio "data\recordings\SESSION.zip"

# Archive
scribe import "OLD\transcript.txt"   # an old transcript as a session
scribe rag ingest         # after adding documents or transcripts
scribe serve              # chat UI on http://127.0.0.1:8000
```

# Craig Transcriber

Turns recordings of our Dungeons & Dragons sessions into readable transcripts,
with each line credited to the player who said it, and lets you **ask questions**
about the sessions, rulebooks and campaign notes in a chat UI that cites its
sources. Built for **Egyptian Arabic mixed with English** ("انت شايف المصباح، but
it doesn't seem that they have noticed you"), and runs locally on an 8 GB GPU.

| You have | Command | What you get |
|---|---|---|
| A **Craig** zip from Discord (one track per player) | `scribe craig` | Transcript; speakers from Discord usernames via `config/speaker_map.txt` |
| **One mixed recording** (phone on the table, Zoom export) | `scribe phone` | Transcript; speakers detected from the audio and named by voice |
| A finished transcript | `scribe correct` | Names, D&D terms and slang fixed by Gemini listening to the audio |
| Documents + transcripts | `scribe serve` / `scribe rag` | A chat that answers from your files, with citations |

**→ Step-by-step instructions are in [docs/GUIDE.md](docs/GUIDE.md).**

---

## Quick start

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1

# Transcribe a Craig recording -> data/transcripts/<recording id>/
scribe craig "data\recordings\craig-v3owGZBtlLQA-....sesx.zip"

# Ask questions: put PDFs / notes in data\documents, index, start the UI
scribe rag ingest
scribe serve            # http://127.0.0.1:8000
```

A transcript reads like this:

```
[00:00:27.827] Dungeon Master:
  [mixed] ما انا متحركين في اتجاه المعاكس فانت you can't really tell ان it's not very bright
  [mixed] even with your dark vision يعني انت شايف المصباح ...
```

The chat UI ("Lorekeeper") shows each answer with numbered citations, how the
question was routed (one search or several), and the source passages with their
relevance scores.

---

## How it works

### Transcription
1. **Who spoke when.** Craig gives one track per player, with exact digital
   silence when someone isn't talking, so speaker and speech timing are known
   exactly. A phone recording is split into speakers first (pyannote, or a
   built-in Silero VAD + voice embeddings + clustering), and each speaker is named
   by matching their voice against profiles learned from Craig sessions (96% of
   speaking time credited correctly in a test).
2. **Transcribe.** [QwenCleo-ASR](https://huggingface.co/mohammedaly22/QwenCleo-ASR),
   a Qwen3-ASR 1.7B model fine-tuned on Egyptian Arabic with English
   code-switching, keeps English in Latin script. Chunks of similar length are
   batched on the GPU.
3. **Filter hallucinations.** Near-silent chunks and clips with little speech are
   skipped; known invented phrases, repetition loops and sentences repeated by
   every player are dropped into `dropped.txt`.
4. **Correct (optional).** Gemini gets 30 minutes of audio at a time with the
   transcript lines and a glossary, and returns only the lines it hears are wrong.

### The archive (RAG)

```
question ─► router ─┬─ one-step ──► hybrid search ─► rerank ──────────────────┐
  (classifier on    │                                                          ├─► LLM answer
   the question's   └─ multi-step ─► LLM plans sub-queries ─► search each ─────┤   with [n]
   embedding)                        ─► one follow-up round if something's missing   citations
```

- **Passages.** Documents are cut into ~220-word passages under their headings
  (PDF pages, Markdown and Word headings); transcripts into windows of up to two
  minutes of consecutive lines.
- **Hybrid search.** `BAAI/bge-m3` embeddings (multilingual: an Arabic question
  finds English text) fused with BM25 keyword search (exact names like "Arnok")
  by reciprocal rank fusion.
- **Reranker.** `BAAI/bge-reranker-v2-m3`, a cross-encoder that reads question
  and passage together, reorders the shortlist.
- **Router.** Logistic regression over the question's embedding plus a few
  keyword counts, trained on `config/router_examples.jsonl` in under a second.
  `scribe rag router-eval` reports its accuracy.
- **LLM.** Gemini (free daily quota, keys rotate) with automatic fallback to a
  local model through Ollama.
- **Incremental index.** Only new or changed files are embedded again.

---

## Project layout

```
.
├── pyproject.toml              package, dependencies, the `scribe` command
├── config/
│   ├── speaker_map.txt         Discord username -> character name (SKIP for bots)
│   ├── glossary.txt            campaign names and terms for the Gemini correction
│   ├── hotwords.txt            vocabulary hints for the qwen3asr / whisper backends
│   ├── rag.toml                archive settings: models, retrieval, router, LLMs
│   └── router_examples.jsonl   questions labelled one-step / multi-step
├── data/
│   ├── recordings/             Craig zips, phone audio            (not in git)
│   ├── transcripts/<id>/       one folder per transcribed session
│   ├── experiments/            trial runs, kept out of the archive
│   ├── documents/              files the archive answers from
│   ├── index/                  the archive's index                (not in git)
│   └── voices.npz              players' voice profiles             (not in git)
├── docs/
│   ├── GUIDE.md                how to run everything
│   └── notes/                  design notes and session log
├── requirements/               per-backend speech dependencies
├── scripts/setup.cmd, setup.sh create the environment for a backend
├── src/craig_transcriber/
│   ├── cli/                    `scribe craig | phone | correct | rag | serve`
│   ├── transcription/          audio, chunking, ASR backends, filters, outputs, phone diarization
│   ├── correction/             the Gemini correction pass
│   ├── llm/                    Gemini and Ollama providers, API key pool
│   ├── rag/                    documents, chunking, index, models, router, retriever, engine
│   └── server/                 FastAPI app and the chat UI (static/)
└── tests/                      pytest suite; runs in seconds with fake models
```

---

## Requirements
- Windows or Linux, an NVIDIA GPU (8 GB is enough), Python 3.12
- `ffmpeg` and `uv` on the PATH
- For the archive's answers: a free Gemini API key in `.env` (see `.env.example`),
  and/or [Ollama](https://ollama.com) for a local model
- Optional: a free Hugging Face token for pyannote (see the [guide](docs/GUIDE.md#b4-better-speaker-detection-with-pyannote-optional-recommended))

Setup: `scripts\setup.cmd qwencleo` (creates `.venv-qwencleo` and installs the project).
Tests: `python -m pytest`.

## Limitations
- **Fantasy names get misheard.** Invented names and D&D terms are often wrong
  until `scribe correct` runs with a good glossary.
- **Some invented lines still get through.** On very quiet audio the model can
  still produce a short sentence that wasn't said; `dropped.txt` shows what the
  filters caught.
- **Phone recordings are harder than Craig.** Overlapping speech goes to the
  loudest person, and a speaker may be mislabelled at fast back-and-forth.
- **The archive only knows what's in the files.** Scanned PDFs need OCR first,
  and answers about sessions are only as good as their transcripts.
- **Free Gemini quota is small.** About 20 calls per key per day, shared between
  corrections and answers; a multi-step question uses 2–3. Ollama has no limit.

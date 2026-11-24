# Commands, in order

PowerShell, from `G:\AI_ML\Projects\Criag_transcriber`, with the environment active:

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1
```

---

## 1. Before importing: name the unknown players

Open `config\speaker_map.txt` and add (put venom6469's character name on the right):

```
venom6469     = CharacterName
alysalah      = SKIP
```

If venom6469's character should get a page, also add them to `config\party.toml` (copy one of the `[[character]]` blocks).

## 2. Rename Vaylor to Vaelor in the two current sessions (seconds, no GPU)

```powershell
scribe craig --rewrite-outputs -o data\transcripts\v3owGZBtlLQA
scribe craig --rewrite-outputs -o data\transcripts\F4fL0XAStKft
```

## 3. Import the old transcripts as sessions

Fix the dates if the year isn't 2026. For `transcript.txt`, add `--date YYYY-MM-DD` if you know it.

```powershell
scribe import "data\documents\sessions\12.txt" "data\documents\sessions\13.txt"
scribe import "data\documents\sessions\27.6.md" --id 2026-06-27 --date 2026-06-27
scribe import "data\documents\sessions\29.6.md" --id 2026-06-29 --date 2026-06-29
scribe import "data\documents\sessions\4.7.md" --date 2026-07-04
scribe import "data\documents\sessions\transcript.txt" --id undated-session
```

Each original moves into its new folder in `data\transcripts\`. Only `Notes for session 4.txt` stays in `data\documents\sessions\`.

Remove the stray copy of the notes from the transcripts folder:

```powershell
Remove-Item "data\transcripts\Notes for session 4.txt"
```

## 4. (Optional) Finish the Gemini correction of the 2026-11-21 session

2 windows left = 2 Gemini calls.

```powershell
scribe correct -o data\transcripts\v3owGZBtlLQA --audio "data\recordings\craig-v3owGZBtlLQA-kLUAHXQ8yB5gQu8PPAhxF3gfZdDmYF.sesx.zip"
```

## 5. Build the index

The first run downloads the two models (about 4.5 GB). This time it re-reads everything once, because PDF reading changed.

```powershell
scribe rag ingest
```

## 6. Start the site

```powershell
scribe serve
```

Open http://127.0.0.1:8000 and leave the terminal open (Ctrl+C stops it).
Character pages: http://127.0.0.1:8000/#kael, `#vaelor`, `#thalion`, `#arnok`, `#companions`.

## 7. Share it with the party (Option A: your PC + Tailscale Funnel)

Logins first; each command asks for that person's password (8+ characters):

```powershell
scribe users add Ahmed --admin
scribe users add Seif
scribe users add Hazem
scribe users add Abdo
```

Install Tailscale from https://tailscale.com/download and sign in. Then once (if it says access is denied, use an Administrator PowerShell):

```powershell
tailscale funnel --bg 8000
```

The first time it prints a link to switch Funnel on: open it, approve, run the command again. It prints the address to send to the party (`https://<your-pc>.<tailnet>.ts.net`). From then on only `scribe serve` is needed; the address stays the same.

Check which Flash-Lite model your key has (the site now answers with `gemini-flash-lite-latest`):

```powershell
scribe correct --list-models
```

---

## Later

### Answers when Gemini's daily calls run out (local model, no limit)

Install Ollama from https://ollama.com, then:

```powershell
ollama pull qwen3:4b
```

### A new session

```powershell
scribe craig "data\recordings\NEW-SESSION.zip"
scribe correct -o data\transcripts\SESSION_ID --audio "data\recordings\NEW-SESSION.zip"
scribe serve        # then Library > Re-index, signed in as admin
```

Stop `scribe serve` while transcribing: both use the GPU. Re-index from the site rather than `scribe rag ingest` while the server runs: the server only sees a new index after a re-index from the site or a restart.

### Logins

```powershell
scribe users list
scribe users passwd Seif      # new password, signs him out everywhere
scribe users remove Seif
tailscale funnel status       # is the site public?
tailscale funnel reset        # take it offline
```

### After adding or changing documents, PDFs or config\party.toml

```powershell
scribe rag ingest            # documents and PDFs (or Re-index in the site)
```

`config\party.toml` (titles, summaries, aliases, sheets) needs no command: reload the page.
Comics go in `assets\comics\<title>\` as `01.jpeg`, `02.jpeg`, ..., with an optional `comic.toml` (see `assets\comics\Two Minutes\`).
Portraits go in each character's folder as `portrait.png` / `.jpg` (not in `data\documents`). Every character has one now; a file named `portrait` is the one shown.

### Checks

```powershell
scribe rag status                                        # what is indexed
scribe rag read "data\documents\characters\Kael.pdf"     # the text it gets from a file
scribe rag search "celestial shards"                     # what search finds, no LLM call
scribe rag router-eval                                   # router accuracy
```

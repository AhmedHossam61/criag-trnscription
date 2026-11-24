# Option B: the lore site on free hosting

Status: **plan, not built yet**. Option A (your PC + Tailscale Funnel, GUIDE D.11) is what runs today.

**Goal:** the VAKT site is online when your PC is off. Cost: **$0**, with no paid tier, no domain, and ideally no card.
Transcription and correction stay on your PC. They need the GPU, and they're the part only you run. After a session, one command uploads the result.

---

## 1. What changes and why

The site needs two things that free hosting can't give it:

| Today (Option A) | Problem on a free host | Option B |
|---|---|---|
| `bge-m3` embedder, 2.2 GB, on the GPU | Free hosts give about 512 MB of RAM and no GPU | **Gemini embedding API** (`gemini-embedding-001`, 768 dimensions). If its free quota turns out to be 0, a small **ONNX** model in the server instead (Phase 0 decides) |
| `bge-reranker-v2-m3`, 2.2 GB | Same | **Off.** Keyword and meaning search, merged with RRF, decide the order alone; `final_k` goes up a little to make up for it |
| Answers: Gemini Flash-Lite, then Ollama on the GPU | No Ollama in the cloud | Gemini Flash-Lite, then a **second free API** (Groq or OpenRouter), not Ollama |
| Index, transcripts, PDFs and portraits on your disk | Free hosts have no lasting disk; it is wiped on every restart | A **"bundle"** in a private GitHub repo. The server downloads it every time it starts |
| Upload, delete and re-index in the site | Changes would vanish on the next restart | **Read-only site.** Everything changes on your PC, then `scribe publish` |
| Logins (`config/users.toml`) | — | Same logins, shipped in the bundle (hashes only). The cookie key is a host secret, so sign-ins survive restarts |

```
 YOUR PC                                                 FREE CLOUD
 scribe craig / correct / import                         ┌───────────────────────────────┐
          │                                              │ Render free web service        │
 scribe publish ─── 1. ingest (Gemini embeddings) ──┐    │  boot: scribe fetch-bundle     │
                    2. push bundle ─────────────────┼──▶ │        (private GitHub repo)   │
                    3. call the deploy hook ────────┘    │  scribe serve --config         │
                                                         │        config/rag.hosted.toml  │
                                                         │   ├ BM25 + vectors in memory   │
                                                         │   ├ Gemini embedding (query)   │
                                                         │   └ Flash-Lite → Groq          │
                                                         └───────────────────────────────┘
```

Your PC site (Option A) can keep its best-quality setup (bge-m3 and the reranker) next to this. The hosted site gets its own index in `data/index-hosted/`, so the two never mix.

---

## 2. The free services, and what each costs you

| Need | Pick | Free limits (check them, they change) | Card? |
|---|---|---|---|
| Host | **Render**, free web service | 512 MB RAM, 750 instance-hours a month (one service can stay on all month). Sleeps after 15 minutes idle; the first visit after that waits about a minute | Reported not needed. Check at sign-up |
| Bundle storage | **Private GitHub repo** (`vakt-lore-data`) | Files under 100 MB each; our bundle is about 30–60 MB in total | No |
| Answers | **Gemini Flash-Lite** | Reported about 500 calls a day **per Google Cloud project** | No |
| Fallback answers | **Groq** free plan (or OpenRouter's `:free` models) | Reported 1,000–14,400 requests a day depending on the model; check console.groq.com | No |
| Embeddings | **Gemini** `gemini-embedding-001` | Reported 1,000 requests a day and 100 a minute, with up to 100 passages per request. **Some accounts report 0; Phase 0 tests this** | No |
| Keep-awake (optional) | cron-job.org pings `/healthz` every 10 minutes | Free | No |

Fallback hosts, if Render's free plan changes:
1. **Hugging Face Space (CPU Basic, 16 GB).** It could even run bge-m3 and the reranker. The docs say it's free, but forum posts from 2026 say new accounts get no quota without PRO. Worth a 5-minute try.
2. **Oracle Cloud Always Free ARM VM (4 cores, 24 GB).** It runs everything, including bge-m3 and the reranker, at $0 forever. Sign-up asks for a card to check identity (nothing is charged), and free machines aren't always in stock.

**Quota math for the party:** 4 players asking 10 questions each a day is 40 questions.
- A one-step question costs about 1 embedding call and 1 answer call.
- A multi-step question costs about 1–2 embedding calls (sub-questions go in one batch) and 2–3 answer calls.
- That comes to about **80–120 Flash-Lite calls and 60 embedding calls a day**: well inside the reported limits.
- Indexing a new session costs about 10–30 embedding requests. The first full index of everything costs about 80, once.

**Privacy:** on Gemini's free tier, Google may use what you send to improve its products. That includes the transcripts' passages and your friends' questions. Groq and OpenRouter's free models have their own terms. Option A has the same issue with Gemini; only Ollama keeps everything at home.

---

## 3. Phases

Each phase ends with something you can check. Phases 1–3 also improve Option A, and nothing in them changes how your PC site works unless you choose to.

### Phase 0 — Check the free tiers (about 1 hour, no code)
1. Go to https://aistudio.google.com/rate-limit and note the daily limits for your project for the Flash-Lite model and for `gemini-embedding-001`.
2. Make one test embedding call. I'll add `scribe rag embed-check`, which embeds three sentences and prints the vector size or the exact error.
3. Create a Render account and confirm it doesn't ask for a card for a free web service.
4. Create a Groq account and an API key, and note its limits page.

**Decision:**
- If the embedding quota is not 0, use the **Gemini embedder**.
- If it is 0, use the **ONNX** route: `intfloat/multilingual-e5-small`, int8, about 120 MB, run by `onnxruntime` with no torch. It is weaker than bge-m3 but still matches Arabic and English questions to each other. Measure its RAM on Render, which needs to stay under 400 MB.

### Phase 1 — Choosing the embedder in `rag.toml`
- Add `[embedding] provider = "transformers" | "gemini" | "onnx"`.
  - `GeminiEmbedder` uses the same key pool and daily counting as answers, under its own model slot.
  - It sends up to 100 texts per request and marks passages and questions differently for the API (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`).
  - It asks for 768 dimensions, normalizes the vectors, and backs off on per-minute 429 errors.
- Add the provider to the index's settings signature, so switching rebuilds the index cleanly.
- Make the `index_dir` setting configurable (`data/index-hosted`).
- Tests use a fake Gemini client; there are no real calls in the suite.
- **Check:** `scribe rag ingest --config config/rag.hosted.toml`, then `scribe rag search "..."`, gives sensible results.

### Phase 2 — The hosted configuration and a light install
- Add `config/rag.hosted.toml`:
  - Gemini embedder, reranker off.
  - `final_k = 8`, `candidates = 40`.
  - `index_dir = "data/index-hosted"`, `providers = ["gemini", "groq"]`.
  - `[server] read_only = true`.
- `read_only` hides upload, delete and re-index for everyone (admins too) and turns those endpoints off.
- Add a `host` install extra: `fastapi`, `uvicorn`, `python-multipart`, `google-genai`, `httpx`, `pypdf`, `numpy` (plus `onnxruntime` and `tokenizers` for the ONNX route). **No torch or transformers.**
  - A test imports the server with `transformers` blocked, to prove nothing pulls it in.
- The router uses the hosted embedder. Its small model (`router.npz`) is trained on your PC during publish and shipped, so the host never retrains it.
- Add a `/healthz` route that needs no login, for Render and the keep-awake ping.
- **Check:** on your PC, in a fresh venv with only `.[host]`, `scribe serve --config config/rag.hosted.toml` answers questions and stays under 400 MB of RAM.

### Phase 3 — A second free answer provider
- Add an `openai_compat` provider (one class covers Groq, OpenRouter and Cerebras). It takes a base URL, a model and a key from `.env` (`GROQ_API=...`).
- Streaming answers, JSON mode for the planner, and a 429 counts as "out of quota", so the next provider takes over.
- **Check:** with Gemini removed from `providers`, the hosted site still answers.

### Phase 4 — Bundle and publish
`scribe publish` on your PC:
1. Runs the hosted ingest (new and changed files only) and trains `router.npz`.
2. Copies into a local clone of the private data repo (`data/publish/`):
   - `data/index-hosted/`
   - each session's `transcript.json` and `recording.json`
   - `data/documents/characters/`
   - `assets/`
   - `config/party.toml`, `config/rag.hosted.toml`, `config/users.toml`
   - It leaves out recordings, `segments.jsonl`, `corrections.json` and `.env`.
3. Commits and pushes with your normal git login.
4. Calls Render's deploy hook (a secret URL in `.env`), so the site restarts with the new bundle about 2 minutes later.

`scribe fetch-bundle` on the host:
- Downloads the repo as a tarball through GitHub's API with a **read-only, single-repo** fine-grained token (`DATA_REPO_TOKEN`).
- Unpacks it into the app folder, then starts the server. No git is needed on the host.

**Check:** run publish, then fetch into an empty folder on your PC, and serve from it. The site matches.

### Phase 5 — Deploy on Render
- Add a `render.yaml` (or a Dockerfile) in the code repo:
  - Python 3.12, `pip install .[host]`.
  - Start command: `scribe fetch-bundle && scribe serve --config config/rag.hosted.toml --host 0.0.0.0 --port $PORT --no-warmup`.
  - Health check path `/healthz`.
- Secrets go in Render's dashboard, never in a repo: `GEMINI_API`, `GROQ_API`, `SCRIBE_SESSION_SECRET` (a long random string), `DATA_REPO`, `DATA_REPO_TOKEN`.
- Render's proxy adds `X-Forwarded-For`, so the site already refuses to run without logins. Cookies are marked secure behind its HTTPS (already handled).
- Optional: point cron-job.org at `https://<name>.onrender.com/healthz` every 10 minutes, so no one waits for a cold start.

**Check:** sign in from a phone on mobile data. Ask a question, open a character page, and check that a member can't see upload or delete.

### Phase 6 — Compare, then switch
- Ask the same 20 real questions on both sites (Arabic and English, one-step and multi-step, about characters and about sessions), using `scribe rag search` and the sources panel. Tune `final_k`, `candidates` and `dense_k` until the hosted answers are close.
- Send the party the `onrender.com` address. Turn Funnel off (`tailscale funnel reset`) or keep it as a backup.

### Day to day after the switch
```powershell
scribe craig "data\recordings\NEW.zip"
scribe correct -o data\transcripts\ID --audio "data\recordings\NEW.zip"
scribe publish            # a couple of minutes later the site has the new session
```

---

## 4. Risks and what happens if they hit

| Risk | Effect | Fallback |
|---|---|---|
| Gemini embedding free quota is 0 or gets cut | No meaning search on the host | ONNX e5-small route (Phase 0), or keyword-only search until it's fixed |
| Gemini answer quota cut further | Answers stop | Groq / OpenRouter take over automatically (Phase 3) |
| Render drops its free plan or asks for a card | Site down | Hugging Face Space or Oracle Always Free; the bundle and start command stay the same |
| Cold starts annoy people | First visit waits about a minute | Keep-awake ping (Phase 5) |
| The bundle repo leaks | Transcripts and password hashes exposed | Keep it private, use a read-only token, give everyone strong passwords, and `scribe users passwd` everyone if it ever happens |
| Smaller embedder or no reranker gives worse answers | Some questions miss | Phase 6 tuning; Oracle (24 GB) can run the full setup for free |

## 5. To do now, whatever happens

**Your code repo `AhmedHossam61/criag-trnscription` is public**, and it contains the session transcripts (`data/transcripts/` is committed). Anyone can read them now. If they're meant for the party only:
1. On GitHub: **Settings > General > Danger Zone > Change visibility > Private**.

Option B also needs the data in a private repo, so this has to happen before Phase 4.

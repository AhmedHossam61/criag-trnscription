# Craig Transcriber

Turns recordings of our Dungeons & Dragons sessions into readable transcripts, with
each line credited to the player who said it. Built for **Egyptian Arabic mixed with
English** ("انت شايف المصباح، but it doesn't seem that they have noticed you"), and
runs locally on an 8 GB GPU.

It handles two kinds of recording:

| Recording | Script | How speakers are known |
|---|---|---|
| **Craig** (Discord bot): a zip with one audio track per player | `craig_transcribe.py` | Each track is one Discord user, renamed with `speaker_map.txt` |
| **One mixed recording**: a phone on the table, a Zoom export, ... | `phone_transcribe.py` | Detected from the audio, then named by matching voices to profiles learned from Craig sessions |

**→ Step-by-step instructions are in [GUIDE.md](GUIDE.md).**

---

## Quick start

```powershell
.\.venv-qwencleo\Scripts\Activate.ps1

# A Craig recording
python craig_transcribe.py "session.zip" -o out-session --speaker-map speaker_map.txt

# A phone recording (voice profiles in voices.npz are already built)
python phone_transcribe.py "session.m4a" -o out-phone-session --num-speakers 5
```

Open `transcript.txt` in the output folder:

```
[00:00:27.827] Dungeon Master:
  [mixed] ما انا متحركين في اتجاه المعاكس فانت you can't really tell ان it's not very bright
  [mixed] even with your dark vision يعني انت شايف المصباح ...

[00:01:01.340] Kael:
  اه استنىهم يبعدوا شويه وبعدين هبرسيد ...
```

`[en]` / `[mixed]` mark lines with English, so code-switches are easy to find. Each
run also writes `.srt` subtitles (play them next to the audio in VLC) and a `.json` file.

---

## How it works

### Craig recordings
1. **Split by player for free.** Craig gives one track per player, and Discord writes
   exact digital silence when someone isn't talking. So "who spoke when" is known
   exactly, and only real speech is sent to the GPU.
2. **Cut into sentences.** A player's speech is split at pauses of 1.5 s or longer,
   capped at 28 s per chunk.
3. **Transcribe in batches.** [QwenCleo-ASR](https://huggingface.co/mohammedaly22/QwenCleo-ASR)
   is a Qwen3-ASR 1.7B model fine-tuned on Egyptian Arabic with English code-switching.
   It keeps English words in Latin script. Chunks of similar length are batched
   together on the GPU (about 4× realtime on an RTX 3050).
4. **Filter hallucinations.** Near-silent chunks (mic hiss) are skipped. Known invented
   phrases, repetition loops and text too long for the audio are dropped into
   `dropped.txt`.

### Phone recordings
1. **Who spoke when.** [pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)
   is used when a Hugging Face token is set. Otherwise a built-in detector runs: Silero
   voice detection, then voice embeddings, then clustering.
2. **Who is who.** Each detected speaker's voice is compared with profiles
   (`voices.npz`) learned from the players' clean Craig tracks. Groups that match the
   same player are merged, which handles the DM's NPC voices.
3. **Transcribe.** Same model, batching, filters and resume as above.

In a test on a simulated phone mix of a real session, the built-in detector credited
**96%** of speaking time to the right player.

---

## Features
- **Resume anywhere.** Stop with `Ctrl+C`, then run the same command again.
- **Safe output folders.** A script refuses to mix two recordings, or two
  incompatible settings, in one output folder.
- **Rename without re-transcribing.** `--rewrite-outputs` re-applies names and
  filters in seconds.
- **Quick trial runs.** `--limit 300` does only the first 5 minutes; `--dry-run`
  analyses without transcribing.
- **Checking phone speakers by ear.** `speaker_samples\` gets a short clip per speaker.
- **Other models.** `--backend` also offers `qwen3asr`, `cohere` and `faster-whisper`,
  each in its own environment (`setup.cmd <backend>`).

---

## Project files

| File | Purpose |
|---|---|
| `craig_transcribe.py` | Transcribe a Craig zip |
| `phone_transcribe.py` | Transcribe one mixed recording; `--enroll` builds voice profiles |
| `llm_correct.py` | Optional pass: Gemini listens to the audio and fixes misheard names, terms and slang |
| `glossary.txt` | Campaign names and terms for the Gemini pass |
| `speaker_map.txt` | Discord username → character name (`SKIP` for bots) |
| `voices.npz` | Voice profiles of the players (not in git: personal data) |
| `GUIDE.md` | How to run everything, options, troubleshooting |
| `setup.cmd` / `setup.sh` | Create the environment for a backend |
| `requirements-*.txt` | Dependencies per backend |
| `plan.md`, `info.md` | Design notes and session log |

---

## Requirements
- Windows or Linux, an NVIDIA GPU (8 GB is enough), Python 3.12
- `ffmpeg` and `uv` on the PATH
- Optional: a free Hugging Face token for pyannote (see [GUIDE.md](GUIDE.md#b4-better-speaker-detection-with-pyannote-optional-recommended))

## Limitations
- **Fantasy names get misheard.** Invented names and D&D terms are often wrong
  ("lights" for "knights").
- **Some invented lines still get through.** On very quiet audio the model can still
  produce a short sentence that wasn't said; `dropped.txt` shows what the filters caught.
- **Phone recordings are harder than Craig.** Overlapping speech goes to the loudest
  person, far or quiet players get less accurate text, and a speaker may be
  mislabelled at fast back-and-forth.

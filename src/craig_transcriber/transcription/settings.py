"""Tunables shared by the Craig and phone pipelines."""


SR = 16000  # every ASR model here wants 16 kHz mono


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A gap must be at least this long to count as a segment boundary. Shorter
# pauses are kept inside one chunk. 0.45 s only bridged packet jitter, which
# cut sentences at every breath ("في حد من ال" / "النائتس بيلمحك") and fed the
# model 1-2 s fragments it could only guess at. 1.5 s keeps a speaker's
# sentence together; MAX_CHUNK_S still bounds the window.
# Changing this changes chunk ids, so an output folder records the value
# and refuses to resume under a different one.
MIN_GAP_S = 1.5

# Never emit a chunk longer than this. Whisper's encoder is hard-capped at 30s;
# Qwen and Cohere handle more but accuracy and VRAM both degrade.
MAX_CHUNK_S = 28.0

# Ignore speech islands shorter than this -- almost always a cough, a mic bump,
# or a single "aha" that the model will hallucinate a sentence out of.
MIN_SPEECH_S = 0.35

# Padding kept either side of a speech island so plosives aren't clipped.
PAD_S = 0.20

# Generation budget per chunk: TOKEN_FLOOR + TOKENS_PER_S * longest chunk in the
# batch, capped at TOKEN_CEIL. Hallucination loops otherwise run to the cap on
# every bad chunk. 15 tok/s sits above what screen()'s text-length ceiling lets
# through, so anything this truncates would have been dropped anyway.
# Skip chunks whose loudest sample is below this. Discord transmits mic hiss
# and breathing as packets, so they survive the zero-run VAD, and QwenCleo
# turns them into stock sentences ("...project في الـ web development").
# Measured on a real session: hallucinated chunks peak around -45 dBFS, real
# speech from the quietest mic around -29. -40 catches most of the former
# without touching the latter.
MIN_PEAK_DBFS = -40.0

# Skip short chunks in which the speech detector (Silero VAD) hears someone
# talking for less than this share of the clip. Checked against the audio of
# one 5-minute stretch: real short lines ("RITUAL!", "ايه ده", "يا جدع") were
# 36-69% speech; lines invented on laughs and breaths were 0-32%, and 0.25
# removed 4 of 6 of them without losing a real one. Only chunks up to
# VAD_MAX_CHUNK_S are checked -- every invented line was shorter -- which keeps
# the check to a minute or two per session on the CPU.
MIN_SPEECH_RATIO = 0.25
VAD_MAX_CHUNK_S = 6.0

TOKENS_PER_S = 15
TOKEN_FLOOR = 24
TOKEN_CEIL = 256

# Right-hand side of a speaker map entry that hides a speaker.
SKIP_NAMES = {"SKIP", "-", "IGNORE"}

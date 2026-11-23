"""Stage 3: hallucination filtering, per line and across the session."""

from __future__ import annotations

import re

from .language import script_mix
from .models import Chunk, Segment


# Phrases these models emit when handed near-silence or noise. The zero-run
# gate removes most of the opportunity, but low-level mic hiss still slips
# through and both Whisper and Cohere are documented as hallucinating on it.
HALLUCINATION_PATTERNS = [
    r"ترجمة\s+نانسي",
    r"اشترك\s+في\s+القناة",
    r"لا\s+تنسى\s+الاشتراك",
    r"شكرا\s+(لكم\s+)?على\s+المشاهدة",
    r"subtitles?\s+by",
    r"amara\.org",
    r"thanks?\s+for\s+watching",
    # "I love you, you love me" (Arabic Barney dub). Comes out of near-silence.
    # Match both orderings but only when BOTH halves appear together, so
    # a genuine standalone "بحبك" from a player isn't flagged.
    r"بحبك[^\n]{0,30}بحبني",
    r"بحبني[^\n]{0,30}بحبك",
    # QwenCleo's stock sentences on noise -- podcast training-data leftovers.
    r"بنعمل\s+project",
    r"web\s+(development|design)",
    r"engineering\s+college",
    r"cairo\s+university",
    # Stock sentences from short clips of laughter/breath. Each came from
    # every player in one session (39x for the first), so none is real speech.
    r"مش\s+متأكد[ةه]?\s+[إا]ن\s+[أا]نا\s+بحبك",
    r"^بصراح[ةه]\s+[أا]نا\s+مش\s+عارف\s+([أا]عمل\s+[إا]يه|[إا]يه\s+اللي\s+بيحصل\s+ده)\W*$",
    r"^(طب\s+)?[أا]نا\s+بحب\s+[أا]عمل\s+(ازاي|[إا]يه\s+في\s+ال)",
    r"بحبك\s+يا\s+حبيبي",
    # Chinese/Japanese/Korean script: nobody in the session speaks it.
    r"[぀-ヿ㐀-鿿가-힯]",
    r"^\W*$",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]

# Arabic diacritics and tatweel -- strip before comparing repeated tokens.
_ARABIC_NOISE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")


def normalize_arabic(text: str) -> str:
    text = _ARABIC_NOISE.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip()


def repetition_ratio(text: str) -> float:
    """Fraction of tokens that are duplicates. Loops score near 1.0."""
    tokens = normalize_arabic(text).split()
    if len(tokens) < 6:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def loop_tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", normalize_arabic(text)).split()


def has_tandem_repeat(tokens: list[str], max_n: int = 6) -> bool:
    """
    True if a phrase of 1..max_n words is said back to back many times
    ("X Y X Y X Y X Y"): what a decoder stuck in a loop produces. Single
    words need 6 in a row, since "لا لا لا" and "ماشي ماشي ماشي" are real.
    """
    for n in range(1, max_n + 1):
        need = 6 if n == 1 else 4
        for i in range(len(tokens) - n * need + 1):
            gram = tokens[i:i + n]
            reps, j = 1, i + n
            while tokens[j:j + n] == gram:
                reps += 1
                j += n
            if reps >= need:
                return True
    return False


def has_dominant_ngram(tokens: list[str], min_cover: float = 0.7) -> bool:
    """
    True if one 2-4 word phrase, said 3+ times, makes up most of the line.
    Occurrences are counted without overlap, so "لا لا لا لا" is two "لا لا",
    not three. 0.7 keeps a real line the model stuttered on ("لا انت انت ما
    كنتش انت ما كنتش هتقول انت ما كنتش هتقول بتحورها", 0.64).
    """
    for n in (2, 3, 4):
        if len(tokens) < 3 * n:
            continue
        counts: dict[tuple[str, ...], int] = {}
        next_free: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if i < next_free.get(gram, 0):
                continue
            counts[gram] = counts.get(gram, 0) + 1
            next_free[gram] = i + n
        top = max(counts.values())
        if top >= 3 and top * n / len(tokens) >= min_cover:
            return True
    return False


def has_ngram_loop(text: str) -> bool:
    """
    Short-phrase loops that repetition_ratio misses.

    Repetition alone is not enough: Egyptian conversation repeats itself
    ("ريت شوال ريت شوال يعني ايه ... مش ريت شوال", "هل انت هل انت نمت").
    Counting any bigram said 3 times as a loop threw away 14 s of real
    dialogue in a 5-minute test, so only back-to-back runs or a phrase that
    dominates the line count.
    """
    tokens = loop_tokens(text)
    return has_tandem_repeat(tokens) or has_dominant_ngram(tokens)


def screen(text: str, chunk: Chunk) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    stripped = text.strip()
    if not stripped:
        return False, "empty"

    for pattern in _HALLUCINATION_RE:
        if pattern.search(stripped):
            return False, "boilerplate-hallucination"

    if repetition_ratio(stripped) > 0.82:
        return False, "token-loop"

    if has_ngram_loop(stripped):
        return False, "ngram-loop"

    # A model that produces far more text than the audio could carry is
    # confabulating. English runs denser per second than Arabic, so use the
    # looser of the two limits when the output contains Latin script.
    latin, _ = script_mix(stripped)
    ceiling = 30.0 if latin > 0.3 else 25.0
    if len(stripped) > ceiling * max(chunk.duration_s, 1.0) * 1.6:
        return False, "text-exceeds-audio-duration"

    return True, ""


# A sentence at least STOCK_MIN_WORDS long that comes out word for word from
# STOCK_MIN_SPEAKERS different people, STOCK_MIN_COUNT+ times, is the model's
# stock filler, not speech. Catches new ones the patterns above don't know.
STOCK_MIN_WORDS = 5
STOCK_MIN_SPEAKERS = 3
STOCK_MIN_COUNT = 4


def drop_stock_phrases(segments: list[Segment]) -> int:
    """Mark session-wide repeated sentences as dropped; return how many."""
    groups: dict[str, list[Segment]] = {}
    for seg in segments:
        if seg.dropped:
            continue
        norm = re.sub(r"[^\w\s]", "", normalize_arabic(seg.text)).strip()
        if len(norm.split()) >= STOCK_MIN_WORDS:
            groups.setdefault(norm, []).append(seg)
    dropped = 0
    for segs in groups.values():
        if len(segs) >= STOCK_MIN_COUNT and len({s.speaker for s in segs}) >= STOCK_MIN_SPEAKERS:
            for seg in segs:
                seg.dropped, seg.drop_reason = True, "stock-phrase"
                dropped += 1
    return dropped


def refresh_drops(segments: list[Segment]) -> None:
    """
    Re-apply screen() and the stock-phrase rule, so filter changes made since
    the transcript was recorded take effect without re-running the model.
    Chunks are reconstructed from each segment's own timing.

    Text dropped by an older, stricter filter comes back too. Segments the
    level gate dropped ("too-quiet") were judged on audio, not text, so they
    stay dropped.
    """
    rescreened = restored = 0
    for seg in segments:
        if seg.dropped and seg.drop_reason in ("too-quiet", "no-speech"):
            continue
        chunk = Chunk(
            track_index=seg.track_index, speaker=seg.speaker,
            chunk_id=seg.chunk_id, start_s=seg.start_s, end_s=seg.end_s,
        )
        keep, reason = screen(seg.text, chunk)
        if not keep and not seg.dropped:
            rescreened += 1
        elif keep and seg.dropped:
            restored += 1
        seg.dropped, seg.drop_reason = not keep, reason
    if rescreened:
        print(f"  re-screen dropped {rescreened} previously-kept segment(s)")
    if restored:
        print(f"  re-screen restored {restored} segment(s) an older filter had dropped")
    stock = drop_stock_phrases(segments)
    if stock:
        print(f"  dropped {stock} segment(s) repeating a stock phrase across speakers")

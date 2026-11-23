"""The transcription pieces that need no audio or GPU: filters, chunk keys, speaker maps."""

from craig_transcriber.transcription.filters import drop_stock_phrases, has_ngram_loop, screen
from craig_transcriber.transcription.language import label_script
from craig_transcriber.transcription.models import Chunk, Segment
from craig_transcriber.transcription.outputs import fmt_timestamp
from craig_transcriber.transcription.pipeline import make_batches
from craig_transcriber.transcription.speakers import is_skipped, load_speaker_map, map_speaker


def chunk(seconds=5.0):
    return Chunk(track_index=1, speaker="Kael", chunk_id=3, start_s=0.0, end_s=seconds)


def test_real_repetition_is_kept_and_loops_dropped():
    # Real Egyptian conversation repeats itself; these lines were once wrongly dropped.
    assert not has_ngram_loop("لا لا لا لا مش كده")
    assert not has_ngram_loop("ريت شوال ريت شوال يعني ايه ... مش ريت شوال")
    assert has_ngram_loop("يا جماعة يا جماعة يا جماعة يا جماعة يا جماعة")
    assert has_ngram_loop(" ".join(["تمام"] * 7))


def test_screen_reasons():
    assert screen("", chunk()) == (False, "empty")
    assert screen("ترجمة نانسي قنقر", chunk())[1] == "boilerplate-hallucination"
    assert screen("انت شايف المصباح، but it doesn't seem that they have noticed you", chunk()) == (True, "")
    assert screen("كلام " * 200, chunk(1.0))[0] is False


def test_stock_phrases_need_many_speakers():
    text = "انا مش عارف ايه اللي بيحصل هنا خالص"
    segs = [Segment(speaker=s, start_s=i, end_s=i + 1, text=text) for i, s in enumerate("ABCD")]
    assert drop_stock_phrases(segs) == 4
    one_speaker = [Segment(speaker="A", start_s=i, end_s=i + 1, text=text) for i in range(5)]
    assert drop_stock_phrases(one_speaker) == 0


def test_script_labels():
    assert label_script("spell slots") == "en"
    assert label_script("يلا بينا") == "ar"
    assert label_script("انا عايز long rest دلوقتي") == "mixed"


def test_batches_are_bounded():
    chunks = [Chunk(1, "k", i, 0.0, float(d)) for i, d in enumerate([28, 27, 3, 2, 2, 1, 1, 1])]
    batches = make_batches(chunks, max_count=4, max_seconds=60)
    assert all(len(b) <= 4 and len(b) * b[0].duration_s <= 60 for b in batches)
    assert sorted(c.chunk_id for b in batches for c in b) == list(range(8))


def test_speaker_map_globs_and_skip(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text("ahmedhoss61 = Kael  # comment\njockie* = SKIP\n", encoding="utf-8")
    mapping = load_speaker_map(path)
    assert map_speaker("AhmedHoss61", mapping) == "Kael"
    assert is_skipped(map_speaker("jockie_music8158", mapping))
    assert map_speaker("stranger", mapping) is None and not is_skipped(None)


def test_timestamps():
    assert fmt_timestamp(3725.5) == "01:02:05,500"
    assert fmt_timestamp(0.25, comma=False) == "00:00:00.250"

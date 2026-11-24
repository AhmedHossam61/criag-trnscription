import json

import pytest

from craig_transcriber.cli.importer import main, parse_transcript_txt, to_segments

OLD = """\
# Recording: AqliOggnsXbo
# Start time: 2026-03-14T19:02:11.000Z

[00:00:06.299] Dungeon Master:
  [mixed] الcult مصدقين إن هما لو جمعوا الshards
  ولما يعملوا كده هياخدهم تحت جناحه

[00:00:20.000] Vaylor:
  [en] I check the altar

[00:00:21.500] jockie_music:
  now playing

[00:01:00] Kael: يلا نعمل long rest
"""


def test_parse_header_turns_and_tags():
    info, turns = parse_transcript_txt(OLD)
    assert info == {"Recording": "AqliOggnsXbo", "Start time": "2026-03-14T19:02:11.000Z"}
    assert [t["speaker"] for t in turns] == ["Dungeon Master", "Vaylor", "jockie_music", "Kael"]
    assert turns[0]["start_s"] == pytest.approx(6.299) and turns[3]["start_s"] == 60
    assert turns[0]["lines"][0] == "الcult مصدقين إن هما لو جمعوا الshards"     # [mixed] tag gone
    assert turns[3]["lines"] == ["يلا نعمل long rest"]                         # one-line form


@pytest.mark.parametrize("text", [
    # ### speaker `time`, text below
    "# DnD Full Session Transcript\n\n**Players:** abdoo_, seifeldien\n\n---\n\n"
    "### seifeldien `0:00:03`\nكيف حالك ؟\n\n### abdoo_ `1:02:04`\nطيب\nماشي\n",
    # `time` **speaker:** text
    "# DnD Session Transcript\n\n`00:03` **seifeldien:** كيف حالك ؟\n\n`1:02:04` **abdoo_:** طيب\nماشي\n",
    # [start - end] **speaker:** text  (Craig id in the title)
    "# Craig Transcript nzvn2MK67jmS\n\n[00:00:03.000 - 00:00:05.500] **seifeldien:** كيف حالك ؟\n\n"
    "[01:02:04.000 - 01:02:09.000] **abdoo_:** طيب\nماشي\n",
])
def test_other_transcript_layouts(text):
    info, turns = parse_transcript_txt(text)
    assert [(t["speaker"], t["start_s"], t["lines"]) for t in turns] == [
        ("seifeldien", 3, ["كيف حالك ؟"]), ("abdoo_", 3724, ["طيب", "ماشي"])]
    segs = to_segments(turns, {"seifeldien": "Vaelor", "abdoo_": "Arnok"})
    assert [s["speaker"] for s in segs] == ["Vaelor", "Arnok", "Arnok"]
    if "Craig" in text:
        assert info["Recording"] == "nzvn2MK67jmS" and segs[0]["end_s"] == 5.5 and segs[2]["end_s"] == 3729


def test_segments_map_speakers_and_stay_inside_their_turn():
    _, turns = parse_transcript_txt(OLD)
    segs = to_segments(turns, {"vaylor": "Vaelor", "jockie*": "SKIP"})
    assert [s["speaker"] for s in segs] == ["Dungeon Master", "Dungeon Master", "Vaelor", "Kael"]
    assert segs[1]["start_s"] == segs[0]["end_s"] and segs[1]["end_s"] <= 20.0


def test_import_writes_a_session_and_protects_real_ones(tmp_path, capsys):
    src = tmp_path / "old.txt"
    src.write_text(OLD, encoding="utf-8")
    out = tmp_path / "transcripts"
    speaker_map = tmp_path / "speaker_map.txt"
    speaker_map.write_text("vaylor = Vaelor\n", encoding="utf-8")
    main([str(src), "--output-root", str(out), "--speaker-map", str(speaker_map)])
    session = out / "AqliOggnsXbo"
    data = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 5 and (session / "original-old.txt").exists() and src.exists()
    assert json.loads((session / "recording.json").read_text(encoding="utf-8"))["Imported from"] == "old.txt"

    with pytest.raises(SystemExit):                       # exists: needs --force
        main([str(src), "--output-root", str(out)])
    main([str(session / "original-old.txt"), "--output-root", str(out), "--force"])   # its own original
    assert sorted(p.name for p in session.iterdir()) == ["original-old.txt", "recording.json", "transcript.json"]
    (session / "segments.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):                       # a real transcription is never replaced
        main([str(src), "--output-root", str(out), "--force"])
    assert "full transcription" in capsys.readouterr().out

    notes = tmp_path / "notes.txt"
    notes.write_text("We met the cult at the altar.", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(notes), "--output-root", str(out)])
    assert "data/documents/sessions" in capsys.readouterr().out

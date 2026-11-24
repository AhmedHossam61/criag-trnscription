import pytest
from fastapi.testclient import TestClient

from craig_transcriber.rag.engine import RagEngine
from craig_transcriber.rag.party import Party, load_roster
from craig_transcriber.server.app import create_app

from conftest import FakeEmbedder, FakeLLM, FakeReranker

ROSTER = """
[[character]]
name = "Arnok"
role = "main"
player = "abdoo_"
aliases = ["أرنوك"]

[[character]]
name = "Kael"
full_name = "Kael Veyron"
role = "main"
sheet = "data/documents/rules.txt"

[[character]]
name = "Ronaldo Raddis the Third"
role = "companion"
aliases = ["Raddis", "رادش"]
"""


@pytest.fixture
def party(archive):
    (archive.root / "config").mkdir()
    (archive.root / "config" / "party.toml").write_text(ROSTER, encoding="utf-8")
    transcript = archive.root / "data" / "transcripts" / "ABC123" / "transcript.json"
    text = transcript.read_text(encoding="utf-8").replace(
        '"I make a religion check on the altar"', '"I make a religion check, ورادش بيبص"').replace(
        '"يلا نعمل long rest"', '"يلا نعمل long rest يا ارنوك"')
    transcript.write_text(text, encoding="utf-8")
    return archive


def test_roster_ids_and_errors(tmp_path):
    path = tmp_path / "party.toml"
    path.write_text(ROSTER, encoding="utf-8")
    assert [c.id for c in load_roster(path)] == ["arnok", "kael", "ronaldo-raddis-the-third"]
    path.write_text('[[character]]\nname = "X"\nrole = "villain"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        load_roster(path)
    path.write_text('[[character]]\nname = "X"\nclass = "wizard"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="class"):
        load_roster(path)


def test_appearances_count_speech_and_mentions(party):
    p = Party(party)
    [arnok] = p.appearances(p.get("arnok"))
    assert arnok["spoken_lines"] == 1                       # his own line, not a mention
    assert [m["speaker"] for m in arnok["mentions"]] == ["Kael"]   # ارنوك matches أرنوك
    [raddis] = p.appearances(p.get("ronaldo-raddis-the-third"))
    assert raddis["spoken_lines"] == 0 and raddis["mentions"][0]["clock"] == "00:00:15"
    assert p.appearances(p.get("kael"))[0]["spoken_lines"] == 1


def test_arabic_nicknames_need_whole_words(tmp_path):
    from craig_transcriber.rag.party import _matcher
    m = _matcher(["نوب", "Noob"])
    for hit in ("يا نوب انت معانا", "انت ونوب", "قلت النوب", "Noob, roll it", "نوب؟"):
        assert m.search(hit), hit
    for miss in ("أول نوبة من المراقبة", "ناحية الجنوب", "ماينوب", "ينوبني", "noobish"):
        assert not m.search(miss), miss


def test_sheet_found_by_name_in_characters_folder(party):
    folder = party.root / "data" / "documents" / "characters"
    folder.mkdir()
    (folder / "Raddis.md").write_text("# Raddis\n\nA radish who dreams of knighthood.", encoding="utf-8")
    p = Party(party)
    sheet = p.sheet(p.get("ronaldo-raddis-the-third"))
    assert sheet["path"] == "data/documents/characters/Raddis.md" and "knighthood" in sheet["sections"][0]["text"]
    assert p.sheet(p.get("arnok")) is None                      # no file named after him
    assert p.sheet(p.get("kael"))["path"] == "data/documents/rules.txt"   # an explicit sheet wins


def test_query_expansion_and_llm_roster(party):
    p = Party(party)
    assert "رادش" in p.expand_query("What does Raddis want?")
    assert p.expand_query("What is a long rest?") == "What is a long rest?"
    roster = p.describe()
    assert "Player characters: Arnok (played by abdoo_; also written أرنوك)" in roster
    assert "Companions: Ronaldo Raddis the Third" in roster


def test_party_endpoints(party):
    (party.root / "assets" / "Arnok").mkdir(parents=True)
    (party.root / "assets" / "Arnok" / "battle.jpg").write_bytes(b"other")
    (party.root / "assets" / "Arnok" / "Portrait.png").write_bytes(b"fake-png")    # "portrait" wins
    (party.root / "assets" / "Raddis").mkdir()                                        # a Latin alias works
    (party.root / "assets" / "Raddis" / "radish.webp").write_bytes(b"radish")
    engine = RagEngine(party, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    engine.reindex(progress=lambda _: None)
    with TestClient(create_app(engine, warm_up=False)) as c:
        cards = {ch["id"]: ch for ch in c.get("/api/party").json()["characters"]}
        assert cards["arnok"]["portrait"].startswith("/api/party/arnok/portrait?v=") and cards["kael"]["portrait"] is None
        assert cards["ronaldo-raddis-the-third"]["role"] == "companion"
        assert c.get("/api/party/arnok/portrait").content == b"fake-png"
        assert c.get("/api/party/ronaldo-raddis-the-third/portrait").content == b"radish"

        kael = c.get("/api/party/kael").json()
        assert kael["sheet"]["sections"] and "Ritual Adept" in kael["sheet"]["sections"][0]["text"]
        assert c.get("/api/party/nobody").status_code == 404

        # The roster is re-read when it changes; a broken one is reported, not a crash.
        (party.root / "config" / "party.toml").write_text('[[character]]\nname = "X"\nrole = "villain"\n', encoding="utf-8")
        import os, time
        os.utime(party.root / "config" / "party.toml", (time.time() + 5, time.time() + 5))
        assert c.get("/api/party").status_code == 422

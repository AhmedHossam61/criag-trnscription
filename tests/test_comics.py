from fastapi.testclient import TestClient

from craig_transcriber.rag.engine import RagEngine
from craig_transcriber.server.app import create_app
from craig_transcriber.server.comics import load_comics

from conftest import FakeEmbedder, FakeLLM, FakeReranker


def make_comic(root, folder, pages, toml=None):
    d = root / "assets" / "comics" / folder
    d.mkdir(parents=True)
    for name in pages:
        (d / name).write_bytes(name.encode())
    if toml is not None:
        (d / "comic.toml").write_text(toml, encoding="utf-8")
    return d


def test_pages_read_in_number_order_and_settings(tmp_path):
    make_comic(tmp_path, "Two Minutes", ["10.png", "2.png", "1.jpeg", "notes.txt"],
               'title = "Two Minutes"\nsummary = "It isn\'t even lit."\ncharacters = ["Arnok", "Kael"]\n')
    make_comic(tmp_path, "empty", [])
    [comic] = load_comics(tmp_path / "assets" / "comics")
    assert [p.name for p in comic.pages] == ["1.jpeg", "2.png", "10.png"]
    assert comic.id == "two-minutes" and comic.summary == "It isn't even lit."
    make_comic(tmp_path, "WhatsApp", ["Image at 3.10 PM (1).jpeg", "Image at 3.10 PM.jpeg"])
    whatsapp = load_comics(tmp_path / "assets" / "comics")[1]
    assert [p.name for p in whatsapp.pages] == ["Image at 3.10 PM.jpeg", "Image at 3.10 PM (1).jpeg"]


def test_comics_endpoints(archive):
    (archive.root / "config").mkdir()
    (archive.root / "config" / "party.toml").write_text(
        '[[character]]\nname = "Arnok"\nrole = "main"\naliases = ["Noob"]\n', encoding="utf-8")
    make_comic(archive.root, "B side", ["01.png", "02.png"], 'title = "Two Minutes"\ncharacters = ["noob", "Kael"]\n')
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    with TestClient(create_app(engine, warm_up=False)) as c:
        [comic] = c.get("/api/comics").json()["comics"]
        assert [p.split("?")[0] for p in comic["pages"]] == ["/api/comics/two-minutes/pages/1",
                                                              "/api/comics/two-minutes/pages/2"]
        # A nickname finds the character's page; someone not in the roster is still listed.
        assert comic["characters"] == [{"name": "Arnok", "id": "arnok"}, {"name": "Kael", "id": None}]
        assert c.get("/api/comics/two-minutes/pages/2").content == b"02.png"
        assert c.get("/api/comics/two-minutes/pages/3").status_code == 404

        (archive.root / "assets" / "comics" / "B side" / "comic.toml").write_text('author = "x"\n', encoding="utf-8")
        assert c.get("/api/comics").status_code == 422

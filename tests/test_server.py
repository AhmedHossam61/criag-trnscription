import io
import json

import pytest
from fastapi.testclient import TestClient

from craig_transcriber.rag.engine import RagEngine
from craig_transcriber.server.app import create_app, safe_name

from conftest import FakeEmbedder, FakeLLM, FakeReranker


@pytest.fixture
def client(archive):
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    engine.reindex(progress=lambda _: None)
    with TestClient(create_app(engine, warm_up=False)) as c:
        yield c


def sse(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_ui_and_status(client):
    assert "{{party}}" not in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["documents"] == 3 and status["passages"] > 0 and status["llm"] == "fake-llm"


def test_party_name_and_emblem(archive):
    archive.party.name = "VAKT"
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    with TestClient(create_app(engine, warm_up=False)) as c:
        page = c.get("/").text
        assert "<title>VAKT Lore</title>" in page and "<svg class=\"sigil\"" in page
        assert c.get("/emblem").status_code == 404

    (archive.root / "assets").mkdir()
    (archive.root / "assets" / "emblem.png").write_bytes(b"fake-png")
    with TestClient(create_app(engine, warm_up=False)) as c:
        page = c.get("/").text
        assert '<img class="sigil" src="/emblem?v=' in page and '<link rel="icon" href="/emblem?v=' in page
        assert "<svg class=\"sigil\"" not in page
        assert c.get("/emblem").content == b"fake-png"
        assert c.get("/emblem?w=96").content == b"fake-png"     # not an image Pillow can read: sent as is


def test_images_shrink_and_stay_cached(archive):
    from PIL import Image

    (archive.root / "assets").mkdir()
    Image.new("RGB", (1000, 1000), "teal").save(archive.root / "assets" / "emblem.png")
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    with TestClient(create_app(engine, warm_up=False)) as c:
        url = c.get("/api/status").json()["emblem"]
        small = c.get(url + "&w=90")                        # snaps to the next size up, 96
        assert small.headers["content-type"] == "image/webp" and "immutable" in small.headers["cache-control"]
        assert Image.open(io.BytesIO(small.content)).size == (96, 96)
        assert list((archive.root / "data" / ".cache" / "images").glob("*.webp"))
        again = c.get(url + "&w=96", headers={"if-none-match": small.headers["etag"]})
        assert again.status_code == 304 and not again.content
        assert c.get("/emblem?w=96&v=old").headers["cache-control"] == "no-cache"   # an out-of-date link


def test_chat_streams_events(client):
    r = client.post("/api/chat", json={"question": "What does Ritual Adept do?", "mode": "single"})
    events = sse(r)
    assert r.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in events][0] == "route" and events[-1]["type"] == "done"
    passage_id = next(e for e in events if e["type"] == "sources")["sources"][0]["id"]
    passage = client.get(f"/api/passages/{passage_id}").json()
    assert passage["id"] == passage_id and "text" in passage


def test_upload_index_and_delete(client, archive):
    r = client.post("/api/documents", files=[
        ("files", ("../../evil name!.md", b"# Shards\n\nThe fifth shard is in the crypt.", "text/markdown")),
        ("files", ("virus.exe", b"MZ", "application/octet-stream")),
    ])
    body = r.json()
    assert body["saved"] == ["evil name_.md"] and "virus.exe" in body["rejected"]
    assert (archive.root / "data/documents/evil name_.md").exists()
    docs = {d["path"] for d in client.get("/api/documents").json()["documents"]}
    assert "data/documents/evil name_.md" in docs

    r = client.delete("/api/documents", params={"path": "data/documents/evil name_.md"})
    assert r.status_code == 200 and not (archive.root / "data/documents/evil name_.md").exists()


def test_delete_refuses_files_outside_documents(client, archive):
    r = client.delete("/api/documents", params={"path": "data/transcripts/ABC123/transcript.json"})
    assert r.status_code == 403
    r = client.delete("/api/documents", params={"path": "data/documents/../transcripts/ABC123/transcript.json"})
    assert r.status_code == 403
    assert (archive.root / "data/transcripts/ABC123/transcript.json").exists()


def test_classify_and_bad_mode(client):
    assert client.post("/api/classify", json={"question": "Compare the cult and the guardians"}).json()["mode"] \
        in ("single", "multi")
    assert client.post("/api/chat", json={"question": "x", "mode": "both"}).status_code == 400


def test_safe_name():
    assert safe_name("../../a/b/notes.md") == "notes.md"
    assert safe_name("ملاحظات الجلسة.md") == "ملاحظات الجلسة.md"
    assert safe_name("") == "upload"

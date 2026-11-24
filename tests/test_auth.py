import pytest
from fastapi.testclient import TestClient

from craig_transcriber.cli import users as users_cli
from craig_transcriber.rag.engine import RagEngine
from craig_transcriber.server import auth
from craig_transcriber.server.app import create_app
from craig_transcriber.server.auth import User, check_password, hash_password, load_users, save_users

from conftest import FakeEmbedder, FakeLLM, FakeReranker


def make_users(archive, **people):
    path = archive.root / "config" / "users.toml"
    save_users(path, [User(name, role, hash_password(pw)) for name, (role, pw) in people.items()])
    return path


@pytest.fixture
def engine(archive):
    engine = RagEngine(archive, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    engine.reindex(progress=lambda _: None)
    return engine


def sign_in(c, name, password):
    return c.post("/login", data={"name": name, "password": password}, follow_redirects=False)


def test_password_hashes():
    stored = hash_password("celestial shards")
    assert "celestial" not in stored
    assert check_password("celestial shards", stored) and not check_password("celestial", stored)
    assert not check_password("x", "not-a-hash")


def test_open_without_users_but_not_through_a_tunnel(engine):
    with TestClient(create_app(engine, warm_up=False)) as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/status").json()["login"] is False
        assert c.post("/api/reindex").status_code == 200           # no logins: you may do anything locally
        r = c.get("/api/status", headers={"X-Forwarded-For": "203.0.113.9"})
        assert r.status_code == 403 and "scribe users add" in r.json()["detail"]


def test_members_read_and_ask_admins_change_the_library(engine, archive):
    make_users(archive, Ahmed=("admin", "keeper-of-lore"), Seif=("member", "modmen-rules"))
    with TestClient(create_app(engine, warm_up=False)) as c:
        assert c.get("/", follow_redirects=False).headers["location"] == "/login"
        assert c.get("/api/status").status_code == 401
        assert c.get("/static/app.css").status_code == 200          # the login page needs it
        assert "Sign in" in c.get("/login").text

        r = sign_in(c, "seif", "wrong-password")
        assert r.status_code == 401 and "don&#x27;t match" in r.text
        assert sign_in(c, "SEIF", "modmen-rules").headers["location"] == "/"   # names ignore case
        status = c.get("/api/status").json()
        assert status["user"] == {"name": "Seif", "admin": False}
        assert c.post("/api/chat", json={"question": "What does Ritual Adept do?"}).status_code == 200
        assert c.post("/api/reindex").status_code == 403
        assert c.delete("/api/documents", params={"path": "data/documents/rules.txt"}).status_code == 403
        assert (archive.root / "data" / "documents" / "rules.txt").exists()

        c.post("/logout")
        assert c.get("/api/status").status_code == 401
        sign_in(c, "Ahmed", "keeper-of-lore")
        assert c.get("/api/status").json()["user"]["admin"] is True
        assert c.post("/api/reindex").status_code == 200


def test_new_password_ends_sessions_and_wrong_guesses_lock(engine, archive, monkeypatch):
    path = make_users(archive, Seif=("member", "modmen-rules"))
    with TestClient(create_app(engine, warm_up=False)) as c:
        sign_in(c, "Seif", "modmen-rules")
        assert c.get("/api/status").status_code == 200
        save_users(path, [User("Seif", "member", hash_password("new-password"))])
        assert c.get("/api/status").status_code == 401

        monkeypatch.setattr(auth, "MAX_FAILURES", 3)
        for _ in range(3):
            assert sign_in(c, "Seif", "guess").status_code == 401
        assert sign_in(c, "Seif", "new-password").status_code == 429    # locked, even with the right one


def test_broken_users_file_keeps_the_site_shut(engine, archive):
    (archive.root / "config").mkdir()
    (archive.root / "config" / "users.toml").write_text('[[user]]\nname = "X"\nrole = "dm"\npassword = "x"\n',
                                                        encoding="utf-8")
    with TestClient(create_app(engine, warm_up=False)) as c:
        r = c.get("/api/status")
        assert r.status_code == 503 and "role" in r.json()["detail"]


def test_users_command(archive, tmp_path, monkeypatch, capsys):
    config = tmp_path / "rag.toml"
    config.write_text(f'[server]\nusers = "{(tmp_path / "users.toml").as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(users_cli, "ask_password", lambda name: "a-long-password")
    users_cli.main(["--config", str(config), "add", "Ahmed", "--admin"])
    users_cli.main(["--config", str(config), "add", "Seif"])
    with pytest.raises(SystemExit):
        users_cli.main(["--config", str(config), "add", "seif"])
    users_cli.main(["--config", str(config), "remove", "SEIF"])
    users = load_users(tmp_path / "users.toml")
    assert list(users) == ["ahmed"] and users["ahmed"].admin
    assert check_password("a-long-password", users["ahmed"].password)
    users_cli.main(["--config", str(config), "list"])
    assert "Ahmed                admin" in capsys.readouterr().out

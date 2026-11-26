import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.main import app
from backend.services.show_memory import show_memory

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    saves = []
    monkeypatch.setattr(main, "_save_checkpoint", lambda: saves.append(1))
    show_memory.reset()
    yield saves
    show_memory.reset()


def test_add_edit_delete_fact(_isolate):
    r = client.post("/api/show/facts", json={"text": "Luke is in West Virginia"})
    fact = r.json()["fact"]
    assert fact["pinned"] is True
    r = client.patch(f"/api/show/facts/{fact['id']}", json={"text": "Luke is in Charleston, WV"})
    assert r.json()["fact"]["text"] == "Luke is in Charleston, WV"
    assert [f["text"] for f in client.get("/api/show/facts").json()["facts"]] == ["Luke is in Charleston, WV"]
    assert client.delete(f"/api/show/facts/{fact['id']}").status_code == 200
    assert client.get("/api/show/facts").json()["facts"] == []
    assert len(_isolate) == 3


def test_edit_pins_extracted_fact():
    fact = show_memory.add_fact("Luke has a cold")
    r = client.patch(f"/api/show/facts/{fact['id']}", json={"text": "Luke has a bad cold"})
    assert r.json()["fact"]["pinned"] is True


def test_empty_or_duplicate_add_400s(_isolate):
    assert client.post("/api/show/facts", json={"text": "  "}).status_code == 400
    client.post("/api/show/facts", json={"text": "Devon is out sick"})
    assert client.post("/api/show/facts", json={"text": "devon is out sick!"}).status_code == 400
    assert len(_isolate) == 1


def test_edit_to_empty_or_duplicate_400s(_isolate):
    a = client.post("/api/show/facts", json={"text": "Luke is in West Virginia"}).json()["fact"]
    b = client.post("/api/show/facts", json={"text": "Devon is out sick"}).json()["fact"]
    assert client.patch(f"/api/show/facts/{b['id']}", json={"text": "  "}).status_code == 400
    assert client.patch(f"/api/show/facts/{b['id']}", json={"text": None}).status_code == 400
    assert client.patch(f"/api/show/facts/{b['id']}", json={"text": "luke is in west virginia!"}).status_code == 400
    assert client.patch(f"/api/show/facts/{a['id']}", json={"text": "Luke is in West Virginia."}).status_code == 200
    assert [f["text"] for f in show_memory.facts] == ["Luke is in West Virginia.", "Devon is out sick"]
    assert len(_isolate) == 3


def test_null_text_add_400s(_isolate):
    assert client.post("/api/show/facts", json={"text": None}).status_code == 400
    assert show_memory.facts == []


def test_missing_fact_404s(_isolate):
    assert client.patch("/api/show/facts/999", json={"text": "x"}).status_code == 404
    assert client.delete("/api/show/facts/999").status_code == 404
    assert _isolate == []

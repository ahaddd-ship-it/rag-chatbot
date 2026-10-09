from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_health_reports_missing_store(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "missing")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"

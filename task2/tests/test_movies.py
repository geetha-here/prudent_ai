from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_movies_empty_query():
    resp = client.get("/api/movies?q=")
    data = resp.json()
    assert data["results"] == []

def test_movies_upstream_error(monkeypatch):
    def fake_request(*a, **k):
        raise Exception("boom")
    monkeypatch.setattr("movies_client.search_movies", fake_request)

    resp = client.get("/api/movies?q=test")
    assert resp.status_code == 502

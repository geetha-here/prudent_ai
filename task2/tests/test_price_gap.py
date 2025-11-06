from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_price_gap_happy_path():
    resp = client.post("/api/price-gap-pair", json={"nums": [5, 5, 5], "k": 0})
    data = resp.json()
    assert data["i"] == 0 and data["j"] == 1

def test_price_gap_validation_error():
    resp = client.post("/api/price-gap-pair", json={"nums": "x", "k": -1})
    assert resp.status_code == 422  # validation error
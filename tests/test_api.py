from fastapi.testclient import TestClient

from quantagent.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoint():
    response = client.post("/research/run", json={"symbol": "DEMO", "periods": 180, "seed": 3})
    assert response.status_code == 200
    assert response.json()["integrity_checks"]["chronological_split"] is True


def test_invalid_request_rejected():
    response = client.post("/research/run", json={"periods": 10})
    assert response.status_code == 422

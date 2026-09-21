from pathlib import Path
import sys

from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(BASE_DIR / "support_assistance" / "src")
)

from api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Zepto Support Assistance API"
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "query": "My payment failed"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "intent" in data
    assert "confidence" in data
    assert "response" in data

    assert isinstance(data["intent"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["response"], str)


def test_predict_validation_error():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
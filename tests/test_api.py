"""Tests for the credit policy optimizer API."""

from fastapi.testclient import TestClient

from credit_policy_optimizer.api import app

client = TestClient(app)


def test_health_check() -> None:
    """Test the /health endpoint returns 200 and expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"

from fastapi.testclient import TestClient

from backend.main import app


def test_platform_health_requires_auth():
    response = TestClient(app).get("/api/platform/health")
    assert response.status_code in {401, 403}


def test_public_health_contract():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    data = response.json()
    for key in ["status", "database", "model_loaded", "model_version", "prediction_logging", "active_nurses", "queue_count", "render_environment"]:
        assert key in data

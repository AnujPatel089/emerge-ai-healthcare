from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint_has_mlops_fields():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "database" in data
    assert "model_loaded" in data
    assert "current_model_version" in data
    assert "prediction_logging" in data
    assert "render_environment_detected" in data

from fastapi.testclient import TestClient

from backend.main import app


def test_model_governance_requires_auth():
    response = TestClient(app).get("/api/mlops/governance")
    assert response.status_code in {401, 403}

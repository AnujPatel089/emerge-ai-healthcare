from fastapi.testclient import TestClient

from backend.main import app


def test_mlops_route_requires_authentication():
    response = TestClient(app).get("/api/mlops/model-health")
    assert response.status_code in {401, 403}

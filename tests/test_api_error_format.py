from fastapi.testclient import TestClient

from backend.main import app


def test_http_errors_use_consistent_json_shape():
    response = TestClient(app).get("/api/platform/status")
    assert response.status_code in {401, 403}
    data = response.json()
    assert data["status"] == "error"
    assert "message" in data
    assert "code" in data

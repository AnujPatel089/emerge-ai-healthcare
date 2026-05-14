from fastapi.testclient import TestClient

from backend.main import app


def test_command_center_requires_auth():
    response = TestClient(app).get("/api/command-center/summary")
    assert response.status_code in {401, 403}

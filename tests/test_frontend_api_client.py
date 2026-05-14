from frontend.api_client import API_URL


def test_frontend_api_client_has_default_url():
    assert API_URL.startswith("http")

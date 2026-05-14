from pathlib import Path


def test_render_config_uses_web_services_without_docker():
    text = Path("render.yaml").read_text(encoding="utf-8")
    assert "type: web" in text
    assert "uvicorn backend.main:app" in text
    assert "streamlit run frontend/app.py" in text or "python frontend/run.py" in text
    assert "MODEL_PATH" in text
    assert "FEATURE_COLUMNS_PATH" in text
    assert not Path("Dockerfile").exists()
    assert not Path("docker-compose.yml").exists()

from src.platform.runtime_config import validate_runtime_config


def test_render_config_validation_contract():
    result = validate_runtime_config()
    assert "status" in result
    assert "checks" in result

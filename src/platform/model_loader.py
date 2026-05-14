from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from src.platform.reliability_monitor import error_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Render Free Tier cold starts are slower and memory is limited, so the model is
# loaded only when prediction is requested. The cached object is reused after that.
_MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "label_encoder": None,
    "model_path": None,
    "label_encoder_path": None,
    "model_error": None,
    "label_encoder_error": None,
}


def _candidate_paths(env_key: str, defaults: list[str]) -> list[Path]:
    paths: list[Path] = []
    env_value = os.getenv(env_key)
    if env_value:
        paths.append(PROJECT_ROOT / env_value)
    for item in defaults:
        paths.append(PROJECT_ROOT / item)
        paths.append(PROJECT_ROOT / "backend" / item)
    return paths


def clear_model_cache() -> None:
    """Clear cached objects after a failure so the next request can retry safely."""
    _MODEL_CACHE.update(
        {
            "model": None,
            "label_encoder": None,
            "model_path": None,
            "label_encoder_path": None,
            "model_error": None,
            "label_encoder_error": None,
        }
    )


def load_model_bundle(force_reload: bool = False) -> dict[str, Any]:
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["label_encoder"] is not None and not force_reload:
        return {**_MODEL_CACHE, "mode": "ml", "available": True}

    if force_reload:
        clear_model_cache()

    model_paths = _candidate_paths(
        "MODEL_PATH",
        [
            "models/triage_xgboost_balanced.pkl",
            "models/emergency_triage_model.pkl",
            "models/triage_xgboost_model.pkl",
        ],
    )
    encoder_paths = _candidate_paths("LABEL_ENCODER_PATH", ["models/esi_label_encoder.pkl"])

    for path in model_paths:
        try:
            if path.exists():
                _MODEL_CACHE["model"] = joblib.load(path)
                _MODEL_CACHE["model_path"] = str(path)
                break
        except Exception as exc:
            _MODEL_CACHE["model_error"] = str(exc)
            error_logger().error("Model load failed from %s: %s", path, exc)

    for path in encoder_paths:
        try:
            if path.exists():
                _MODEL_CACHE["label_encoder"] = joblib.load(path)
                _MODEL_CACHE["label_encoder_path"] = str(path)
                break
        except Exception as exc:
            _MODEL_CACHE["label_encoder_error"] = str(exc)
            error_logger().error("Label encoder load failed from %s: %s", path, exc)

    available = _MODEL_CACHE["model"] is not None and _MODEL_CACHE["label_encoder"] is not None
    if not available:
        # Fallback mode keeps the backend alive when model artifacts are absent on
        # Render or the service is waking up with an ephemeral filesystem.
        return {**_MODEL_CACHE, "mode": "fallback_rules", "available": False}
    return {**_MODEL_CACHE, "mode": "ml", "available": True}


def model_status() -> dict[str, Any]:
    bundle = load_model_bundle(force_reload=False) if _MODEL_CACHE["model"] is not None else {**_MODEL_CACHE}
    model_path_exists = any(
        path.exists()
        for path in _candidate_paths(
            "MODEL_PATH",
            ["models/triage_xgboost_balanced.pkl", "models/emergency_triage_model.pkl", "models/triage_xgboost_model.pkl"],
        )
    )
    return {
        "loaded": _MODEL_CACHE["model"] is not None,
        "label_encoder_loaded": _MODEL_CACHE["label_encoder"] is not None,
        "model_path": _MODEL_CACHE["model_path"],
        "label_encoder_path": _MODEL_CACHE["label_encoder_path"],
        "model_path_exists": model_path_exists,
        "mode": "ml" if _MODEL_CACHE["model"] is not None else "lazy" if model_path_exists else "fallback_rules",
        "model_error": _MODEL_CACHE["model_error"],
        "label_encoder_error": _MODEL_CACHE["label_encoder_error"],
    }

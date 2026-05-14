from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_render_environment() -> bool:
    return os.getenv("RENDER", "").lower() == "true" or bool(os.getenv("RENDER_SERVICE_ID"))


def runtime_config() -> dict[str, Any]:
    return {
        "render_environment": is_render_environment(),
        "database_url_configured": bool(os.getenv("DATABASE_URL")),
        "secret_key_configured": bool(os.getenv("SECRET_KEY")),
        "api_url": os.getenv("API_URL"),
        "cors_origins": [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()],
        "model_path": os.getenv("MODEL_PATH", "models/triage_xgboost_balanced.pkl"),
        "feature_columns_path": os.getenv("FEATURE_COLUMNS_PATH", "models/emergency_feature_columns.pkl"),
    }


def validate_runtime_config() -> dict[str, Any]:
    config = runtime_config()
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    model_path = PROJECT_ROOT / config["model_path"]
    feature_path = PROJECT_ROOT / config["feature_columns_path"]
    cors_origins = config["cors_origins"]
    api_url = config["api_url"] or ""

    add("DATABASE_URL", config["database_url_configured"], "DATABASE_URL is configured." if config["database_url_configured"] else "DATABASE_URL is missing.")
    add("SECRET_KEY", config["secret_key_configured"], "SECRET_KEY is configured." if config["secret_key_configured"] else "SECRET_KEY is missing.")
    add("MODEL_PATH", model_path.exists(), f"Model path checked: {model_path}")
    add("FEATURE_COLUMNS_PATH", feature_path.exists(), f"Feature columns path checked: {feature_path}")
    add("CORS_ORIGINS", bool(cors_origins), "CORS_ORIGINS configured." if cors_origins else "CORS_ORIGINS should include the frontend Render URL.")
    add("API_URL", bool(api_url), f"API_URL configured: {api_url}" if api_url else "API_URL is not set for frontend/backend coordination.")

    frontend_render_allowed = True
    if api_url and "onrender.com" in api_url:
        frontend_render_allowed = any("onrender.com" in origin for origin in cors_origins)
    add("render_cors_alignment", frontend_render_allowed, "CORS appears Render-aware." if frontend_render_allowed else "CORS may not include the Render frontend URL.")

    return {
        "status": "healthy" if all(item["ok"] for item in checks[:5]) else "warning",
        "config": config,
        "checks": checks,
    }

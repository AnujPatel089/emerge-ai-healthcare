from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.mlops.model_registry import get_current_production_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_render_deployment(db: Session) -> dict[str, Any]:
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    try:
        db.execute(text("SELECT 1"))
        add("database", True, "PostgreSQL connection is available.")
    except Exception as exc:
        add("database", False, f"Database check failed: {exc}")

    production = get_current_production_model(db)
    model_path = Path(production.get("model_path") or "")
    add("model_registry", True, f"Production registry version: {production.get('model_version')}")
    add("model_artifact", model_path.exists(), f"Artifact path: {model_path}")
    add("secret_key", bool(os.getenv("SECRET_KEY")), "SECRET_KEY is configured." if os.getenv("SECRET_KEY") else "SECRET_KEY is missing.")
    add("database_url", bool(os.getenv("DATABASE_URL")), "DATABASE_URL is configured." if os.getenv("DATABASE_URL") else "DATABASE_URL is missing.")
    add("cors_origins", bool(os.getenv("CORS_ORIGINS")), "CORS_ORIGINS is configured." if os.getenv("CORS_ORIGINS") else "CORS_ORIGINS should include the Render frontend URL.")
    add("render_flag", os.getenv("RENDER", "").lower() == "true", "RENDER=true detected." if os.getenv("RENDER", "").lower() == "true" else "Local/non-Render environment detected.")

    status = "healthy" if all(item["ok"] for item in checks if item["name"] != "render_flag") else "warning"
    if not any(item["ok"] for item in checks if item["name"] == "database"):
        status = "critical"

    return {
        "status": status,
        "render_environment_detected": os.getenv("RENDER", "").lower() == "true" or bool(os.getenv("RENDER_SERVICE_ID")),
        "api_base_url": os.getenv("API_URL") or os.getenv("RENDER_EXTERNAL_URL"),
        "checks": checks,
        "filesystem_warning": "Render local disk may be ephemeral; PostgreSQL should hold critical MLOps metadata.",
    }

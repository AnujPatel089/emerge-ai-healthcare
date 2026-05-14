import os
from fastapi import APIRouter, Depends

from src.platform.runtime_config import validate_runtime_config


def build_router(require_role):
    router = APIRouter(prefix="/api/platform", tags=["Render Status"])

    @router.get("/render-status")
    def render_status(current_user=Depends(require_role(["doctor", "admin", "super_admin"]))):
        validation = validate_runtime_config()
        config = validation["config"]
        return {
            "backend_url": os.getenv("RENDER_EXTERNAL_URL"),
            "frontend_url": os.getenv("FRONTEND_URL"),
            "api_url": config.get("api_url"),
            "database_url_configured": config["database_url_configured"],
            "secret_key_configured": config["secret_key_configured"],
            "cors_configured": bool(config["cors_origins"]),
            "model_path_valid": next((c["ok"] for c in validation["checks"] if c["name"] == "MODEL_PATH"), False),
            "feature_columns_path_valid": next((c["ok"] for c in validation["checks"] if c["name"] == "FEATURE_COLUMNS_PATH"), False),
            "render_environment_detected": config["render_environment"],
            "checks": validation["checks"],
        }

    return router

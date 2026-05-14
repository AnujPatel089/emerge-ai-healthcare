from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.platform.ai_reliability import ai_reliability_status
from src.platform.incident_manager import list_incidents
from src.platform.runtime_config import validate_runtime_config
from src.platform.system_health import health_summary


def platform_status(db: Session, *, model_loaded: bool, label_encoder_loaded: bool) -> dict[str, Any]:
    health = health_summary(db, model_loaded=model_loaded, label_encoder_loaded=label_encoder_loaded)
    reliability = ai_reliability_status(db)
    incidents = list_incidents(db, include_resolved=False, limit=25)
    runtime = validate_runtime_config()

    status = "healthy"
    if health["status"] != "ok" or reliability["status"] == "critical" or any(i["severity"] == "critical" for i in incidents):
        status = "critical"
    elif reliability["status"] == "warning" or incidents or runtime["status"] == "warning":
        status = "warning"

    return {
        "status": status,
        "health": health,
        "ai_reliability": reliability,
        "active_incidents": incidents,
        "runtime_validation": runtime,
    }

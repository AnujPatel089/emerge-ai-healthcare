from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.mlops.model_registry import get_current_production_model
from src.models import EmergencyQueue, MLPredictionMonitoring, Nurse
from src.platform.reliability_monitor import request_metrics, uptime_seconds
from src.platform.runtime_config import is_render_environment


def database_status(db: Session) -> str:
    try:
        db.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "disconnected"


def _safe_count(query, default: int = 0) -> int:
    try:
        return query.count()
    except Exception:
        return default


def health_summary(db: Session, *, model_loaded: bool, label_encoder_loaded: bool) -> dict[str, Any]:
    db_status = database_status(db)
    try:
        production = get_current_production_model(db)
        model_version = production["model_version"]
    except Exception:
        model_version = "unknown"
    active_nurses = _safe_count(db.query(Nurse).filter(Nurse.available_status.is_(True))) if db_status == "connected" else 0
    queue_count = _safe_count(db.query(EmergencyQueue)) if db_status == "connected" else 0
    failed_predictions = (
        _safe_count(db.query(MLPredictionMonitoring).filter(MLPredictionMonitoring.failed.is_(True)))
        if db_status == "connected"
        else 0
    )

    metrics = request_metrics()
    return {
        "status": "ok" if db_status == "connected" and model_loaded else "degraded",
        "database": db_status,
        "model_loaded": model_loaded,
        "label_encoder_loaded": label_encoder_loaded,
        "model_version": model_version,
        "prediction_logging": "enabled" if db_status == "connected" else "degraded",
        "active_nurses": active_nurses,
        "queue_count": queue_count,
        "render_environment": is_render_environment(),
        "uptime_seconds": uptime_seconds(),
        "api_latency_ms": metrics["average_api_latency_ms"],
        "failed_requests": metrics["failed_requests"],
        "failed_predictions": failed_predictions,
    }

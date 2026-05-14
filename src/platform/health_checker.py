from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models import EmergencyQueue, MLPredictionMonitoring, Nurse
from src.platform.reliability_monitor import request_metrics
from src.platform.runtime_config import PROJECT_ROOT, runtime_config, validate_runtime_config


def run_platform_checks(db: Session, model_loaded: bool = True) -> dict[str, Any]:
    config_result = validate_runtime_config()
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "critical"

    config = runtime_config()
    model_path = PROJECT_ROOT / config["model_path"]
    feature_path = PROJECT_ROOT / config["feature_columns_path"]
    checks["model"] = "healthy" if model_loaded and model_path.exists() else "fallback_active"
    checks["feature_columns"] = "healthy" if feature_path.exists() else "manual_required"

    try:
        queue_count = db.query(EmergencyQueue).count()
        active_nurses = db.query(Nurse).filter(Nurse.available_status.is_(True)).count()
        checks["queue"] = "healthy" if queue_count >= 0 else "degraded"
        checks["nurse_assignment"] = "healthy" if active_nurses > 0 else "manual_required"
    except Exception:
        queue_count = 0
        active_nurses = 0
        checks["queue"] = "degraded"
        checks["nurse_assignment"] = "manual_required"

    metrics = request_metrics()
    checks["api_latency"] = "degraded" if (metrics["average_api_latency_ms"] or 0) > 3000 else "healthy"
    checks["repeated_errors"] = "degraded" if metrics["failed_requests"] >= 5 else "healthy"
    failed_predictions = 0
    try:
        failed_predictions = db.query(MLPredictionMonitoring).filter(MLPredictionMonitoring.failed.is_(True)).count()
    except Exception:
        pass
    checks["prediction"] = "degraded" if failed_predictions else "healthy"

    statuses = list(checks.values()) + [config_result["status"]]
    overall = "healthy"
    if "critical" in statuses:
        overall = "critical"
    elif "manual_required" in statuses:
        overall = "manual_required"
    elif "degraded" in statuses or "warning" in statuses or "fallback_active" in statuses:
        overall = "degraded"

    message = "Platform healthy."
    if checks["model"] == "fallback_active":
        message = "ML model unavailable. Rule-based fallback is active. Clinician review required."
    elif overall != "healthy":
        message = "System is running in degraded mode. Manual review required."

    return {
        "overall_status": overall,
        "backend": "healthy",
        "database": checks["database"],
        "model": checks["model"],
        "queue": checks["queue"],
        "upload_ocr": "healthy",
        "prediction": checks["prediction"],
        "nurse_assignment": checks["nurse_assignment"],
        "render_config": config_result["status"],
        "active_nurses": active_nurses,
        "queue_count": queue_count,
        "active_incidents": 0,
        "recovery_available": overall in {"degraded", "critical", "manual_required"},
        "message": message,
        "checks": checks,
        "runtime_validation": config_result,
    }

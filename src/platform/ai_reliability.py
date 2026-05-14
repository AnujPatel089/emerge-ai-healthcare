from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.mlops.model_monitor import model_health
from src.models import MLPredictionMonitoring


def _is_critical_esi(value: Any) -> bool:
    text = str(value or "")
    return "1" in text or "2" in text


def ai_reliability_status(db: Session) -> dict[str, Any]:
    health = model_health(db)
    rows = (
        db.query(MLPredictionMonitoring)
        .order_by(MLPredictionMonitoring.timestamp.desc())
        .limit(500)
        .all()
    )
    low_confidence = sum(1 for row in rows if row.confidence is not None and row.confidence < 0.55)
    failed_predictions = sum(1 for row in rows if row.failed)
    critical_cases = sum(1 for row in rows if _is_critical_esi(row.final_esi or row.predicted_esi))
    possible_under_triage = sum(
        1
        for row in rows
        if (row.safety_rule_triggered or (row.icu_risk is not None and row.icu_risk >= 0.7))
        and not _is_critical_esi(row.final_esi or row.predicted_esi)
    )
    missing_feature_rows = sum(1 for row in rows if not row.input_features or row.input_features == "{}")

    alerts = []
    if low_confidence >= max(5, len(rows) * 0.2):
        alerts.append({"level": "warning", "message": "Low confidence predictions increased; clinician review required."})
    if possible_under_triage:
        alerts.append({"level": "critical", "message": "Possible under-triage pattern detected; AI-supported triage only."})
    if health.get("override_rate", 0) >= 0.2:
        alerts.append({"level": "warning", "message": "Doctor override rate is elevated; review possible risk patterns."})
    if failed_predictions:
        alerts.append({"level": "critical", "message": "Failed predictions detected; clinician review required."})
    if health.get("drift_status") in {"warning", "critical"}:
        alerts.append({"level": health["drift_status"], "message": "MLOps drift warning is active."})
    if missing_feature_rows:
        alerts.append({"level": "warning", "message": "Some prediction rows are missing feature payloads."})

    status = "healthy"
    if any(alert["level"] == "critical" for alert in alerts):
        status = "critical"
    elif alerts:
        status = "warning"

    return {
        "status": status,
        "message": "AI-supported triage only. Outputs describe possible risk; clinician review required.",
        "low_confidence_predictions": low_confidence,
        "possible_under_triage_patterns": possible_under_triage,
        "doctor_override_rate": health.get("override_rate", 0),
        "esi_1_2_cases": critical_cases,
        "failed_predictions": failed_predictions,
        "failed_shap_explanations": 0,
        "missing_feature_columns": missing_feature_rows,
        "drift_status": health.get("drift_status"),
        "alerts": alerts,
    }

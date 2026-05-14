from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from sqlalchemy.orm import Session

from src.mlops.drift_monitor import latest_drift_report
from src.mlops.model_registry import get_current_production_model
from src.models import ClinicalFeedback, MLPredictionMonitoring


def model_health(db: Session) -> dict[str, Any]:
    production = get_current_production_model(db)
    version = production["model_version"]
    rows = (
        db.query(MLPredictionMonitoring)
        .filter(MLPredictionMonitoring.model_version == version)
        .order_by(MLPredictionMonitoring.timestamp.desc())
        .limit(1000)
        .all()
    )
    count = len(rows)
    confidences = [row.confidence for row in rows if row.confidence is not None]
    latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
    average_confidence = round(mean(confidences), 4) if confidences else None
    average_latency = round(mean(latencies), 2) if latencies else None
    low_confidence_count = sum(1 for value in confidences if value < 0.55)
    failed_predictions = sum(1 for row in rows if row.failed)
    prediction_ids = [row.prediction_id for row in rows if row.prediction_id]
    override_count = 0
    if prediction_ids:
        override_count = (
            db.query(ClinicalFeedback)
            .filter(ClinicalFeedback.prediction_id.in_(prediction_ids))
            .filter(ClinicalFeedback.accepted.is_(False))
            .count()
        )
    override_rate = round(override_count / count, 4) if count else 0.0
    esi_distribution = dict(Counter(str(row.final_esi or row.predicted_esi) for row in rows if row.final_esi or row.predicted_esi))
    drift = latest_drift_report(db, version)

    status = "healthy"
    recommendation = "Model monitoring is stable. Continue clinician review and routine observation."
    if failed_predictions > 0 or drift["drift_status"] == "critical" or override_rate >= 0.35:
        status = "critical"
        recommendation = "Clinician review required; investigate failures, overrides, and drift before promotion."
    elif drift["drift_status"] == "warning" or low_confidence_count >= max(5, count * 0.2) or override_rate >= 0.15:
        status = "warning"
        recommendation = "Monitor closely and prepare a validated retraining run if this pattern continues."

    return {
        "model_version": version,
        "status": status,
        "prediction_count": count,
        "average_confidence": average_confidence,
        "low_confidence_predictions": low_confidence_count,
        "override_rate": override_rate,
        "drift_status": drift["drift_status"],
        "drift_score": drift["drift_score"],
        "latency_ms": average_latency,
        "failed_predictions": failed_predictions,
        "esi_distribution": esi_distribution,
        "recommendation": recommendation,
    }

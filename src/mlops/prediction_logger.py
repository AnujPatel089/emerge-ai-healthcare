from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.models import ClinicalFeedback, MLPredictionMonitoring
from src.triage_rules import estimate_icu_risk, estimate_readmission_risk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "prediction_monitoring.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _parse_risk(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def estimate_operational_risks(features: dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        icu = estimate_icu_risk(features)
    except Exception:
        icu = None
    try:
        readmission = estimate_readmission_risk(features)
    except Exception:
        readmission = None
    return _parse_risk(icu), _parse_risk(readmission)


def log_prediction_monitoring(
    db: Session,
    *,
    prediction_id: int | None,
    patient_id: str | None,
    model_version: str,
    input_features: dict[str, Any],
    predicted_esi: str | None,
    confidence: float | None,
    safety_rule_triggered: bool,
    final_esi: str | None,
    latency_ms: float | None,
    failed: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    icu_risk, readmission_risk = estimate_operational_risks(input_features)
    doctor_override = False
    if prediction_id:
        doctor_override = (
            db.query(ClinicalFeedback)
            .filter(ClinicalFeedback.prediction_id == prediction_id)
            .filter(ClinicalFeedback.accepted.is_(False))
            .count()
            > 0
        )

    row = MLPredictionMonitoring(
        prediction_id=prediction_id,
        patient_id=patient_id,
        model_version=model_version,
        input_features=_to_json(input_features),
        predicted_esi=str(predicted_esi) if predicted_esi is not None else None,
        confidence=confidence,
        icu_risk=icu_risk,
        readmission_risk=readmission_risk,
        safety_rule_triggered=safety_rule_triggered,
        doctor_override=doctor_override,
        final_esi=str(final_esi) if final_esi is not None else None,
        latency_ms=latency_ms,
        failed=failed,
        error_message=error_message,
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
        return serialize_monitoring_row(row)
    except Exception:
        db.rollback()
        append_prediction_csv(
            {
                "prediction_id": prediction_id,
                "patient_id": patient_id,
                "model_version": model_version,
                "input_features": _to_json(input_features),
                "predicted_esi": predicted_esi,
                "confidence": confidence,
                "icu_risk": icu_risk,
                "readmission_risk": readmission_risk,
                "safety_rule_triggered": safety_rule_triggered,
                "doctor_override": doctor_override,
                "final_esi": final_esi,
                "latency_ms": latency_ms,
                "failed": failed,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        raise


def append_prediction_csv(row: dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def serialize_monitoring_row(row: MLPredictionMonitoring) -> dict[str, Any]:
    return {
        "id": row.id,
        "prediction_id": row.prediction_id,
        "patient_id": row.patient_id,
        "model_version": row.model_version,
        "input_features": json.loads(row.input_features) if row.input_features else {},
        "predicted_esi": row.predicted_esi,
        "confidence": row.confidence,
        "icu_risk": row.icu_risk,
        "readmission_risk": row.readmission_risk,
        "safety_rule_triggered": row.safety_rule_triggered,
        "doctor_override": row.doctor_override,
        "final_esi": row.final_esi,
        "latency_ms": row.latency_ms,
        "failed": row.failed,
        "error_message": row.error_message,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
    }


def get_prediction_monitoring(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    rows = (
        db.query(MLPredictionMonitoring)
        .order_by(MLPredictionMonitoring.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [serialize_monitoring_row(row) for row in rows]

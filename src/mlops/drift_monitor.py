from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from sqlalchemy.orm import Session

from src.models import MLDriftReport, MLPredictionMonitoring, PredictionLog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "age",
    "triage_vital_hr",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "triage_vital_sbp",
    "triage_vital_dbp",
]

BINARY_FEATURES = [
    "cc_chestpain",
    "cc_shortnessofbreath",
]


def _load_features(row: MLPredictionMonitoring) -> dict[str, Any]:
    try:
        return json.loads(row.input_features or "{}")
    except Exception:
        return {}


def _status(score: float) -> str:
    if score >= 0.55:
        return "critical"
    if score >= 0.25:
        return "warning"
    return "stable"


def _numeric_drift(baseline: list[float], live: list[float]) -> float:
    if not baseline or not live:
        return 0.0
    base_mean = mean(baseline)
    live_mean = mean(live)
    spread = pstdev(baseline) or max(abs(base_mean), 1.0)
    return min(abs(live_mean - base_mean) / spread, 3.0) / 3.0


def _frequency_drift(baseline: list[Any], live: list[Any]) -> float:
    if not baseline or not live:
        return 0.0
    base = Counter(str(v) for v in baseline)
    current = Counter(str(v) for v in live)
    keys = set(base) | set(current)
    total_base = sum(base.values()) or 1
    total_live = sum(current.values()) or 1
    distance = sum(abs(base[k] / total_base - current[k] / total_live) for k in keys) / 2
    return min(distance, 1.0)


def _prediction_baseline(db: Session, limit: int = 500) -> list[dict[str, Any]]:
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "age": r.age,
            "triage_vital_hr": r.triage_vital_hr,
            "triage_vital_rr": r.triage_vital_rr,
            "triage_vital_o2": r.triage_vital_o2,
            "triage_vital_temp": r.triage_vital_temp,
            "triage_vital_sbp": r.triage_vital_sbp,
            "triage_vital_dbp": r.triage_vital_dbp,
            "cc_chestpain": r.cc_chestpain,
            "cc_shortnessofbreath": r.cc_shortnessofbreath,
            "icu_risk": None,
            "esi": r.final_prediction,
        }
        for r in rows
    ]


def _live_window(db: Session, limit: int = 200) -> list[dict[str, Any]]:
    rows = (
        db.query(MLPredictionMonitoring)
        .order_by(MLPredictionMonitoring.timestamp.desc())
        .limit(limit)
        .all()
    )
    live = []
    for row in rows:
        features = _load_features(row)
        features["icu_risk"] = row.icu_risk
        features["esi"] = row.final_esi or row.predicted_esi
        live.append(features)
    return live


def compute_drift_report(db: Session, model_version: str, persist: bool = True) -> dict[str, Any]:
    baseline = _prediction_baseline(db)
    live = _live_window(db)
    feature_drift: dict[str, float] = {}

    for feature in NUMERIC_FEATURES + ["icu_risk"]:
        base_values = [float(row[feature]) for row in baseline if row.get(feature) is not None]
        live_values = [float(row[feature]) for row in live if row.get(feature) is not None]
        feature_drift[feature] = round(_numeric_drift(base_values, live_values), 4)

    for feature in BINARY_FEATURES + ["esi"]:
        feature_drift[feature] = round(
            _frequency_drift(
                [row.get(feature) for row in baseline if row.get(feature) is not None],
                [row.get(feature) for row in live if row.get(feature) is not None],
            ),
            4,
        )

    drift_score = round(mean(feature_drift.values()) if feature_drift else 0.0, 4)
    drift_status = _status(drift_score)
    recommendation = {
        "stable": "Continue monitoring AI-supported triage performance.",
        "warning": "Review recent prediction mix and clinician overrides; consider retraining if this persists.",
        "critical": "Clinician review required for model governance; prepare offline retraining and validation.",
    }[drift_status]

    report = {
        "model_version": model_version,
        "drift_score": drift_score,
        "drift_status": drift_status,
        "feature_drift": feature_drift,
        "baseline_window": f"{len(baseline)} historical predictions",
        "live_window": f"{len(live)} monitored predictions",
        "recommendation": recommendation,
        "created_at": datetime.utcnow().isoformat(),
    }

    if persist:
        save_drift_report(db, report)
    write_drift_fallback(report)
    return report


def save_drift_report(db: Session, report: dict[str, Any]) -> None:
    row = MLDriftReport(
        model_version=report["model_version"],
        drift_score=report["drift_score"],
        drift_status=report["drift_status"],
        feature_drift=json.dumps(report["feature_drift"], sort_keys=True),
        baseline_window=report["baseline_window"],
        live_window=report["live_window"],
        recommendation=report["recommendation"],
    )
    try:
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()


def latest_drift_report(db: Session, model_version: str) -> dict[str, Any]:
    row = (
        db.query(MLDriftReport)
        .filter(MLDriftReport.model_version == model_version)
        .order_by(MLDriftReport.created_at.desc())
        .first()
    )
    if not row:
        return compute_drift_report(db, model_version)
    return {
        "model_version": row.model_version,
        "drift_score": row.drift_score,
        "drift_status": row.drift_status,
        "feature_drift": json.loads(row.feature_drift or "{}"),
        "baseline_window": row.baseline_window,
        "live_window": row.live_window,
        "recommendation": row.recommendation,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def write_drift_fallback(report: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "drift_report.json"
    csv_path = REPORT_DIR / "drift_report.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "drift_score"])
        writer.writeheader()
        for feature, score in report.get("feature_drift", {}).items():
            writer.writerow({"feature": feature, "drift_score": score})

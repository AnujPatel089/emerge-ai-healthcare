from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sqlalchemy.orm import Session

from src.mlops.model_card_generator import save_model_card
from src.mlops.model_registry import (
    ModelRegistryRecord,
    copy_artifact_to_registry,
    get_current_production_model,
    next_model_version,
    register_model,
)
from src.models import PredictionLog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"

FEATURE_COLUMNS = [
    "age",
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "cc_chestpain",
    "cc_shortnessofbreath",
    "cc_headache",
    "cc_fever",
    "cc_abdominalpain",
    "cc_dizziness",
    "cc_syncope",
    "cc_weakness",
]


def _extract_esi(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    for candidate in ["1", "2", "3", "4", "5"]:
        if candidate in text:
            return int(candidate)
    return None


def load_feedback_dataset(db: Session) -> pd.DataFrame:
    rows = db.query(PredictionLog).all()
    records = []
    for row in rows:
        label = _extract_esi(row.final_prediction)
        if label is None:
            continue
        records.append(
            {
                "age": row.age,
                "triage_vital_hr": row.triage_vital_hr,
                "triage_vital_sbp": row.triage_vital_sbp,
                "triage_vital_dbp": row.triage_vital_dbp,
                "triage_vital_rr": row.triage_vital_rr,
                "triage_vital_o2": row.triage_vital_o2,
                "triage_vital_temp": row.triage_vital_temp,
                "cc_chestpain": row.cc_chestpain,
                "cc_shortnessofbreath": row.cc_shortnessofbreath,
                "cc_headache": row.cc_headache,
                "cc_fever": row.cc_fever,
                "cc_abdominalpain": row.cc_abdominalpain,
                "cc_dizziness": row.cc_dizziness,
                "cc_syncope": row.cc_syncope,
                "cc_weakness": row.cc_weakness,
                "label": label,
            }
        )
    return pd.DataFrame(records)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    failures = []
    if df.empty or len(df) < 30:
        failures.append("At least 30 labeled prediction records are required for safe retraining.")
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        failures.append(f"Missing feature columns: {', '.join(missing)}")
    if "label" not in df.columns or df["label"].nunique() < 2:
        failures.append("At least two ESI classes are required for candidate training.")
    return failures


def _metrics(y_true, y_pred) -> dict[str, float]:
    critical_true = [1 if int(y) in {1, 2} else 0 for y in y_true]
    critical_pred = [1 if int(y) in {1, 2} else 0 for y in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_esi_1_2": float(recall_score(critical_true, critical_pred, zero_division=0)),
    }


def run_retraining(db: Session, requested_by: str) -> dict[str, Any]:
    version = next_model_version(db)
    production = get_current_production_model(db)
    candidate_record: dict[str, Any] | None = None

    try:
        df = load_feedback_dataset(db)
        failures = validate_dataset(df)
        if failures:
            candidate_record = register_model(
                db,
                ModelRegistryRecord(
                    model_name="esi_model",
                    model_version=version,
                    model_path=str(REGISTRY_DIR / f"{version}.pkl"),
                    training_dataset="prediction_logs with clinician feedback",
                    training_date=datetime.utcnow(),
                    status="candidate",
                    notes="Retraining validation failed: " + "; ".join(failures),
                ),
            )
            return {
                "status": "failed_validation",
                "validation_failures": failures,
                "candidate": candidate_record,
                "recommendation": "Run retraining locally/offline after collecting more reviewed examples.",
            }

        df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
        X = df[FEATURE_COLUMNS]
        y = df["label"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() > 1 and y.value_counts().min() > 1 else None,
        )
        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train - 1)
        raw_pred = model.predict(X_test)
        y_pred = raw_pred + 1
        metrics = _metrics(y_test, y_pred)

        artifact_path = REGISTRY_DIR / f"{version}.pkl"
        joblib.dump(model, artifact_path)
        feature_path = REGISTRY_DIR / f"{version}_feature_columns.json"
        feature_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

        old_f1 = production.get("f1_score") or 0
        old_critical_recall = production.get("recall_esi_1_2") or 0
        can_stage = (
            (metrics["f1_score"] > old_f1 or metrics["recall_esi_1_2"] > old_critical_recall)
            and not validate_dataset(df)
        )
        status = "staging" if can_stage else "candidate"
        notes = "Candidate eligible for staging." if can_stage else "Candidate did not outperform production guardrail."

        candidate_record = register_model(
            db,
            ModelRegistryRecord(
                model_name="esi_model",
                model_version=version,
                model_path=copy_artifact_to_registry(artifact_path, version),
                feature_columns_path=str(feature_path),
                training_dataset="prediction_logs with clinician feedback",
                training_date=datetime.utcnow(),
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                recall_esi_1_2=metrics["recall_esi_1_2"],
                status=status,
                deployed_by=requested_by if status == "staging" else None,
                notes=notes,
            ),
        )
        model_card = save_model_card(db, candidate_record, created_by=requested_by)
        return {
            "status": "staging" if can_stage else "candidate",
            "candidate": candidate_record,
            "metrics": metrics,
            "model_card": model_card,
            "promotion_rule": "f1_score improves OR recall for ESI 1/2 improves, with no major validation failure.",
            "recommendation": "Promote only after admin review and deployment validation.",
        }
    except MemoryError:
        return {
            "status": "resource_limited",
            "candidate": candidate_record,
            "recommendation": "Render resource limit reached. Run retraining locally/offline, then register and promote the validated artifact.",
        }
    except Exception as exc:
        try:
            register_model(
                db,
                ModelRegistryRecord(
                    model_name="esi_model",
                    model_version=version,
                    model_path=str(REGISTRY_DIR / f"{version}.pkl"),
                    training_dataset="prediction_logs with clinician feedback",
                    training_date=datetime.utcnow(),
                    status="candidate",
                    notes=f"Retraining failed safely: {exc}",
                ),
            )
        except Exception:
            pass
        return {
            "status": "failed",
            "error": str(exc),
            "recommendation": "No model was deployed. Review logs and prefer local/offline retraining for heavier jobs.",
        }

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.models import MLModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

VALID_STATUSES = {"candidate", "staging", "production", "archived"}


@dataclass
class ModelRegistryRecord:
    model_name: str
    model_version: str
    model_path: str
    feature_columns_path: str | None = None
    training_dataset: str | None = None
    training_date: datetime | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    recall_esi_1_2: float | None = None
    confusion_matrix_path: str | None = None
    feature_importance_path: str | None = None
    status: str = "candidate"
    deployed_at: datetime | None = None
    deployed_by: str | None = None
    notes: str | None = None


def _serialize(record: MLModelRegistry) -> dict[str, Any]:
    return {
        "id": record.id,
        "model_name": record.model_name,
        "model_version": record.model_version,
        "model_path": record.model_path,
        "feature_columns_path": record.feature_columns_path,
        "training_dataset": record.training_dataset,
        "training_date": record.training_date.isoformat() if record.training_date else None,
        "accuracy": record.accuracy,
        "precision": record.precision,
        "recall": record.recall,
        "f1_score": record.f1_score,
        "recall_esi_1_2": record.recall_esi_1_2,
        "confusion_matrix_path": record.confusion_matrix_path,
        "feature_importance_path": record.feature_importance_path,
        "status": record.status,
        "deployed_at": record.deployed_at.isoformat() if record.deployed_at else None,
        "deployed_by": record.deployed_by,
        "notes": record.notes,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def register_model(db: Session, record: ModelRegistryRecord) -> dict[str, Any]:
    if record.status not in VALID_STATUSES:
        raise ValueError(f"Invalid model status: {record.status}")

    existing = (
        db.query(MLModelRegistry)
        .filter(MLModelRegistry.model_version == record.model_version)
        .first()
    )
    data = asdict(record)
    if data["training_date"] is None:
        data["training_date"] = datetime.utcnow()

    if existing:
        for key, value in data.items():
            setattr(existing, key, value)
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return _serialize(existing)

    item = MLModelRegistry(**data)
    db.add(item)
    db.commit()
    db.refresh(item)
    return _serialize(item)


def list_models(db: Session) -> list[dict[str, Any]]:
    rows = db.query(MLModelRegistry).order_by(MLModelRegistry.created_at.desc()).all()
    return [_serialize(row) for row in rows]


def get_model_by_version(db: Session, version: str) -> Optional[dict[str, Any]]:
    row = (
        db.query(MLModelRegistry)
        .filter(MLModelRegistry.model_version == version)
        .first()
    )
    return _serialize(row) if row else None


def get_current_production_model(db: Session) -> dict[str, Any]:
    row = (
        db.query(MLModelRegistry)
        .filter(MLModelRegistry.status == "production")
        .order_by(MLModelRegistry.deployed_at.desc().nullslast(), MLModelRegistry.created_at.desc())
        .first()
    )
    if row:
        return _serialize(row)

    fallback = ensure_bootstrap_registry(db)
    return fallback


def next_model_version(db: Session, model_name: str = "esi_model") -> str:
    count = db.query(MLModelRegistry).filter(MLModelRegistry.model_name == model_name).count()
    return f"{model_name}_v{count + 1}"


def copy_artifact_to_registry(source_path: str | Path, version: str) -> str:
    source = Path(source_path)
    target = REGISTRY_DIR / f"{version}{source.suffix or '.pkl'}"
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def promote_model(db: Session, version: str, deployed_by: str) -> dict[str, Any]:
    target = (
        db.query(MLModelRegistry)
        .filter(MLModelRegistry.model_version == version)
        .first()
    )
    if not target:
        raise ValueError(f"Model version not found: {version}")
    if target.status not in {"staging", "production"}:
        raise ValueError("Only staging models can be promoted to production.")

    db.query(MLModelRegistry).filter(
        MLModelRegistry.status == "production",
        MLModelRegistry.model_version != version,
    ).update({"status": "archived"})

    target.status = "production"
    target.deployed_at = datetime.utcnow()
    target.deployed_by = deployed_by
    target.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(target)
    return _serialize(target)


def ensure_bootstrap_registry(db: Session) -> dict[str, Any]:
    existing = db.query(MLModelRegistry).first()
    if existing:
        return _serialize(existing)

    model_path = PROJECT_ROOT / "models" / "triage_xgboost_balanced.pkl"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "emergency_triage_model.pkl"
    feature_path = PROJECT_ROOT / "models" / "emergency_feature_columns.pkl"
    version = "esi_model_v1"

    try:
        registry_path = copy_artifact_to_registry(model_path, version)
    except Exception:
        registry_path = str(model_path)

    return register_model(
        db,
        ModelRegistryRecord(
            model_name="esi_model",
            model_version=version,
            model_path=registry_path,
            feature_columns_path=str(feature_path) if feature_path.exists() else None,
            training_dataset="Existing project training data",
            training_date=datetime.utcnow(),
            status="production",
            deployed_at=datetime.utcnow(),
            deployed_by="bootstrap",
            notes=(
                "Bootstrap registry entry for the currently deployed educational "
                "AI-supported triage model. Render local artifacts may be ephemeral; "
                "PostgreSQL metadata is the source of truth."
            ),
        ),
    )


def write_registry_snapshot(records: list[dict[str, Any]]) -> None:
    path = REGISTRY_DIR / "registry_snapshot.json"
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

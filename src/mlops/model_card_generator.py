from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.models import MLModelCard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_DIR = PROJECT_ROOT / "reports" / "model_cards"
MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)


def build_model_card(record: dict[str, Any]) -> str:
    metrics = {
        "accuracy": record.get("accuracy"),
        "precision": record.get("precision"),
        "recall": record.get("recall"),
        "f1_score": record.get("f1_score"),
        "recall_esi_1_2": record.get("recall_esi_1_2"),
    }
    metric_lines = "\n".join(f"- {key}: {value if value is not None else 'not available'}" for key, value in metrics.items())
    return f"""# Model Card: {record.get("model_version")}

## Purpose
This model supports educational AI-supported triage by estimating possible risk and an ESI-style priority level. Clinician review required.

## Training Data
- Dataset: {record.get("training_dataset") or "not documented"}
- Training date: {record.get("training_date") or "not documented"}
- Feature columns: {record.get("feature_columns_path") or "not documented"}

## Features
Age, arrival mode, vital signs, and chief complaint flags are used where available. The model must not be used for final diagnosis or treatment decisions.

## Metrics
{metric_lines}

## Limitations
This is educational/demo healthcare software. It may be affected by synthetic data quality, missing values, workflow bias, and changes in patient mix.

## Safety Disclaimer
Outputs describe possible risk only. AI-supported triage requires clinician review and must not replace emergency medical judgment.

## Bias Considerations
Monitor performance across demographic and arrival-mode groups. Review override patterns and low-confidence predictions for potential inequity.

## Clinical Limitations
The model does not produce a final medical diagnosis. It does not interpret all clinical context, labs, imaging, or bedside examination findings.

## Monitoring Plan
Track prediction volume, confidence, clinician overrides, ESI distribution, latency, failures, and data drift.

## Retraining Trigger
Consider retraining when drift is warning/critical, override rate increases, low-confidence predictions rise, or recall for ESI 1/2 decreases.

## Render Deployment Notes
Render service disks can be ephemeral. Keep registry metadata, monitoring logs, drift reports, and model cards in PostgreSQL. Local files under `models/registry` and `reports/model_cards` are fallback artifacts only.

Generated at {datetime.utcnow().isoformat()} UTC.
"""


def save_model_card(db: Session, record: dict[str, Any], created_by: str | None = None) -> dict[str, Any]:
    markdown = build_model_card(record)
    version = record["model_version"]
    path = MODEL_CARD_DIR / f"{version}.md"
    path.write_text(markdown, encoding="utf-8")

    existing = db.query(MLModelCard).filter(MLModelCard.model_version == version).first()
    if existing:
        existing.card_markdown = markdown
        existing.card_path = str(path)
        existing.created_by = created_by
        db.commit()
        db.refresh(existing)
        row = existing
    else:
        row = MLModelCard(
            model_version=version,
            model_name=record.get("model_name", "esi_model"),
            card_markdown=markdown,
            card_path=str(path),
            created_by=created_by,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

    return {
        "model_version": row.model_version,
        "model_name": row.model_name,
        "card_markdown": row.card_markdown,
        "card_path": row.card_path,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def get_model_card(db: Session, version: str) -> dict[str, Any] | None:
    row = db.query(MLModelCard).filter(MLModelCard.model_version == version).first()
    if not row:
        return None
    return {
        "model_version": row.model_version,
        "model_name": row.model_name,
        "card_markdown": row.card_markdown,
        "card_path": row.card_path,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }

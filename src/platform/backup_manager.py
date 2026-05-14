from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.models import (
    AppUser,
    AuditLog,
    ClinicalFeedback,
    EmergencyQueue,
    MLModelRegistry,
    NurseAssignment,
    PlatformAlert,
    PlatformIncident,
    PredictionLog,
)


def backup_metadata(db: Session) -> dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "tables": {
            "users_metadata": db.query(AppUser).count(),
            "predictions": db.query(PredictionLog).count(),
            "clinical_feedback": db.query(ClinicalFeedback).count(),
            "nurse_assignments": db.query(NurseAssignment).count(),
            "queue_records": db.query(EmergencyQueue).count(),
            "audit_logs": db.query(AuditLog).count(),
            "incident_logs": db.query(PlatformIncident).count(),
            "alert_logs": db.query(PlatformAlert).count(),
            "model_registry": db.query(MLModelRegistry).count(),
        },
        "excluded": ["password_hash", "SECRET_KEY", "DATABASE_URL", "SMTP_PASSWORD", "API keys"],
    }


def _safe_user(user: AppUser) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "requested_role": user.requested_role,
        "status": user.status,
        "account_status": user.account_status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def export_backup(db: Session) -> dict[str, Any]:
    return {
        "metadata": backup_metadata(db),
        "users_metadata": [_safe_user(row) for row in db.query(AppUser).limit(500).all()],
        "predictions": [{"id": r.id, "final_prediction": r.final_prediction, "created_at": r.created_at.isoformat() if r.created_at else None} for r in db.query(PredictionLog).limit(500).all()],
        "clinical_feedback": [{"id": r.id, "prediction_id": r.prediction_id, "accepted": r.accepted, "created_at": r.created_at.isoformat() if r.created_at else None} for r in db.query(ClinicalFeedback).limit(500).all()],
        "nurse_assignments": [{"id": r.id, "prediction_id": r.prediction_id, "nurse_id": r.nurse_id, "status": r.status} for r in db.query(NurseAssignment).limit(500).all()],
        "queue_records": [{"id": r.id, "prediction_id": r.prediction_id, "priority": r.priority, "status": r.status} for r in db.query(EmergencyQueue).limit(500).all()],
        "audit_logs": [{"id": r.id, "username": r.username, "action": r.action, "created_at": r.created_at.isoformat() if r.created_at else None} for r in db.query(AuditLog).limit(500).all()],
        "incident_logs": [{"id": r.id, "incident_type": r.incident_type, "severity": r.severity, "status": r.status} for r in db.query(PlatformIncident).limit(500).all()],
        "alert_logs": [{"id": r.id, "alert_type": r.alert_type, "severity": r.severity, "status": r.status} for r in db.query(PlatformAlert).limit(500).all()],
        "model_registry": [{"model_name": r.model_name, "model_version": r.model_version, "status": r.status} for r in db.query(MLModelRegistry).limit(500).all()],
    }

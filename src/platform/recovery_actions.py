from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def reconnect_database(db: Session) -> dict[str, Any]:
    try:
        db.rollback()
        db.execute(text("SELECT 1"))
        return {"status": "recovered", "action": "database_session_reconnected"}
    except Exception as exc:
        db.rollback()
        return {"status": "failed", "action": "database_session_reconnect", "detail": str(exc)}


def mark_rule_fallback_active() -> dict[str, Any]:
    return {
        "status": "recovered",
        "action": "rule_based_triage_fallback",
        "message": "Fallback rule-based triage used. Clinician review required.",
    }


def manual_assignment_required() -> dict[str, Any]:
    return {
        "status": "manual_required",
        "action": "manual_nurse_assignment_required",
        "message": "Manual assignment required. Patient remains in waiting queue.",
    }


def safe_recovery_for_incident(incident_type: str, db: Session | None = None) -> dict[str, Any]:
    if incident_type == "database_failure" and db is not None:
        return reconnect_database(db)
    if incident_type in {"model_failure", "prediction_failure"}:
        return mark_rule_fallback_active()
    if incident_type in {"queue_failure", "queue_assignment_failure"}:
        return manual_assignment_required()
    if incident_type in {"render_config_failure", "frontend_connection_failure"}:
        return {"status": "manual_required", "action": "render_configuration_review", "message": "Review Render environment variables and service URLs."}
    return {"status": "manual_required", "action": "operator_review", "message": "Manual review required."}

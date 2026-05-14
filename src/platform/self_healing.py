from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.models import PlatformIncident
from src.platform.health_checker import run_platform_checks
from src.platform.incident_manager import create_incident, list_incidents, resolve_incident
from src.platform.recovery_actions import safe_recovery_for_incident


def self_healing_status(db: Session, model_loaded: bool = True) -> dict[str, Any]:
    status = run_platform_checks(db, model_loaded=model_loaded)
    incidents = list_incidents(db, include_resolved=False)
    status["active_incidents"] = len(incidents)
    status["incidents"] = incidents
    return status


def run_checks_and_log(db: Session, model_loaded: bool = True) -> dict[str, Any]:
    status = self_healing_status(db, model_loaded=model_loaded)
    checks = status.get("checks", {})
    for service, check_status in checks.items():
        if check_status in {"critical", "degraded", "manual_required", "fallback_active"}:
            incident_type = {
                "database": "database_failure",
                "model": "model_failure",
                "prediction": "prediction_failure",
                "queue": "queue_failure",
                "nurse_assignment": "queue_assignment_failure",
                "render_config": "render_config_failure",
                "api_latency": "high_latency",
                "repeated_errors": "repeated_errors",
            }.get(service, "backend_failure")
            create_incident(
                db,
                incident_type=incident_type,
                severity="critical" if check_status == "critical" else "warning",
                message=f"{service} status is {check_status}. Clinician review required for affected AI-supported triage workflows.",
                related_service=service,
            )
    status["incidents"] = list_incidents(db, include_resolved=False)
    status["active_incidents"] = len(status["incidents"])
    return status


def attempt_recovery(db: Session, incident_id: int | None = None) -> dict[str, Any]:
    query = db.query(PlatformIncident).filter(PlatformIncident.status == "open")
    if incident_id is not None:
        query = query.filter(PlatformIncident.id == incident_id)
    incidents = query.order_by(PlatformIncident.created_at.desc()).all()
    results = []
    for incident in incidents:
        incident.recovery_attempted = True
        incident.recovery_status = "recovering"
        db.commit()
        recovery = safe_recovery_for_incident(incident.incident_type, db=db)
        incident.recovery_action = recovery.get("action")
        incident.recovery_status = recovery.get("status", "manual_required")
        if incident.recovery_status == "recovered":
            incident.status = "resolved"
            incident.resolved_at = datetime.utcnow()
            incident.resolved_by = "self_healing"
        db.commit()
        results.append({"incident_id": incident.id, **recovery})
    return {
        "status": "completed",
        "results": results,
        "message": "Auto-recovery attempted. Unsafe actions remain manual_required.",
    }


def resolve_self_healing_incident(db: Session, incident_id: int, resolved_by: str) -> dict[str, Any] | None:
    return resolve_incident(db, incident_id, resolved_by)

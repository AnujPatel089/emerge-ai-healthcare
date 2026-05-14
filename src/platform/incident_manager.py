from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.models import PlatformIncident
from src.platform.reliability_monitor import app_logger


VALID_INCIDENT_TYPES = {
    "backend_error",
    "database_error",
    "model_error",
    "prediction_failure",
    "high_latency",
    "drift_warning",
    "override_rate_warning",
    "queue_assignment_failure",
}
VALID_SEVERITIES = {"healthy", "warning", "critical"}


def serialize_incident(row: PlatformIncident) -> dict[str, Any]:
    return {
        "id": row.id,
        "incident_type": row.incident_type,
        "severity": row.severity,
        "message": row.message,
        "related_service": row.related_service,
        "status": row.status,
        "service": row.service,
        "detected_at": row.detected_at.isoformat() if row.detected_at else None,
        "recovery_attempted": bool(row.recovery_attempted),
        "recovery_action": row.recovery_action,
        "recovery_status": row.recovery_status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
        "resolved_by": row.resolved_by,
    }


def create_incident(
    db: Session,
    *,
    incident_type: str,
    severity: str,
    message: str,
    related_service: str | None = None,
    service: str | None = None,
) -> dict[str, Any]:
    if incident_type not in VALID_INCIDENT_TYPES:
        incident_type = "backend_error"
    if severity not in VALID_SEVERITIES:
        severity = "warning"

    existing = (
        db.query(PlatformIncident)
        .filter(PlatformIncident.incident_type == incident_type)
        .filter(PlatformIncident.related_service == related_service)
        .filter(PlatformIncident.status == "open")
        .first()
    )
    if existing:
        existing.severity = severity
        existing.message = message
        db.commit()
        db.refresh(existing)
        return serialize_incident(existing)

    row = PlatformIncident(
        incident_type=incident_type,
        severity=severity,
        message=message,
        related_service=related_service,
        service=service or related_service,
        status="open",
        recovery_status="detected",
        recovery_attempted=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    app_logger().warning("Incident created: %s %s", incident_type, message)
    return serialize_incident(row)


def list_incidents(db: Session, include_resolved: bool = False, limit: int = 100) -> list[dict[str, Any]]:
    query = db.query(PlatformIncident)
    if not include_resolved:
        query = query.filter(PlatformIncident.status == "open")
    rows = query.order_by(PlatformIncident.created_at.desc()).limit(limit).all()
    return [serialize_incident(row) for row in rows]


def resolve_incident(db: Session, incident_id: int, resolved_by: str) -> dict[str, Any] | None:
    row = db.query(PlatformIncident).filter(PlatformIncident.id == incident_id).first()
    if not row:
        return None
    row.status = "resolved"
    row.resolved_by = resolved_by
    row.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return serialize_incident(row)

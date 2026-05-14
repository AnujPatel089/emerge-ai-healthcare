from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.models import PlatformAlert


def serialize_alert(alert: PlatformAlert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "status": alert.status,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
        "resolved_by": alert.resolved_by,
    }


def create_alert(db: Session, alert_type: str, severity: str, title: str, message: str) -> dict[str, Any]:
    row = PlatformAlert(alert_type=alert_type, severity=severity, title=title, message=message, status="open")
    db.add(row)
    db.commit()
    db.refresh(row)
    return serialize_alert(row)


def list_alerts(db: Session, include_resolved: bool = False) -> list[dict[str, Any]]:
    query = db.query(PlatformAlert)
    if not include_resolved:
        query = query.filter(PlatformAlert.status == "open")
    return [serialize_alert(row) for row in query.order_by(PlatformAlert.created_at.desc()).limit(100).all()]


def resolve_alert(db: Session, alert_id: int, resolved_by: str) -> dict[str, Any] | None:
    row = db.query(PlatformAlert).filter(PlatformAlert.id == alert_id).first()
    if not row:
        return None
    row.status = "resolved"
    row.resolved_by = resolved_by
    row.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return serialize_alert(row)

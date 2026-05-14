import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.database import get_db
from src.models import PatientTimelineEvent


class TimelineEventInput(BaseModel):
    patient_id: str
    event_type: str
    event_title: str
    event_description: str | None = None
    metadata_json: dict | None = None


def _serialize(row):
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "event_type": row.event_type,
        "event_title": row.event_title,
        "event_description": row.event_description,
        "actor_role": row.actor_role,
        "actor_name": row.actor_name,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "metadata_json": json.loads(row.metadata_json) if row.metadata_json else {},
    }


def build_router(require_role):
    router = APIRouter(prefix="/api/patient-timeline", tags=["Patient Timeline"])

    @router.get("/{patient_id}")
    def get_timeline(patient_id: str, current_user=Depends(require_role(["doctor", "admin", "super_admin", "nurse"])), db=Depends(get_db)):
        rows = db.query(PatientTimelineEvent).filter(PatientTimelineEvent.patient_id == patient_id).order_by(PatientTimelineEvent.created_at.asc()).all()
        return {"patient_id": patient_id, "events": [_serialize(row) for row in rows]}

    @router.post("/event")
    def add_event(payload: TimelineEventInput, current_user=Depends(require_role(["doctor", "admin", "super_admin", "nurse"])), db=Depends(get_db)):
        row = PatientTimelineEvent(
            patient_id=payload.patient_id,
            event_type=payload.event_type,
            event_title=payload.event_title,
            event_description=payload.event_description,
            actor_role=current_user["role"],
            actor_name=current_user["username"],
            metadata_json=json.dumps(payload.metadata_json or {}),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _serialize(row)

    return router

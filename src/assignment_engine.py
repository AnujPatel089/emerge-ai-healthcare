"""Patient-to-nurse assignment engine for realistic ER workflow."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func

from src.models import AppUser, EmergencyQueue, Nurse, NurseAssignment, PredictionLog
from src.queue_manager import ensure_emergency_queue_entry, queue_priority_from_prediction


ACTIVE_ASSIGNMENT_STATUSES = {"Assigned", "In Triage", "Doctor Review", "Critical"}


def sync_active_nurse_users(db) -> list[Nurse]:
    """Ensure approved nurse users also exist in the operational nurses table."""
    active_users = (
        db.query(AppUser)
        .filter(AppUser.role == "nurse", AppUser.account_status == "active")
        .all()
    )
    synced: list[Nurse] = []
    for user in active_users:
        nurse = db.query(Nurse).filter(Nurse.email == user.email).first()
        if not nurse:
            nurse = Nurse(
                name=user.full_name or user.username,
                email=user.email,
                department="Emergency",
                available_status=True,
                active_patient_count=0,
                experience_level="normal",
            )
            db.add(nurse)
            db.flush()
        else:
            nurse.name = user.full_name or nurse.name
            nurse.available_status = user.account_status == "active" and (nurse.active_patient_count or 0) < 7
        synced.append(nurse)
    return synced


def nurse_active_count(db, nurse_id: int) -> int:
    return (
        db.query(func.count(NurseAssignment.id))
        .filter(
            NurseAssignment.nurse_id == nurse_id,
            NurseAssignment.status.in_(ACTIVE_ASSIGNMENT_STATUSES),
        )
        .scalar()
        or 0
    )


def refresh_nurse_workload(db, nurse: Nurse) -> int:
    count = nurse_active_count(db, nurse.id)
    nurse.active_patient_count = count
    nurse.available_status = count < 5
    return count


def select_best_nurse(db, esi_level: int) -> Nurse | None:
    sync_active_nurse_users(db)
    nurses = db.query(Nurse).filter(Nurse.available_status == True).all()
    if not nurses:
        nurses = db.query(Nurse).all()
    if not nurses:
        return None

    for nurse in nurses:
        refresh_nurse_workload(db, nurse)

    if esi_level in {1, 2}:
        experienced = [
            nurse for nurse in nurses
            if str(nurse.experience_level or "").lower() in {"experienced", "senior", "critical"}
        ]
        if experienced:
            nurses = experienced

    return sorted(nurses, key=lambda nurse: (nurse.active_patient_count or 0, nurse.name.lower()))[0]


def assign_nurse_to_prediction(
    db,
    prediction: PredictionLog,
    nurse: Nurse,
    assigned_by: str,
    assignment_type: str = "manual",
    status: str | None = None,
    notes: str | None = None,
) -> NurseAssignment:
    queue_entry = ensure_emergency_queue_entry(db, prediction)
    priority = queue_priority_from_prediction(prediction)
    assignment_status = status or "Assigned"

    existing_active = (
        db.query(NurseAssignment)
        .filter(
            NurseAssignment.prediction_id == prediction.id,
            NurseAssignment.status.in_(ACTIVE_ASSIGNMENT_STATUSES),
        )
        .all()
    )
    for assignment in existing_active:
        if assignment.nurse_id != nurse.id:
            assignment.status = "Reassigned"

    assignment = NurseAssignment(
        prediction_id=prediction.id,
        nurse_id=nurse.id,
        status=assignment_status,
        notes=notes,
        assigned_by=assigned_by,
        assignment_type=assignment_type,
        priority_level=priority["priority"],
    )
    db.add(assignment)

    queue_entry.assigned_nurse_id = nurse.id
    queue_entry.assigned_nurse_name = nurse.name
    queue_entry.assignment_status = "assigned"
    queue_entry.assigned_at = datetime.utcnow()
    queue_entry.updated_at = datetime.utcnow()

    prediction.assigned_nurse_id = nurse.id
    prediction.assigned_nurse_name = nurse.name
    prediction.assignment_status = "assigned"
    prediction.assigned_at = queue_entry.assigned_at

    refresh_nurse_workload(db, nurse)
    nurse.active_patient_count = (nurse.active_patient_count or 0) + 1
    nurse.available_status = nurse.active_patient_count < 5
    return assignment


def auto_assign_nurse(db, prediction: PredictionLog, assigned_by: str = "system") -> NurseAssignment | None:
    priority = queue_priority_from_prediction(prediction)
    nurse = select_best_nurse(db, priority["esi_level"])
    if not nurse:
        ensure_emergency_queue_entry(db, prediction)
        return None
    return assign_nurse_to_prediction(
        db,
        prediction,
        nurse,
        assigned_by=assigned_by,
        assignment_type="auto",
        notes="Automatically assigned by least-busy ER assignment engine.",
    )

"""Assignment API routes for patient-to-nurse ER workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.assignment_engine import auto_assign_nurse, assign_nurse_to_prediction, refresh_nurse_workload, sync_active_nurse_users
from src.auth import get_current_user
from src.database import get_db
from src.models import AppUser, AuditLog, EmergencyQueue, Nurse, NurseAssignment, PredictionLog
from src.queue_manager import ensure_emergency_queue_entry, ordered_queue_query, queue_priority_from_prediction


router = APIRouter()


class AssignNurseInput(BaseModel):
    prediction_id: int
    nurse_id: int
    notes: Optional[str] = None


class AutoAssignInput(BaseModel):
    prediction_id: int


class UpdateAssignmentStatusInput(BaseModel):
    status: str
    assignment_id: Optional[int] = None
    prediction_id: Optional[int] = None


def require_role(allowed_roles: list[str]):
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")
        return current_user

    return role_checker


def _audit(db, current_user: dict, action: str, entity_id: int | str | None, details: str) -> None:
    db.add(
        AuditLog(
            username=current_user.get("username", "system"),
            role=current_user.get("role"),
            action=action,
            entity_type="assignment",
            entity_id=str(entity_id) if entity_id is not None else None,
            details=details,
        )
    )


def _validate_assignable_nurse(db, nurse: Nurse) -> None:
    app_user = db.query(AppUser).filter(AppUser.email == nurse.email).first()
    if app_user and (app_user.role != "nurse" or app_user.account_status != "active"):
        raise HTTPException(status_code=400, detail="Selected nurse account is not active.")
    active_count = refresh_nurse_workload(db, nurse)
    if nurse.available_status is False and active_count >= 7:
        raise HTTPException(status_code=400, detail="Selected nurse is at critical workload.")


def _validate_patient_can_be_assigned(prediction: PredictionLog) -> None:
    if str(getattr(prediction, "assignment_status", "") or "").lower() == "completed":
        raise HTTPException(status_code=400, detail="Completed patients cannot be reassigned.")


def _serialize_nurse(nurse: Nurse) -> dict:
    return {
        "id": nurse.id,
        "name": nurse.name,
        "email": nurse.email,
        "department": nurse.department,
        "available_status": nurse.available_status,
        "active_patient_count": getattr(nurse, "active_patient_count", 0) or 0,
        "experience_level": getattr(nurse, "experience_level", "normal") or "normal",
    }


def _serialize_queue(entry: EmergencyQueue) -> dict:
    prediction = entry.prediction
    priority = queue_priority_from_prediction(prediction) if prediction else {}
    return {
        "id": entry.id,
        "patient_id": entry.prediction_id,
        "prediction_id": entry.prediction_id,
        "patient_name": f"Patient #{entry.prediction_id}",
        "esi_level": priority.get("esi_level"),
        "esi_severity": entry.esi_severity,
        "icu_risk": bool(prediction and prediction.triage_vital_o2 is not None and prediction.triage_vital_o2 < 90),
        "priority": entry.priority,
        "priority_level": entry.priority,
        "assignment_status": entry.assignment_status or "waiting",
        "assigned_nurse_id": entry.assigned_nurse_id,
        "assigned_nurse_name": entry.assigned_nurse_name,
        "assigned_time": str(entry.assigned_at) if entry.assigned_at else None,
        "assigned_at": str(entry.assigned_at) if entry.assigned_at else None,
        "arrival_time": str(entry.arrival_time) if entry.arrival_time else None,
        "created_at": str(entry.arrival_time) if entry.arrival_time else None,
        "estimated_wait_time": entry.estimated_wait_time,
        "wait_time": entry.estimated_wait_time,
        "critical_status": entry.critical_status,
        "readmission_risk": None,
        "uploaded_report": False,
    }


def _serialize_assignment(assignment: NurseAssignment) -> dict:
    prediction = assignment.prediction
    priority = queue_priority_from_prediction(prediction) if prediction else {}
    return {
        "id": assignment.id,
        "assignment_id": assignment.id,
        "patient_id": assignment.prediction_id,
        "prediction_id": assignment.prediction_id,
        "patient_name": f"Patient #{assignment.prediction_id}",
        "nurse_id": assignment.nurse_id,
        "nurse": _serialize_nurse(assignment.nurse) if assignment.nurse else None,
        "esi_level": priority.get("esi_level"),
        "icu_risk": bool(prediction and prediction.triage_vital_o2 is not None and prediction.triage_vital_o2 < 90),
        "priority_level": getattr(assignment, "priority_level", None) or priority.get("priority"),
        "assignment_status": assignment.status,
        "status": assignment.status,
        "assigned_at": str(assignment.assigned_at) if assignment.assigned_at else None,
        "assigned_by": getattr(assignment, "assigned_by", None),
        "assignment_type": getattr(assignment, "assignment_type", None),
        "notes": assignment.notes,
    }


def _get_prediction_and_nurse(db, prediction_id: int, nurse_id: int) -> tuple[PredictionLog, Nurse]:
    prediction = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Patient/prediction record not found.")
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found.")
    return prediction, nurse


@router.get("/waiting-queue")
def get_waiting_queue(
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    predictions = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(100).all()
    for prediction in predictions:
        ensure_emergency_queue_entry(db, prediction)
    db.commit()

    entries = ordered_queue_query(db).filter(EmergencyQueue.assignment_status != "completed").all()
    if current_user["role"] == "nurse":
        nurse = (
            db.query(Nurse)
            .filter((Nurse.email == current_user["username"]) | (Nurse.name == current_user["username"]))
            .first()
        )
        if not nurse:
            nurse = db.query(Nurse).filter(Nurse.name.ilike(f"%{current_user['username']}%")).first()
        entries = [entry for entry in entries if nurse and entry.assigned_nurse_id == nurse.id]
    rows = [_serialize_queue(entry) for entry in entries]
    return {
        "status": "success",
        "waiting_queue": rows,
        "queue": rows,
        "patients": rows,
        "waiting_patients": [row for row in rows if row.get("assignment_status") == "waiting"],
    }


@router.get("/nurse-workload")
def get_nurse_workload(
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    sync_active_nurse_users(db)
    nurses = db.query(Nurse).order_by(Nurse.name.asc()).all()
    rows = []
    for nurse in nurses:
        active_count = refresh_nurse_workload(db, nurse)
        assigned = db.query(NurseAssignment).filter(NurseAssignment.nurse_id == nurse.id).count()
        in_triage = db.query(NurseAssignment).filter(NurseAssignment.nurse_id == nurse.id, NurseAssignment.status == "In Triage").count()
        doctor_review = db.query(NurseAssignment).filter(NurseAssignment.nurse_id == nurse.id, NurseAssignment.status == "Doctor Review").count()
        critical = db.query(NurseAssignment).filter(NurseAssignment.nurse_id == nurse.id, NurseAssignment.priority_level.in_(["Critical", "High"])).count()
        availability = "Available" if active_count <= 3 else "Busy" if active_count <= 6 else "Critical Load"
        rows.append({
            **_serialize_nurse(nurse),
            "role": "nurse",
            "account_status": "active" if nurse.available_status else "busy",
            "current_assigned_patient_count": active_count,
            "assigned_patients": assigned,
            "in_triage_patients": in_triage,
            "doctor_review_patients": doctor_review,
            "critical_patients": critical,
            "availability_status": availability,
        })
    db.commit()
    return {"status": "success", "nurses": rows, "workload": rows}


@router.post("/auto-assign")
def auto_assign_patient(
    data: AutoAssignInput,
    current_user: dict = Depends(require_role(["doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    prediction = db.query(PredictionLog).filter(PredictionLog.id == data.prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Patient/prediction record not found.")
    assignment = auto_assign_nurse(db, prediction, assigned_by=current_user["username"])
    _audit(db, current_user, "auto_assign_patient", prediction.id, "Auto assignment requested for one patient.")
    db.commit()
    if not assignment:
        return {"status": "no_nurse_available", "message": "No active nurses available", "assignment": None}
    db.refresh(assignment)
    return {"status": "success", "assignment": _serialize_assignment(assignment)}


@router.post("/auto-assign-all")
def auto_assign_all_waiting(
    current_user: dict = Depends(require_role(["doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    predictions = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(200).all()
    for prediction in predictions:
        ensure_emergency_queue_entry(db, prediction)
    waiting_entries = (
        ordered_queue_query(db)
        .filter(EmergencyQueue.assignment_status == "waiting")
        .all()
    )
    assignments = []
    skipped = []
    for entry in waiting_entries:
        if not entry.prediction:
            skipped.append(entry.prediction_id)
            continue
        assignment = auto_assign_nurse(db, entry.prediction, assigned_by=current_user["username"])
        if assignment:
            assignments.append(assignment)
        else:
            skipped.append(entry.prediction_id)
    _audit(
        db,
        current_user,
        "auto_assign_all_waiting",
        None,
        f"Assigned {len(assignments)} waiting patients; skipped {len(skipped)}.",
    )
    db.commit()
    return {
        "status": "success",
        "assigned_count": len(assignments),
        "skipped_count": len(skipped),
        "message": "No active nurses available" if waiting_entries and not assignments else "Auto assignment completed",
        "assignments": [_serialize_assignment(item) for item in assignments],
        "skipped_prediction_ids": skipped,
    }


@router.post("/assign-nurse")
def assign_nurse(
    data: AssignNurseInput,
    current_user: dict = Depends(require_role(["doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    prediction, nurse = _get_prediction_and_nurse(db, data.prediction_id, data.nurse_id)
    _validate_patient_can_be_assigned(prediction)
    _validate_assignable_nurse(db, nurse)
    assignment = assign_nurse_to_prediction(
        db,
        prediction,
        nurse,
        assigned_by=current_user["username"],
        assignment_type="manual",
        notes=data.notes or "Manual assignment from assignment dashboard.",
    )
    _audit(db, current_user, "assign_nurse", prediction.id, f"Assigned nurse {nurse.id} to prediction {prediction.id}.")
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": _serialize_assignment(assignment)}


@router.post("/reassign-nurse")
def reassign_nurse(
    data: AssignNurseInput,
    current_user: dict = Depends(require_role(["doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    prediction, nurse = _get_prediction_and_nurse(db, data.prediction_id, data.nurse_id)
    _validate_patient_can_be_assigned(prediction)
    _validate_assignable_nurse(db, nurse)
    assignment = assign_nurse_to_prediction(
        db,
        prediction,
        nurse,
        assigned_by=current_user["username"],
        assignment_type="reassign",
        notes=data.notes or "Manual reassignment from assignment dashboard.",
    )
    _audit(db, current_user, "reassign_nurse", prediction.id, f"Reassigned nurse {nurse.id} to prediction {prediction.id}.")
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": _serialize_assignment(assignment)}


@router.post("/update-status")
@router.put("/update-status")
def update_assignment_status(
    data: UpdateAssignmentStatusInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    allowed = {"waiting", "assigned", "in_triage", "doctor_review", "completed", "critical"}
    normalized = data.status.strip().lower()
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid status. Use one of: {sorted(allowed)}")

    query = db.query(NurseAssignment)
    if data.assignment_id:
        query = query.filter(NurseAssignment.id == data.assignment_id)
    elif data.prediction_id:
        query = query.filter(NurseAssignment.prediction_id == data.prediction_id).order_by(NurseAssignment.assigned_at.desc())
    else:
        raise HTTPException(status_code=400, detail="assignment_id or prediction_id is required.")

    assignment = query.first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")

    label = {
        "waiting": "Waiting",
        "assigned": "Assigned",
        "in_triage": "In Triage",
        "doctor_review": "Doctor Review",
        "completed": "Completed",
        "critical": "Critical",
    }[normalized]
    assignment.status = label

    entry = db.query(EmergencyQueue).filter(EmergencyQueue.prediction_id == assignment.prediction_id).first()
    now = datetime.utcnow()
    if entry:
        entry.assignment_status = normalized
        if normalized == "in_triage":
            entry.triage_started_at = entry.triage_started_at or now
        elif normalized == "doctor_review":
            entry.doctor_review_at = entry.doctor_review_at or now
        elif normalized == "completed":
            entry.completed_at = entry.completed_at or now
        entry.updated_at = now
    if assignment.prediction:
        assignment.prediction.assignment_status = normalized
        if normalized == "in_triage":
            assignment.prediction.triage_started_at = assignment.prediction.triage_started_at or now
        elif normalized == "doctor_review":
            assignment.prediction.doctor_review_at = assignment.prediction.doctor_review_at or now
        elif normalized == "completed":
            assignment.prediction.completed_at = assignment.prediction.completed_at or now
    if assignment.nurse:
        refresh_nurse_workload(db, assignment.nurse)

    _audit(db, current_user, "update_assignment_status", assignment.prediction_id, f"Changed assignment status to {normalized}.")
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": _serialize_assignment(assignment)}


@router.get("/my-patients")
def get_my_patients(
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    query = db.query(NurseAssignment)
    if current_user["role"] == "nurse":
        sync_active_nurse_users(db)
        nurse = (
            db.query(Nurse)
            .filter((Nurse.email == current_user["username"]) | (Nurse.name == current_user["username"]))
            .first()
        )
        if not nurse:
            nurse = db.query(Nurse).filter(Nurse.name.ilike(f"%{current_user['username']}%")).first()
        if not nurse:
            return {"status": "success", "patients": []}
        query = query.filter(NurseAssignment.nurse_id == nurse.id)
    assignments = query.order_by(NurseAssignment.assigned_at.desc()).all()
    return {"status": "success", "patients": [_serialize_assignment(item) for item in assignments]}


@router.get("/patient/{patient_id}")
def get_assignment_patient(
    patient_id: int,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    entry = db.query(EmergencyQueue).filter(EmergencyQueue.prediction_id == patient_id).first()
    if not entry:
        prediction = db.query(PredictionLog).filter(PredictionLog.id == patient_id).first()
        if not prediction:
            return {"status": "success", "patient": None}
        entry = ensure_emergency_queue_entry(db, prediction)
        db.commit()
    return {"status": "success", "patient": _serialize_queue(entry)}

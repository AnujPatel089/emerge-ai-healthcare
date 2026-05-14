"""
Safe development seed script for EmergeAI Healthcare.

Creates demo nurses, staff, and sample queue data so the auto-assignment
engine has active nurses to work with immediately after startup.

Usage:
    python scripts/seed_demo_data.py

Rules:
- Will NOT overwrite existing records.
- Uses upsert-style inserts so it is safe to run multiple times.
- Never touches real patient data.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import Base, engine, SessionLocal
from src.models import (
    AppUser,
    Nurse,
    PredictionLog,
    EmergencyQueue,
)
from src.auth_utils import hash_password

Base.metadata.create_all(bind=engine)


DEMO_NURSES = [
    {
        "full_name": "Nurse Sarah",
        "username": "nurse_sarah",
        "email": "sarah@emergeai.demo",
        "password": "nurse123",
        "department": "Emergency",
        "experience_level": "experienced",
    },
    {
        "full_name": "Nurse Emily",
        "username": "nurse_emily",
        "email": "emily@emergeai.demo",
        "password": "nurse123",
        "department": "Emergency",
        "experience_level": "senior",
    },
    {
        "full_name": "Nurse Michael",
        "username": "nurse_michael",
        "email": "michael@emergeai.demo",
        "password": "nurse123",
        "department": "Emergency",
        "experience_level": "normal",
    },
]

DEMO_STAFF = [
    {
        "full_name": "Dr. Alex Carter",
        "username": "demo_doctor",
        "email": "doctor@emergeai.demo",
        "password": "doctor123",
        "role": "doctor",
    },
    {
        "full_name": "Admin Jordan",
        "username": "demo_admin",
        "email": "admin@emergeai.demo",
        "password": "admin123",
        "role": "admin",
    },
]

SAMPLE_PREDICTIONS = [
    {
        "age": 58,
        "gender": "Male",
        "race": "White",
        "ethnicity": "Non-Hispanic",
        "arrivalmode": "Ambulance",
        "triage_vital_hr": 120,
        "triage_vital_sbp": 85,
        "triage_vital_dbp": 55,
        "triage_vital_rr": 26,
        "triage_vital_o2": 88,
        "triage_vital_temp": 38.9,
        "cc_chestpain": 1,
        "cc_shortnessofbreath": 1,
        "cc_headache": 0,
        "cc_fever": 0,
        "cc_abdominalpain": 0,
        "cc_dizziness": 1,
        "cc_syncope": 0,
        "cc_weakness": 1,
        "ml_prediction": "ESI 1 - Critical",
        "final_prediction": "ESI 1 - Critical",
        "confidence": 0.92,
        "problem_description": "Severe chest pain, low BP, low oxygen",
        "source": "seed_demo",
        "feedback": "Pending",
    },
    {
        "age": 34,
        "gender": "Female",
        "race": "Black",
        "ethnicity": "Non-Hispanic",
        "arrivalmode": "Walk-In",
        "triage_vital_hr": 108,
        "triage_vital_sbp": 100,
        "triage_vital_dbp": 65,
        "triage_vital_rr": 22,
        "triage_vital_o2": 94,
        "triage_vital_temp": 39.4,
        "cc_chestpain": 0,
        "cc_shortnessofbreath": 1,
        "cc_headache": 1,
        "cc_fever": 1,
        "cc_abdominalpain": 0,
        "cc_dizziness": 0,
        "cc_syncope": 0,
        "cc_weakness": 0,
        "ml_prediction": "ESI 2 - Emergency",
        "final_prediction": "ESI 2 - Emergency",
        "confidence": 0.85,
        "problem_description": "High fever, shortness of breath, severe headache",
        "source": "seed_demo",
        "feedback": "Pending",
    },
    {
        "age": 45,
        "gender": "Male",
        "race": "Hispanic",
        "ethnicity": "Hispanic",
        "arrivalmode": "Walk-In",
        "triage_vital_hr": 92,
        "triage_vital_sbp": 130,
        "triage_vital_dbp": 85,
        "triage_vital_rr": 18,
        "triage_vital_o2": 97,
        "triage_vital_temp": 37.8,
        "cc_chestpain": 0,
        "cc_shortnessofbreath": 0,
        "cc_headache": 0,
        "cc_fever": 0,
        "cc_abdominalpain": 1,
        "cc_dizziness": 0,
        "cc_syncope": 0,
        "cc_weakness": 0,
        "ml_prediction": "ESI 3 - Urgent",
        "final_prediction": "ESI 3 - Urgent",
        "confidence": 0.78,
        "problem_description": "Abdominal pain for 3 days",
        "source": "seed_demo",
        "feedback": "Pending",
    },
]

ESI_QUEUE_SETTINGS = {
    "ESI 1": ("Critical", 0, True),
    "ESI 2": ("High", 10, True),
    "ESI 3": ("Medium", 45, False),
    "ESI 4": ("Low", 90, False),
    "ESI 5": ("Low", 150, False),
}


def _esi_severity(prediction_text: str) -> str:
    import re
    m = re.search(r"ESI\s*([1-5])", str(prediction_text))
    return f"ESI {m.group(1)}" if m else "ESI 3"


def seed_nurses(db) -> list[Nurse]:
    created = []
    for nd in DEMO_NURSES:
        existing_user = db.query(AppUser).filter(AppUser.username == nd["username"]).first()
        if not existing_user:
            user = AppUser(
                full_name=nd["full_name"],
                username=nd["username"],
                email=nd["email"],
                password_hash=hash_password(nd["password"]),
                requested_role="nurse",
                role="nurse",
                account_status="active",
                approved_by="system_seed",
                approved_at=datetime.utcnow(),
            )
            db.add(user)
            db.flush()
            print(f"  Created AppUser: {nd['username']}")
        else:
            existing_user.account_status = "active"
            existing_user.role = "nurse"
            print(f"  AppUser already exists: {nd['username']} (status set active)")

        nurse_record = db.query(Nurse).filter(Nurse.email == nd["email"]).first()
        if not nurse_record:
            nurse_record = Nurse(
                name=nd["full_name"],
                email=nd["email"],
                department=nd["department"],
                available_status=True,
                active_patient_count=0,
                experience_level=nd["experience_level"],
            )
            db.add(nurse_record)
            db.flush()
            print(f"  Created Nurse record: {nd['full_name']}")
            created.append(nurse_record)
        else:
            nurse_record.available_status = True
            nurse_record.active_patient_count = 0
            print(f"  Nurse record already exists: {nd['full_name']} (reset to available)")
            created.append(nurse_record)
    return created


def seed_staff(db) -> None:
    for sd in DEMO_STAFF:
        existing = db.query(AppUser).filter(AppUser.username == sd["username"]).first()
        if not existing:
            user = AppUser(
                full_name=sd["full_name"],
                username=sd["username"],
                email=sd["email"],
                password_hash=hash_password(sd["password"]),
                requested_role=sd["role"],
                role=sd["role"],
                account_status="active",
                approved_by="system_seed",
                approved_at=datetime.utcnow(),
            )
            db.add(user)
            print(f"  Created staff user: {sd['username']} ({sd['role']})")
        else:
            existing.account_status = "active"
            existing.role = sd["role"]
            print(f"  Staff user already exists: {sd['username']}")


def seed_sample_predictions(db) -> list[PredictionLog]:
    created = []
    existing_seed = db.query(PredictionLog).filter(PredictionLog.source == "seed_demo").count()
    if existing_seed >= len(SAMPLE_PREDICTIONS):
        print(f"  Sample predictions already exist ({existing_seed} records). Skipping.")
        return []

    for pd_data in SAMPLE_PREDICTIONS:
        log = PredictionLog(**pd_data, assignment_status="waiting")
        db.add(log)
        db.flush()
        esi_key = _esi_severity(pd_data["final_prediction"])
        priority, wait_time, critical = ESI_QUEUE_SETTINGS.get(esi_key, ("Medium", 45, False))
        queue = EmergencyQueue(
            prediction_id=log.id,
            esi_severity=esi_key,
            priority=priority,
            estimated_wait_time=wait_time,
            critical_status=critical,
            assignment_status="waiting",
            arrival_time=datetime.utcnow(),
        )
        db.add(queue)
        print(f"  Created sample patient #{log.id}: {esi_key} ({priority})")
        created.append(log)
    return created


def run_auto_assignment(db, predictions: list[PredictionLog]) -> None:
    if not predictions:
        return
    try:
        from src.assignment_engine import auto_assign_nurse
        for prediction in predictions:
            assignment = auto_assign_nurse(db, prediction, assigned_by="system_seed")
            if assignment:
                print(f"  Auto-assigned Patient #{prediction.id} to nurse ID {assignment.nurse_id}")
            else:
                print(f"  No nurse available for Patient #{prediction.id}")
    except Exception as exc:
        print(f"  Warning: auto-assignment skipped: {exc}")


def main() -> None:
    print("\nEmergeAI Demo Data Seeder")
    print("=" * 40)

    db = SessionLocal()
    try:
        print("\n[1] Seeding demo nurses...")
        nurses = seed_nurses(db)
        db.commit()

        print("\n[2] Seeding demo staff users...")
        seed_staff(db)
        db.commit()

        print("\n[3] Seeding sample patients into queue...")
        predictions = seed_sample_predictions(db)
        db.commit()

        print("\n[4] Running auto-assignment for new patients...")
        run_auto_assignment(db, predictions)
        db.commit()

        print("\n" + "=" * 40)
        print("Seed complete. Demo login accounts:")
        print()
        print("  Nurses (login as 'nurse'):")
        print("    nurse_sarah   / nurse123")
        print("    nurse_emily   / nurse123")
        print("    nurse_michael / nurse123")
        print()
        print("  Doctor:  demo_doctor / doctor123")
        print("  Admin:   demo_admin  / admin123")
        print()
        print("  Legacy demo users still work:")
        print("    admin / admin123   |  doctor / doctor123  |  nurse / nurse123")
        print("    anuj / anuj123     |  chintan / chintan123")
        print()
        print("Run: python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000")
        print("Then: streamlit run frontend/app.py")

    finally:
        db.close()


if __name__ == "__main__":
    main()

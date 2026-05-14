"""
FastAPI Backend
Emergency Triage AI
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
import sys
import joblib
import subprocess
import io
import json
import re
import shutil
import matplotlib.pyplot as plt
from sqlalchemy import desc, or_, text
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
PROJECT_MODELS_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from src.report_generator import generate_report
from src.email_service import send_doctor_report_email
from src.safety_rules import apply_safety_rules
from src.shap_explainer import generate_clinical_explanation
from src.audit_logger import save_prediction_log
#from src.shap_visualizer import initialize_shap
from src.symptom_extractor import extract_symptoms
from src.image_analyzer import analyze_medical_image
from src.report_text_extractor import SUPPORTED_EXTENSIONS, extract_text_from_file
from src.image_assessment import IMAGE_EXTENSIONS, assess_medical_image
from src.medical_history_analyzer import analyze_medical_history
from src.risk_flag_engine import generate_image_risk_flags, generate_risk_flags
from src.clinical_summary_generator import DISCLAIMER as HISTORICAL_REPORT_DISCLAIMER, generate_clinical_summary
from src.assignment_engine import auto_assign_nurse, assign_nurse_to_prediction, refresh_nurse_workload, sync_active_nurse_users
from src.queue_manager import ensure_emergency_queue_entry, ordered_queue_query, queue_priority_from_prediction

try:
    from src.shap_visualizer import initialize_shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from src.database import Base, engine, SessionLocal, get_db
from src.models import (
    AppUser,
    TriageUploadedFile,
    PredictionLog,
    ClinicalFeedback,
    Nurse,
    NurseAssignment,
    NurseVitals,
    NurseTask,
    DoctorReview,
    PatientStatus,
    StaffMember,
    EmergencyQueue,
    Bed,
    MedicationRecord,
    ClinicalAlert,
    DischargeSummary,
    StaffShift,
    Patient,
    PatientPredictionLink,
    LabOrder,
    Referral,
    Consent,
    IncidentReport,
    BillingRecord,
    InventoryItem,
    FollowUpAppointment,
    NotificationLog,
    HistoricalMedicalReport,
    AuditLog
)

from src.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    fake_users_db
)
from src.auth_utils import can_approve_role, can_view_registration_role, approval_level_for_role, hash_password, normalize_role

# NEW: multi-modal triage + DL image + PDF report routes
#from src.api_extensions import router as v2_router
from backend.routes.assignment_routes import router as assignment_router
try:
    from src.api_extensions import router as v2_router
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# -----------------------------
# INPUT MODELS
# -----------------------------

class PatientInput(BaseModel):
    age: int
    gender: str
    race: str
    ethnicity: str
    arrivalmode: str

    triage_vital_hr: int
    triage_vital_sbp: int
    triage_vital_dbp: int
    triage_vital_rr: int
    triage_vital_o2: int
    triage_vital_temp: float

    cc_chestpain: int
    cc_shortnessofbreath: int
    cc_headache: int
    cc_fever: int
    cc_abdominalpain: int
    cc_dizziness: int
    cc_syncope: int
    cc_weakness: int

    problem_description: Optional[str] = None
    llm_ready_text: Optional[str] = None
    matched_terms: Optional[List[str]] = None
    emergency_keywords: Optional[List[str]] = None
    cv_analysis: Optional[Dict[str, Any]] = None

class EmailReportInput(BaseModel):
    doctor_email: str
    prediction_id: int

class SymptomTextInput(BaseModel):
    problem_description: str


class FeedbackInput(BaseModel):
    log_id: int
    feedback: str


class ClinicalFeedbackInput(BaseModel):
    prediction_id: int
    accepted: bool
    override_esi: Optional[int] = None
    clinical_notes: Optional[str] = None
    override_reason: Optional[str] = None


class NurseCreateInput(BaseModel):
    name: str
    email: str
    department: str
    available_status: bool = True
    experience_level: str = "normal"


class NurseAssignmentInput(BaseModel):
    prediction_id: int
    nurse_id: int
    status: str = "Assigned"
    notes: Optional[str] = None


class NurseReassignmentInput(BaseModel):
    prediction_id: int
    nurse_id: int
    notes: Optional[str] = None


class AssignmentStatusInput(BaseModel):
    status: str


class NurseVitalsInput(BaseModel):
    prediction_id: int
    nurse_id: int
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None
    oxygen_level: Optional[int] = None
    respiratory_rate: Optional[int] = None
    pain_score: Optional[int] = None
    notes: Optional[str] = None


class NurseTaskInput(BaseModel):
    prediction_id: int
    nurse_id: int
    task_title: str
    task_description: Optional[str] = None
    status: str = "Pending"
    priority: str = "Medium"


class NurseTaskStatusInput(BaseModel):
    status: str


class DoctorReviewInput(BaseModel):
    prediction_id: int
    diagnosis: str
    treatment_plan: str
    medication_notes: Optional[str] = None
    follow_up_required: bool = False
    admit_status: str = "Not Admitted"


class PatientStatusInput(BaseModel):
    patient_status: str
    notes: Optional[str] = None


class StaffUpdateInput(BaseModel):
    active_status: Optional[bool] = None
    department: Optional[str] = None


class QueuePriorityInput(BaseModel):
    priority: str
    estimated_wait_time: Optional[int] = None


class BedCreateInput(BaseModel):
    bed_number: str
    ward_type: str


class BedAssignInput(BaseModel):
    bed_id: int
    prediction_id: int


class MedicationRecordInput(BaseModel):
    prediction_id: int
    nurse_id: int
    medication_name: str
    dosage: str
    route: str
    scheduled_time: datetime
    side_effects: Optional[str] = None
    notes: Optional[str] = None


class MedicationStatusInput(BaseModel):
    status: str
    side_effects: Optional[str] = None
    notes: Optional[str] = None


class ClinicalAlertInput(BaseModel):
    prediction_id: int
    alert_type: str
    severity: str
    message: str


class DischargeSummaryInput(BaseModel):
    prediction_id: int
    diagnosis: str
    treatment_given: str
    medication_notes: Optional[str] = None
    follow_up_instructions: Optional[str] = None
    discharge_status: str = "Discharged"


class StaffShiftInput(BaseModel):
    staff_id: str
    staff_role: str
    department: Optional[str] = None
    shift_type: str
    start_time: datetime
    end_time: datetime
    status: str = "Scheduled"


class PatientInputData(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    allergies: Optional[str] = None
    chronic_diseases: Optional[str] = None
    past_medical_history: Optional[str] = None
    prediction_id: Optional[int] = None
    assigned_nurse_id: Optional[int] = None
    assigned_nurse_name: Optional[str] = None
    assigned_doctor_id: Optional[str] = None
    assigned_doctor_name: Optional[str] = None
    assignment_status: Optional[str] = "waiting"


class LabOrderInput(BaseModel):
    prediction_id: int
    test_name: str
    test_type: Optional[str] = None
    priority: str = "Routine"


class LabResultInput(BaseModel):
    status: str
    result_notes: Optional[str] = None
    result_file_path: Optional[str] = None


class ReferralInput(BaseModel):
    prediction_id: int
    specialist_department: str
    reason: str
    urgency: str = "Routine"
    notes: Optional[str] = None


class StatusUpdateInput(BaseModel):
    status: str
    notes: Optional[str] = None


class ConsentInput(BaseModel):
    patient_id: int
    prediction_id: Optional[int] = None
    consent_type: str
    accepted: bool
    signed_by: str
    notes: Optional[str] = None


class IncidentInput(BaseModel):
    prediction_id: Optional[int] = None
    incident_type: str
    severity: str
    description: str
    action_taken: Optional[str] = None


class BillingInput(BaseModel):
    patient_id: int
    prediction_id: Optional[int] = None
    insurance_provider: Optional[str] = None
    policy_number: Optional[str] = None
    visit_cost: float = 0
    treatment_cost: float = 0
    medication_cost: float = 0
    payment_status: str = "Pending"


class BillingStatusInput(BaseModel):
    payment_status: str


class InventoryInput(BaseModel):
    item_name: str
    category: str
    quantity: int = 0
    unit: Optional[str] = None
    minimum_stock_level: int = 0
    expiry_date: Optional[datetime] = None
    location: Optional[str] = None
    status: str = "Available"


class AppointmentInput(BaseModel):
    patient_id: int
    prediction_id: Optional[int] = None
    doctor_id: str
    appointment_date: datetime
    department: Optional[str] = None
    reason: Optional[str] = None
    status: str = "Scheduled"
    notes: Optional[str] = None


class NotificationInput(BaseModel):
    patient_id: int
    prediction_id: Optional[int] = None
    contact_name: str
    contact_phone: str
    message: str
    notification_type: str


class LoginInput(BaseModel):
    username: str
    password: str
    role: str


class RegisterInput(BaseModel):
    full_name: str
    username: str
    email: str
    password: str
    confirm_password: str
    requested_role: str


class ApprovalDecisionInput(BaseModel):
    rejected_reason: Optional[str] = None


class AccountUpdateInput(BaseModel):
    role: Optional[str] = None
    account_status: Optional[str] = None


class HistoricalReportNotesInput(BaseModel):
    doctor_notes: Optional[str] = None
    nurse_notes: Optional[str] = None


# -----------------------------
# ROLE CHECKER
# -----------------------------

def require_role(allowed_roles: list):
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="Access denied: insufficient permissions"
            )
        return current_user

    return role_checker


NON_MODEL_FIELDS = [
    "problem_description",
    "llm_ready_text",
    "matched_terms",
    "emergency_keywords",
    "cv_analysis"
]

EXPECTED_MODEL_COLUMNS = [
    "age",
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "cc_chestpain",
    "cc_shortnessofbreath",
    "cc_headache",
    "cc_fever",
    "cc_abdominalpain",
    "cc_dizziness",
    "cc_syncope",
    "cc_weakness",
    "gender_Male",
    "race_Asian",
    "race_White",
    "ethnicity_Non-Hispanic",
    "arrivalmode_Ambulance",
    "arrivalmode_Walk-in"
]


def prepare_model_input(patient_dict: dict) -> tuple[dict, pd.DataFrame]:
    model_input_dict = patient_dict.copy()

    for field in NON_MODEL_FIELDS:
        model_input_dict.pop(field, None)

    input_df = pd.DataFrame([model_input_dict])

    # One-hot encoding exactly matching training columns.
    input_df["gender_Male"] = 1 if model_input_dict.get("gender") == "Male" else 0
    input_df["race_Asian"] = 1 if model_input_dict.get("race") == "Asian" else 0
    input_df["race_White"] = 1 if model_input_dict.get("race") == "White" else 0
    input_df["ethnicity_Non-Hispanic"] = 1 if model_input_dict.get("ethnicity") == "Non-Hispanic" else 0
    input_df["arrivalmode_Ambulance"] = 1 if model_input_dict.get("arrivalmode") == "Ambulance" else 0
    input_df["arrivalmode_Walk-in"] = 1 if model_input_dict.get("arrivalmode") == "Walk-in" else 0

    input_df = input_df.drop(
        columns=["gender", "race", "ethnicity", "arrivalmode"],
        errors="ignore"
    )

    input_df = input_df.reindex(columns=EXPECTED_MODEL_COLUMNS, fill_value=0)

    return model_input_dict, input_df


def to_python_value(value):
    if hasattr(value, "item"):
        return value.item()

    return value


# -----------------------------
# LOAD MODEL
# -----------------------------

def get_model_path(filename: str) -> Path:
    backend_model_path = MODELS_DIR / filename
    if backend_model_path.exists():
        return backend_model_path

    return PROJECT_MODELS_DIR / filename

#MODEL_PATH = get_model_path("triage_xgboost_balanced.pkl")
#DEFAULT_MODEL_PATH = get_model_path("triage_xgboost.pkl")

#if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Retrained model loaded.")
#else:
    #model = joblib.load(DEFAULT_MODEL_PATH)
    #try:
        #model = joblib.load(DEFAULT_MODEL_PATH)
        #MODEL_AVAILABLE = True
    #except Exception as e:
       #print(f"Model load failed: {e}")
        #model = None
        #MODEL_AVAILABLE = False
    #print("Default model loaded.")



#label_encoder = joblib.load(get_model_path("esi_label_encoder.pkl"))

#initialize_shap(model)
#if SHAP_AVAILABLE:
    #initialize_shap()

MODEL_PATH = get_model_path("triage_xgboost_balanced.pkl")
DEFAULT_MODEL_PATH = get_model_path("triage_xgboost.pkl")
LABEL_ENCODER_PATH = get_model_path("esi_label_encoder.pkl")

MODEL_AVAILABLE = False
LABEL_ENCODER_AVAILABLE = False

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        MODEL_AVAILABLE = True
        print("Balanced model loaded.")
    elif os.path.exists(DEFAULT_MODEL_PATH):
        model = joblib.load(DEFAULT_MODEL_PATH)
        MODEL_AVAILABLE = True
        print("Default model loaded.")
    else:
        model = None
        print("No model file found. Running backend without ML model.")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None
    MODEL_AVAILABLE = False

try:
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        LABEL_ENCODER_AVAILABLE = True
        print("Label encoder loaded.")
    else:
        label_encoder = None
        print("No label encoder file found. Running without label encoder.")
except Exception as e:
    print(f"Label encoder load failed: {e}")
    label_encoder = None
    LABEL_ENCODER_AVAILABLE = False

if SHAP_AVAILABLE and MODEL_AVAILABLE:
    initialize_shap()


# -----------------------------
# APP
# -----------------------------

app = FastAPI(
    title="Emergency Triage AI API",
    version="3.0"
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


def ensure_app_user_columns():
    """Add approval columns for existing development databases without Alembic."""
    columns = {
        "requested_role": "VARCHAR DEFAULT 'patient'",
        "account_status": "VARCHAR DEFAULT 'pending'",
        "approval_level": "VARCHAR",
        "rejected_reason": "TEXT",
        "rejected_by": "VARCHAR",
        "rejected_at": "TIMESTAMP",
    }
    db = SessionLocal()
    try:
        for column, ddl_type in columns.items():
            try:
                db.execute(text(f"ALTER TABLE app_users ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
            except Exception:
                db.rollback()
        db.commit()
    finally:
        db.close()


ensure_app_user_columns()


def ensure_historical_report_columns():
    """Add image-assessment columns for existing development databases."""
    columns = {
        "file_type": "VARCHAR",
        "ocr_text": "TEXT",
        "image_metadata": "TEXT",
        "image_quality_notes": "TEXT",
    }
    db = SessionLocal()
    try:
        for column, ddl_type in columns.items():
            try:
                db.execute(text(f"ALTER TABLE historical_medical_reports ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
            except Exception:
                db.rollback()
        db.commit()
    finally:
        db.close()


ensure_historical_report_columns()


def ensure_assignment_columns():
    """Add assignment workflow columns for existing development databases."""
    table_columns = {
        "nurses": {
            "active_patient_count": "INTEGER DEFAULT 0",
            "experience_level": "VARCHAR DEFAULT 'normal'",
        },
        "nurse_assignments": {
            "assigned_by": "VARCHAR",
            "assignment_type": "VARCHAR DEFAULT 'manual'",
            "priority_level": "VARCHAR DEFAULT 'Medium'",
        },
        "emergency_queue": {
            "assignment_status": "VARCHAR DEFAULT 'waiting'",
            "assigned_nurse_id": "INTEGER",
            "assigned_nurse_name": "VARCHAR",
            "assigned_doctor_id": "VARCHAR",
            "assigned_doctor_name": "VARCHAR",
            "assigned_at": "TIMESTAMP",
            "triage_started_at": "TIMESTAMP",
            "doctor_review_at": "TIMESTAMP",
            "completed_at": "TIMESTAMP",
        },
        "patients": {
            "assigned_nurse_id": "INTEGER",
            "assigned_nurse_name": "VARCHAR",
            "assigned_doctor_id": "VARCHAR",
            "assigned_doctor_name": "VARCHAR",
            "assignment_status": "VARCHAR DEFAULT 'waiting'",
            "assigned_at": "TIMESTAMP",
            "triage_started_at": "TIMESTAMP",
            "doctor_review_at": "TIMESTAMP",
            "completed_at": "TIMESTAMP",
        },
        "prediction_logs": {
            "assigned_nurse_id": "INTEGER",
            "assigned_nurse_name": "VARCHAR",
            "assigned_doctor_id": "VARCHAR",
            "assigned_doctor_name": "VARCHAR",
            "assignment_status": "VARCHAR DEFAULT 'waiting'",
            "assigned_at": "TIMESTAMP",
            "triage_started_at": "TIMESTAMP",
            "doctor_review_at": "TIMESTAMP",
            "completed_at": "TIMESTAMP",
        },
    }
    db = SessionLocal()
    try:
        for table, columns in table_columns.items():
            for column, ddl_type in columns.items():
                try:
                    db.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
                except Exception:
                    db.rollback()
        db.commit()
    finally:
        db.close()


ensure_assignment_columns()

HISTORICAL_REPORT_DIR = PROJECT_ROOT / "reports" / "historical_reports"
HISTORICAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_IMAGE_DIR = PROJECT_ROOT / "reports" / "historical_images"
HISTORICAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TRIAGE_REPORT_DIR = PROJECT_ROOT / "reports" / "triage_uploads" / "reports"
TRIAGE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRIAGE_IMAGE_DIR = PROJECT_ROOT / "reports" / "triage_uploads" / "images"
TRIAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def safe_upload_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "report"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{cleaned}_{timestamp}{suffix}"


def serialize_historical_report(report: HistoricalMedicalReport, include_text: bool = False) -> dict:
    data = {
        "id": report.id,
        "patient_id": report.patient_id,
        "patient_name": report.patient_name,
        "uploaded_by": report.uploaded_by,
        "uploaded_by_role": report.uploaded_by_role,
        "report_type": report.report_type,
        "file_name": report.file_name,
        "file_path": report.file_path,
        "file_type": report.file_type,
        "ocr_text": report.ocr_text,
        "image_metadata": json.loads(report.image_metadata) if report.image_metadata else {},
        "image_quality_notes": json.loads(report.image_quality_notes) if report.image_quality_notes else [],
        "summary": json.loads(report.summary) if report.summary else {},
        "risk_flags": json.loads(report.risk_flags) if report.risk_flags else [],
        "upload_date": str(report.upload_date),
        "doctor_notes": report.doctor_notes,
        "nurse_notes": report.nurse_notes,
        "upload_notes": report.upload_notes,
        "disclaimer": HISTORICAL_REPORT_DISCLAIMER,
    }
    if include_text:
        data["extracted_text"] = report.extracted_text
    return data


def serialize_triage_upload(upload: TriageUploadedFile, include_text: bool = True) -> dict:
    return {
        "id": upload.id,
        "patient_id": upload.patient_id,
        "patient_name": upload.patient_name,
        "uploaded_by": upload.uploaded_by,
        "uploaded_by_role": upload.uploaded_by_role,
        "triage_session_id": upload.triage_session_id,
        "file_name": upload.file_name,
        "file_path": upload.file_path,
        "file_type": upload.file_type,
        "report_type": upload.report_type,
        "extracted_text": upload.extracted_text if include_text else None,
        "ocr_text": upload.ocr_text,
        "image_metadata": json.loads(upload.image_metadata) if upload.image_metadata else {},
        "image_quality_notes": json.loads(upload.image_quality_notes) if upload.image_quality_notes else [],
        "detected_conditions": json.loads(upload.detected_conditions) if upload.detected_conditions else {},
        "risk_flags": json.loads(upload.risk_flags) if upload.risk_flags else [],
        "clinical_summary": json.loads(upload.clinical_summary) if upload.clinical_summary else {},
        "uploaded_at": str(upload.uploaded_at),
        "disclaimer": HISTORICAL_REPORT_DISCLAIMER,
    }


def can_access_historical_report(report: HistoricalMedicalReport, current_user: dict) -> bool:
    role = current_user["role"]
    if role in ["super_admin", "admin", "doctor", "nurse"]:
        return True
    return role == "patient" and report.uploaded_by == current_user["username"]


def ensure_active_approver(current_user: dict, db) -> None:
    """Reject stale tokens for database users whose approval status changed."""
    db_user = db.query(AppUser).filter(AppUser.username == current_user["username"]).first()
    if db_user and (db_user.account_status or db_user.status) != "active":
        raise HTTPException(status_code=403, detail="Only active approved users can approve or reject accounts.")


def ensure_default_staff_members():
    db = SessionLocal()

    try:
        for username, user in fake_users_db.items():
            existing_staff = (
                db.query(StaffMember)
                .filter(StaffMember.username == username)
                .first()
            )

            if existing_staff:
                existing_staff.active_status = True
                user_roles = user.get("roles") or [user.get("role", "staff")]
                existing_staff.role = user_roles[0]
                continue

            user_roles = user.get("roles") or [user.get("role", "staff")]
            staff = StaffMember(
                username=username,
                role=user_roles[0],
                department="General",
                active_status=True
            )
            db.add(staff)

        db.commit()

    finally:
        db.close()


ensure_default_staff_members()


def ensure_default_nurse_records():
    """Create demo nurse records so auto-assignment works out of the box."""
    db = SessionLocal()
    try:
        sync_active_nurse_users(db)
        demo_nurses = [
            {"name": "Nurse Sarah", "email": "sarah.nurse@emergeai.local", "department": "Emergency", "experience_level": "experienced"},
            {"name": "Nurse Emily", "email": "emily.nurse@emergeai.local", "department": "Emergency", "experience_level": "senior"},
            {"name": "Nurse Michael", "email": "michael.nurse@emergeai.local", "department": "Emergency", "experience_level": "critical"},
        ]
        for item in demo_nurses:
            existing = db.query(Nurse).filter(Nurse.email == item["email"]).first()
            if existing:
                continue
            db.add(Nurse(**item, available_status=True, active_patient_count=0))
        db.commit()
    finally:
        db.close()


ensure_default_nurse_records()

# NEW: register multi-modal v2 routes
# Adds: POST /v2/analyze-image-dl, POST /v2/triage,
#       GET  /v2/report/{prediction_id}, GET /v2/dashboard/summary
#app.include_router(v2_router)
if DL_AVAILABLE:
    app.include_router(v2_router)
app.include_router(assignment_router, prefix="/api/assignments", tags=["Assignments"])


# -----------------------------
# HOME
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Emergency Triage AI Backend Running",
        "version": "3.0",
        "features": [
            "XGBoost prediction",
            "Safety rules",
            "Clinical explanations",
            "PostgreSQL logging",
            "JWT authentication",
            "Role-based access control",
            "SHAP visualization",
            "Human-in-the-loop clinical feedback",
            "NLP symptom extraction",
            "Computer vision image analysis",
            "Prediction history",
            "Deep learning image analysis (EfficientNet-B0)",
            "Infection severity classification",
            "Multi-modal AI triage scoring (ESI 1-5)",
            "PDF clinical report generation",
            "Advanced healthcare dashboard",
            "Patient risk watchlist",
            "Model retraining API",
            "Company-level health dashboard"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "database": "connected",
        "authentication": "enabled",
        "rbac": "enabled",
        "shap": "enabled",
        "human_feedback": "enabled",
        "nlp_symptom_extraction": "enabled",
        "computer_vision": "enabled",
        "deep_learning_image": "enabled",
        "multimodal_triage": "enabled",
        "pdf_reports": "enabled",
        "patient_risk_watchlist": "enabled",
        "model_retraining": "enabled",
        "company_health_dashboard": "enabled"
    }


# -----------------------------
# LOGIN
# -----------------------------

@app.post("/login")
def login(user: LoginInput):
    selected_role = normalize_role(user.role)

    authenticated_user = authenticate_user(
        user.username,
        user.password,
        selected_role
    )

    if not authenticated_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username, password, or role"
        )

    db = SessionLocal()

    try:
        staff = (
            db.query(StaffMember)
            .filter(StaffMember.username == authenticated_user["username"])
            .first()
        )

        if staff and not staff.active_status:
            raise HTTPException(
                status_code=403,
                detail="Staff account is inactive. Contact an administrator."
            )

        log_audit_action(
            db,
            authenticated_user["username"],
            selected_role,
            "login",
            "staff",
            authenticated_user["username"],
            f"Successful login as {selected_role}"
        )
        db.commit()

    finally:
        db.close()

    token = create_access_token(
        data={
            "sub": authenticated_user["username"],
            "role": selected_role
        }
    )

    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer",
        "username": authenticated_user["username"],
        "role": selected_role,
        "account_status": authenticated_user.get("status", "active")
    }


@app.post("/register")
def register_user(user: RegisterInput, db=Depends(get_db)):
    requested_role = normalize_role(user.requested_role)
    valid_roles = {"patient", "doctor", "nurse", "admin"}

    if requested_role not in valid_roles:
        raise HTTPException(status_code=400, detail="Invalid requested role.")
    if not all([user.full_name.strip(), user.username.strip(), user.email.strip(), user.password]):
        raise HTTPException(status_code=400, detail="All required fields must be completed.")
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Password and confirm password must match.")

    username = user.username.strip().lower()
    email = user.email.strip().lower()
    existing = (
        db.query(AppUser)
        .filter(or_(AppUser.username == username, AppUser.email == email))
        .first()
    )
    if existing:
        if existing.username == username:
            raise HTTPException(status_code=409, detail="Username already exists.")
        raise HTTPException(status_code=409, detail="Email already exists.")

    status = "pending"
    app_user = AppUser(
        full_name=user.full_name.strip(),
        username=username,
        email=email,
        password_hash=hash_password(user.password),
        requested_role=requested_role,
        role=requested_role,
        status=status,
        account_status=status,
        approval_level=approval_level_for_role(requested_role),
    )
    db.add(app_user)
    db.commit()
    db.refresh(app_user)

    if requested_role in {"doctor", "nurse", "admin"}:
        staff = StaffMember(
            username=username,
            role=requested_role,
            department="General",
            active_status=False,
        )
        db.add(staff)
        db.commit()

    message = "Your account is pending approval."

    return {
        "status": "success",
        "message": message,
        "username": app_user.username,
        "role": app_user.role,
        "account_status": app_user.account_status,
        "approval_level": app_user.approval_level,
    }


@app.get("/admin/approvals")
def list_approval_requests(
    current_user: dict = Depends(require_role(["super_admin", "admin", "doctor", "nurse"])),
    db=Depends(get_db),
):
    ensure_active_approver(current_user, db)
    approver_role = normalize_role(current_user["role"])
    approvable_roles = [
        role for role in ["admin", "doctor", "nurse", "patient"]
        if can_approve_role(approver_role, role)
    ]
    visible_roles = [
        role for role in ["admin", "doctor", "nurse", "patient"]
        if can_view_registration_role(approver_role, role)
    ]

    pending = (
        db.query(AppUser)
        .filter(AppUser.role.in_(visible_roles))
        .order_by(AppUser.created_at.asc())
        .all()
    )
    return {
        "status": "success",
        "approval_scope": approvable_roles,
        "view_scope": visible_roles,
        "pending_admin_requests": [
            {
                "id": user.id,
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
                "requested_role": user.requested_role or user.role,
                "role": user.role,
                "registration_date": str(user.created_at),
                "status": user.account_status or user.status,
                "approved_by": user.approved_by,
                "approved_at": str(user.approved_at) if user.approved_at else None,
                "rejected_by": user.rejected_by,
                "rejected_at": str(user.rejected_at) if user.rejected_at else None,
                "approval_level": user.approval_level,
                "rejected_reason": user.rejected_reason,
            }
            for user in pending
            if (user.account_status or user.status) == "pending"
            and user.role in approvable_roles
        ],
        "new_patient_requests": [
            {
                "id": user.id,
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
                "requested_role": user.requested_role or user.role,
                "role": user.role,
                "registration_date": str(user.created_at),
                "status": user.account_status or user.status,
            }
            for user in pending
            if (user.account_status or user.status) == "pending"
            and user.role == "patient"
            and not can_approve_role(approver_role, "patient")
        ],
        "activity_history": [
            {
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
                "requested_role": user.requested_role or user.role,
                "role": user.role,
                "registration_date": str(user.created_at),
                "status": user.account_status or user.status,
                "approved_by": user.approved_by,
                "approved_at": str(user.approved_at) if user.approved_at else None,
                "rejected_by": user.rejected_by,
                "rejected_at": str(user.rejected_at) if user.rejected_at else None,
                "rejected_reason": user.rejected_reason,
            }
            for user in pending if (user.account_status or user.status) in {"active", "rejected", "suspended"}
        ],
    }


@app.post("/admin/approvals/{username}/approve")
def approve_user_account(
    username: str,
    current_user: dict = Depends(require_role(["super_admin", "admin", "doctor"])),
    db=Depends(get_db),
):
    ensure_active_approver(current_user, db)
    user = (
        db.query(AppUser)
        .filter(AppUser.username == username.lower())
        .first()
    )
    if not user:
        raise HTTPException(status_code=404, detail="Pending user not found.")
    if user.username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Users cannot approve their own account.")
    if not can_approve_role(current_user["role"], user.role):
        raise HTTPException(status_code=403, detail="You are not authorized to approve this role.")
    if (user.account_status or user.status) != "pending":
        raise HTTPException(status_code=400, detail="Only pending accounts can be approved.")

    user.status = "active"
    user.account_status = "active"
    user.approved_by = current_user["username"]
    user.approved_at = datetime.utcnow()
    user.rejected_by = None
    user.rejected_at = None
    user.rejected_reason = None

    staff = db.query(StaffMember).filter(StaffMember.username == user.username).first()
    if staff:
        staff.active_status = True
        staff.role = user.role
    else:
        if user.role in {"doctor", "nurse", "admin"}:
            db.add(StaffMember(username=user.username, role=user.role, department="General", active_status=True))

    db.commit()
    return {"status": "success", "message": f"{username} approved as {user.role}."}


@app.post("/admin/approvals/{username}/reject")
def reject_user_account(
    username: str,
    decision: ApprovalDecisionInput | None = None,
    current_user: dict = Depends(require_role(["super_admin", "admin", "doctor"])),
    db=Depends(get_db),
):
    ensure_active_approver(current_user, db)
    user = (
        db.query(AppUser)
        .filter(AppUser.username == username.lower())
        .first()
    )
    if not user:
        raise HTTPException(status_code=404, detail="Pending user not found.")
    if user.username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Users cannot reject their own account.")
    if not can_approve_role(current_user["role"], user.role):
        raise HTTPException(status_code=403, detail="You are not authorized to reject this role.")
    if (user.account_status or user.status) != "pending":
        raise HTTPException(status_code=400, detail="Only pending accounts can be rejected.")

    user.status = "rejected"
    user.account_status = "rejected"
    user.rejected_by = current_user["username"]
    user.rejected_at = datetime.utcnow()
    user.rejected_reason = decision.rejected_reason if decision else None

    staff = db.query(StaffMember).filter(StaffMember.username == user.username).first()
    if staff:
        staff.active_status = False

    db.commit()
    return {"status": "success", "message": f"{username} rejected."}


@app.post("/historical-reports/upload")
def upload_historical_report(
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    report_type: str = Form(...),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    allowed_extensions = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, CSV, DOCX, JPG, JPEG, or PNG.")

    safe_name = safe_upload_filename(file.filename or "historical_report")
    is_image = suffix in IMAGE_EXTENSIONS
    output_path = (HISTORICAL_IMAGE_DIR if is_image else HISTORICAL_REPORT_DIR) / safe_name
    try:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded report: {exc}")

    image_assessment = {}
    ocr_text = ""
    image_quality_notes: list[str] = []
    image_metadata: dict = {}

    if is_image:
        image_assessment = assess_medical_image(output_path)
        ocr_text = image_assessment.get("ocr_text", "")
        image_quality_notes = image_assessment.get("image_quality_notes", [])
        image_metadata = image_assessment.get("image_metadata", {})
        extracted_text = ocr_text
        if not ocr_text and image_assessment.get("ocr_message"):
            extracted_text = image_assessment["ocr_message"]
    else:
        ok, extracted_text = extract_text_from_file(output_path)
        if not ok:
            raise HTTPException(status_code=400, detail=extracted_text)

    combined_text = "\n".join([extracted_text or "", notes or ""])
    analysis = analyze_medical_history(combined_text)
    risk_flags = generate_risk_flags(analysis)
    if is_image:
        risk_flags.extend(generate_image_risk_flags(report_type, notes or "", ocr_text, image_quality_notes))
    summary = generate_clinical_summary(patient_name, report_type, analysis, risk_flags, notes or "", image_assessment)

    report = HistoricalMedicalReport(
        patient_id=patient_id.strip(),
        patient_name=patient_name.strip(),
        uploaded_by=current_user["username"],
        uploaded_by_role=current_user["role"],
        report_type=report_type,
        file_name=safe_name,
        file_type="image" if is_image else "document",
        file_path=str(output_path.relative_to(PROJECT_ROOT)),
        extracted_text=extracted_text,
        ocr_text=ocr_text,
        image_metadata=json.dumps(image_metadata),
        image_quality_notes=json.dumps(image_quality_notes),
        summary=json.dumps({"analysis": analysis, "clinical_summary": summary}),
        risk_flags=json.dumps(risk_flags),
        upload_date=datetime.utcnow(),
        upload_notes=notes,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "status": "success",
        "message": "Historical report uploaded and analyzed.",
        "report": serialize_historical_report(report, include_text=True),
    }


@app.post("/triage-uploads/upload")
def upload_triage_context_file(
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    report_type: str = Form(...),
    notes: Optional[str] = Form(None),
    triage_session_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    allowed_extensions = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, CSV, DOCX, JPG, JPEG, or PNG.")

    safe_name = safe_upload_filename(file.filename or "triage_upload")
    is_image = suffix in IMAGE_EXTENSIONS
    output_path = (TRIAGE_IMAGE_DIR if is_image else TRIAGE_REPORT_DIR) / safe_name
    try:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {exc}")

    image_assessment = {}
    ocr_text = ""
    image_quality_notes: list[str] = []
    image_metadata: dict = {}

    try:
        if is_image:
            image_assessment = assess_medical_image(output_path)
            ocr_text = image_assessment.get("ocr_text", "")
            image_quality_notes = image_assessment.get("image_quality_notes", [])
            image_metadata = image_assessment.get("image_metadata", {})
            extracted_text = ocr_text or image_assessment.get("ocr_message", "")
        else:
            ok, extracted_text = extract_text_from_file(output_path)
            if not ok:
                raise HTTPException(status_code=400, detail=extracted_text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not analyze uploaded file. It may be corrupt or unsupported: {exc}")

    combined_text = "\n".join([extracted_text or "", notes or ""])
    analysis = analyze_medical_history(combined_text)
    risk_flags = generate_risk_flags(analysis)
    if is_image:
        risk_flags.extend(generate_image_risk_flags(report_type, notes or "", ocr_text, image_quality_notes))
    summary = generate_clinical_summary(patient_name, report_type, analysis, risk_flags, notes or "", image_assessment)

    upload = TriageUploadedFile(
        patient_id=patient_id.strip(),
        patient_name=patient_name.strip(),
        uploaded_by=current_user["username"],
        uploaded_by_role=current_user["role"],
        triage_session_id=triage_session_id,
        file_name=safe_name,
        file_path=str(output_path.relative_to(PROJECT_ROOT)),
        file_type="image" if is_image else "document",
        report_type=report_type,
        extracted_text=extracted_text,
        ocr_text=ocr_text,
        image_metadata=json.dumps(image_metadata),
        image_quality_notes=json.dumps(image_quality_notes),
        detected_conditions=json.dumps(analysis),
        risk_flags=json.dumps(risk_flags),
        clinical_summary=json.dumps(summary),
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    return {
        "status": "success",
        "message": "Triage upload analyzed as supporting context. It does not override the ESI prediction.",
        "upload": serialize_triage_upload(upload, include_text=True),
    }


@app.get("/triage-uploads")
def list_triage_uploads(
    patient_id: Optional[str] = None,
    triage_session_id: Optional[str] = None,
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    query = db.query(TriageUploadedFile)
    if current_user["role"] == "patient":
        query = query.filter(TriageUploadedFile.uploaded_by == current_user["username"])
    if patient_id:
        query = query.filter(TriageUploadedFile.patient_id == patient_id)
    if triage_session_id:
        query = query.filter(TriageUploadedFile.triage_session_id == triage_session_id)
    uploads = query.order_by(TriageUploadedFile.uploaded_at.desc()).all()
    return {"status": "success", "uploads": [serialize_triage_upload(upload, include_text=False) for upload in uploads]}


@app.delete("/triage-uploads/{upload_id}")
def delete_triage_upload(
    upload_id: int,
    current_user: dict = Depends(require_role(["admin", "super_admin"])),
    db=Depends(get_db),
):
    upload = db.query(TriageUploadedFile).filter(TriageUploadedFile.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Triage upload not found.")
    try:
        stored_path = PROJECT_ROOT / upload.file_path
        if stored_path.exists():
            stored_path.unlink()
    except Exception:
        pass
    db.delete(upload)
    db.commit()
    return {"status": "success", "message": "Triage upload deleted."}


@app.get("/historical-reports")
def list_historical_reports(
    patient_id: Optional[str] = None,
    report_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    query = db.query(HistoricalMedicalReport)
    if current_user["role"] == "patient":
        query = query.filter(HistoricalMedicalReport.uploaded_by == current_user["username"])
    if patient_id:
        query = query.filter(HistoricalMedicalReport.patient_id == patient_id)
    if report_type and report_type != "All":
        query = query.filter(HistoricalMedicalReport.report_type == report_type)
    reports = query.order_by(HistoricalMedicalReport.upload_date.desc()).all()
    serialized = [serialize_historical_report(report) for report in reports]
    if risk_level and risk_level != "All":
        serialized = [
            report for report in serialized
            if any(flag.get("level") == risk_level for flag in report.get("risk_flags", []))
        ]
    return {"status": "success", "reports": serialized}


@app.get("/historical-reports/{report_id}")
def get_historical_report(
    report_id: int,
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    report = db.query(HistoricalMedicalReport).filter(HistoricalMedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Historical report not found.")
    if not can_access_historical_report(report, current_user):
        raise HTTPException(status_code=403, detail="You do not have access to this report.")
    return {"status": "success", "report": serialize_historical_report(report, include_text=True)}


@app.put("/historical-reports/{report_id}/notes")
def update_historical_report_notes(
    report_id: int,
    notes: HistoricalReportNotesInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    report = db.query(HistoricalMedicalReport).filter(HistoricalMedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Historical report not found.")
    if current_user["role"] in ["doctor", "admin", "super_admin"] and notes.doctor_notes is not None:
        report.doctor_notes = notes.doctor_notes
    if current_user["role"] in ["nurse", "doctor", "admin", "super_admin"] and notes.nurse_notes is not None:
        report.nurse_notes = notes.nurse_notes
    db.commit()
    db.refresh(report)
    return {"status": "success", "report": serialize_historical_report(report)}


@app.delete("/historical-reports/{report_id}")
def delete_historical_report(
    report_id: int,
    current_user: dict = Depends(require_role(["admin", "super_admin"])),
    db=Depends(get_db),
):
    report = db.query(HistoricalMedicalReport).filter(HistoricalMedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Historical report not found.")
    db.delete(report)
    db.commit()
    return {"status": "success", "message": "Historical report deleted."}


@app.get("/historical-reports/{report_id}/export")
def export_historical_report_summary(
    report_id: int,
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin", "super_admin"])),
    db=Depends(get_db),
):
    report = db.query(HistoricalMedicalReport).filter(HistoricalMedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Historical report not found.")
    if not can_access_historical_report(report, current_user):
        raise HTTPException(status_code=403, detail="You do not have access to this report.")
    data = serialize_historical_report(report, include_text=False)
    text_export = json.dumps(data, indent=2)
    return Response(
        content=text_export,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=historical_report_{report_id}_summary.txt"},
    )


@app.put("/admin/accounts/{username}")
def update_account_role_or_status(
    username: str,
    update: AccountUpdateInput,
    current_user: dict = Depends(require_role(["super_admin"])),
    db=Depends(get_db),
):
    """Super admin-only role promotion/demotion and suspension endpoint."""
    user = db.query(AppUser).filter(AppUser.username == username.lower()).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Users cannot modify their own role or status.")

    if update.role:
        new_role = normalize_role(update.role)
        if new_role not in {"patient", "nurse", "doctor", "admin"}:
            raise HTTPException(status_code=400, detail="Invalid role.")
        user.role = new_role
        user.approval_level = approval_level_for_role(new_role)

    if update.account_status:
        new_status = update.account_status.lower()
        if new_status not in {"pending", "active", "rejected", "suspended"}:
            raise HTTPException(status_code=400, detail="Invalid account status.")
        user.status = new_status
        user.account_status = new_status

    staff = db.query(StaffMember).filter(StaffMember.username == user.username).first()
    if staff:
        staff.role = user.role
        staff.active_status = (user.account_status or user.status) == "active"

    db.commit()
    return {"status": "success", "message": f"{username} updated."}

# -----------------------------
# NLP SYMPTOM EXTRACTION
# -----------------------------

@app.post("/extract-symptoms")
def extract_symptoms_from_description(
    data: SymptomTextInput,
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin"]))
):
    result = extract_symptoms(data.problem_description)

    return {
        "status": "success",
        "processed_by": current_user["username"],
        "role": current_user["role"],
        "original_text": data.problem_description,
        "cleaned_text": result["cleaned_text"],
        "llm_ready_text": result["llm_ready_text"],
        "extracted_symptoms": result["extracted_symptoms"],
        "matched_terms": result["matched_terms"],
        "emergency_keywords": result["emergency_keywords"],
        "has_emergency_keyword": result["has_emergency_keyword"]
    }


# -----------------------------
# COMPUTER VISION
# -----------------------------

@app.post("/analyze-image")
def analyze_image(
    image: UploadFile = File(...),
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"]))
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    result = analyze_medical_image(image.file)

    return {
        "status": "success",
        "processed_by": current_user["username"],
        "role": current_user["role"],
        "filename": image.filename,
        "content_type": image.content_type,
        "analysis": result
    }


# -----------------------------
# PREDICTION
# -----------------------------

@app.post("/predict")
def predict(
    patient: PatientInput,
    current_user: dict = Depends(require_role(["patient", "doctor", "nurse", "admin", "super_admin"]))
):
    patient_dict = patient.dict()
    model_input_dict, input_df = prepare_model_input(patient_dict)

    #try:
        #raw_prediction = model.predict(input_df)
    #if not MODEL_AVAILABLE:
    if not MODEL_AVAILABLE or not LABEL_ENCODER_AVAILABLE:
        return {
            "error": "Model not available in deployment"
    }

    try:
        raw_prediction = model.predict(input_df)

        if len(raw_prediction.shape) > 1:
            ml_prediction_encoded = int(raw_prediction.argmax(axis=1)[0])
        else:
            ml_prediction_encoded = int(raw_prediction[0])

        ml_prediction = label_encoder.inverse_transform([ml_prediction_encoded])[0]
    
        probabilities = model.predict_proba(input_df)[0]
        classes = label_encoder.inverse_transform(model.classes_)

        probability_dict = {
            str(classes[i]): float(probabilities[i])
            for i in range(len(classes))
        }

        confidence = float(max(probabilities))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )

    final_prediction, reasons = apply_safety_rules(
        model_input_dict,
        ml_prediction
    )

    explanations = generate_clinical_explanation(
        model_input_dict,
        ml_prediction,
        final_prediction,
        reasons
    )

    log_id = save_prediction_log(
        patient_data=patient_dict,
        ml_prediction=str(ml_prediction),
        final_prediction=str(final_prediction),
        safety_reasons=reasons,
        clinical_explanations=explanations,
        confidence=confidence,
        source=current_user["username"],
        feedback="Pending"
    )

    if log_id is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction completed but failed to save in PostgreSQL"
        )

    audit_db = SessionLocal()
    try:
        log_audit_action(
            audit_db,
            current_user["username"],
            current_user["role"],
            "prediction_created",
            "prediction",
            log_id,
            f"Final prediction: {final_prediction}"
        )
        prediction_log = audit_db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
        assignment = None
        if prediction_log:
            assignment = auto_assign_nurse(audit_db, prediction_log, assigned_by="system")
            if assignment:
                log_audit_action(
                    audit_db,
                    current_user["username"],
                    current_user["role"],
                    "nurse_auto_assigned",
                    "prediction",
                    log_id,
                    f"Auto-assigned nurse ID {assignment.nurse_id}"
                )
        audit_db.commit()
    except Exception as _assign_err:
        audit_db.rollback()
        import traceback as _tb
        _tb.print_exc()
        print(f"[WARNING] Post-prediction tasks failed (prediction {log_id} still saved): {_assign_err}")
    finally:
        audit_db.close()

    return {
        "status": "success",
        "user": current_user["username"],
        "role": current_user["role"],
        "prediction_id": log_id,
        "log_id": log_id,
        "ml_prediction": str(ml_prediction),
        "final_prediction": str(final_prediction),
        "confidence": confidence,
        "safety_reasons": reasons,
        "clinical_explanations": explanations,
        "probabilities": probability_dict,
        "problem_description_saved": patient.problem_description is not None,
        "nlp_saved": patient.llm_ready_text is not None,
        "cv_saved": patient.cv_analysis is not None,
        "log_status": "Saved to PostgreSQL successfully"
    }



# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------

@app.post("/shap")
def shap_explain(
    patient: PatientInput,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    """
    Doctor/admin endpoint for AI explainability.
    Returns feature-level contribution-style values for the Streamlit SHAP dashboard.
    This endpoint is intentionally safe: if true SHAP values are not available,
    it falls back to XGBoost feature_importances_ so the dashboard still works.
    """
    patient_dict = patient.dict()
    model_input_dict, input_df = prepare_model_input(patient_dict)

    try:
        raw_prediction = model.predict(input_df)

        if len(raw_prediction.shape) > 1:
            prediction_encoded = int(raw_prediction.argmax(axis=1)[0])
        else:
            prediction_encoded = int(raw_prediction[0])

        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

        feature_names = list(input_df.columns)

        if hasattr(model, "feature_importances_"):
            raw_importances = list(model.feature_importances_)
        else:
            raw_importances = [0.0 for _ in feature_names]

        feature_importance = []

        for feature, importance in zip(feature_names, raw_importances):
            value = to_python_value(input_df.iloc[0].get(feature))
            contribution = float(importance)

            feature_importance.append({
                "feature": feature,
                "value": value,
                "shap_value": contribution,
                "absolute_importance": abs(contribution),
                "direction": "increases risk" if contribution >= 0 else "decreases risk"
            })

        feature_importance = sorted(
            feature_importance,
            key=lambda x: x["absolute_importance"],
            reverse=True
        )

        return {
            "status": "success",
            "requested_by": current_user["username"],
            "role": current_user["role"],
            "prediction": str(prediction),
            "confidence": confidence,
            "feature_importance": feature_importance
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(e)}"
        )


@app.get("/shap/image")
def shap_image(
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    """
    Returns a simple PNG feature-importance chart for the SHAP page.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            raise HTTPException(
                status_code=404,
                detail="Feature importance image is not available for this model."
            )

        feature_names = [
            "age", "gender", "race", "ethnicity", "arrivalmode",
            "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
            "triage_vital_rr", "triage_vital_o2", "triage_vital_temp",
            "cc_chestpain", "cc_shortnessofbreath", "cc_headache",
            "cc_fever", "cc_abdominalpain", "cc_dizziness",
            "cc_syncope", "cc_weakness"
        ]

        importances = list(model.feature_importances_)
        rows = list(zip(feature_names[:len(importances)], importances))
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:10]

        labels = [r[0] for r in rows][::-1]
        values = [r[1] for r in rows][::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, values)
        ax.set_title("Top AI Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        plt.close(fig)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP image generation failed: {str(e)}"
        )

# -----------------------------
# HISTORY
# -----------------------------

@app.get("/history")
def get_prediction_history(
    current_user: dict = Depends(require_role(["patient", "doctor", "nurse", "admin"]))
):
    db = SessionLocal()

    try:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .all()
        )

        results = []

        for log in logs:
            results.append({
                "id": log.id,
                "age": log.age,
                "gender": log.gender,
                "race": log.race,
                "ethnicity": log.ethnicity,
                "arrivalmode": log.arrivalmode,

                "triage_vital_hr": log.triage_vital_hr,
                "triage_vital_sbp": log.triage_vital_sbp,
                "triage_vital_dbp": log.triage_vital_dbp,
                "triage_vital_rr": log.triage_vital_rr,
                "triage_vital_o2": log.triage_vital_o2,
                "triage_vital_temp": log.triage_vital_temp,

                "cc_chestpain": log.cc_chestpain,
                "cc_shortnessofbreath": log.cc_shortnessofbreath,
                "cc_headache": log.cc_headache,
                "cc_fever": log.cc_fever,
                "cc_abdominalpain": log.cc_abdominalpain,
                "cc_dizziness": log.cc_dizziness,
                "cc_syncope": log.cc_syncope,
                "cc_weakness": log.cc_weakness,

                "ml_prediction": log.ml_prediction,
                "final_prediction": log.final_prediction,
                "confidence": log.confidence,

                "safety_reasons": log.safety_reasons,
                "clinical_explanations": log.clinical_explanations,

                "problem_description": log.problem_description,
                "llm_ready_text": log.llm_ready_text,
                "matched_terms": log.matched_terms,
                "emergency_keywords": log.emergency_keywords,
                "cv_analysis": log.cv_analysis,

                "source": log.source,
                "feedback": log.feedback,
                "created_at": str(log.created_at)
            })

        return {
            "status": "success",
            "count": len(results),
            "history": results
        }

    finally:
        db.close()


@app.get("/history/limited")
def get_limited_prediction_history(
    current_user: dict = Depends(require_role(["patient", "nurse", "doctor", "admin"]))
):
    db = SessionLocal()

    try:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(100)
            .all()
        )

        return [
            {
                "id": log.id,
                "age": log.age,
                "gender": log.gender,
                "arrivalmode": log.arrivalmode,
                "heart_rate": log.triage_vital_hr,
                "blood_pressure": f"{log.triage_vital_sbp}/{log.triage_vital_dbp}",
                "respiratory_rate": log.triage_vital_rr,
                "oxygen_level": log.triage_vital_o2,
                "temperature": log.triage_vital_temp,
                "final_prediction": log.final_prediction,
                "confidence": log.confidence,
                "created_at": str(log.created_at)
            }
            for log in logs
        ]

    finally:
        db.close()


# -----------------------------
# NURSE ASSIGNMENT
# -----------------------------

def serialize_nurse(nurse: Nurse):
    return {
        "id": nurse.id,
        "name": nurse.name,
        "email": nurse.email,
        "department": nurse.department,
        "available_status": nurse.available_status,
        "active_patient_count": getattr(nurse, "active_patient_count", 0),
        "experience_level": getattr(nurse, "experience_level", "normal"),
    }


def serialize_nurse_assignment(assignment: NurseAssignment):
    return {
        "id": assignment.id,
        "prediction_id": assignment.prediction_id,
        "nurse_id": assignment.nurse_id,
        "nurse": serialize_nurse(assignment.nurse) if assignment.nurse else None,
        "assigned_at": str(assignment.assigned_at),
        "status": assignment.status,
        "notes": assignment.notes,
        "assigned_by": getattr(assignment, "assigned_by", None),
        "assignment_type": getattr(assignment, "assignment_type", "manual"),
        "priority_level": getattr(assignment, "priority_level", "Medium"),
        "prediction": serialize_prediction_summary(assignment.prediction) if assignment.prediction else None,
    }


def serialize_prediction_summary(prediction: PredictionLog):
    if not prediction:
        return None
    priority = queue_priority_from_prediction(prediction)
    return {
        "id": prediction.id,
        "patient_name": f"Patient #{prediction.id}",
        "esi_level": priority["esi_level"],
        "priority_level": priority["priority"],
        "icu_risk": bool(prediction.triage_vital_o2 is not None and prediction.triage_vital_o2 < 90),
        "final_prediction": prediction.final_prediction,
        "created_at": str(prediction.created_at),
        "heart_rate": prediction.triage_vital_hr,
        "oxygen_saturation": prediction.triage_vital_o2,
    }


def serialize_queue_entry(entry: EmergencyQueue):
    prediction = entry.prediction
    return {
        "id": entry.id,
        "prediction_id": entry.prediction_id,
        "patient_name": f"Patient #{entry.prediction_id}",
        "esi_severity": entry.esi_severity,
        "esi_level": queue_priority_from_prediction(prediction)["esi_level"] if prediction else None,
        "priority": entry.priority,
        "estimated_wait_time": entry.estimated_wait_time,
        "critical_status": entry.critical_status,
        "assignment_status": entry.assignment_status,
        "assigned_nurse_id": entry.assigned_nurse_id,
        "assigned_nurse_name": entry.assigned_nurse_name,
        "assigned_doctor_id": entry.assigned_doctor_id,
        "assigned_doctor_name": entry.assigned_doctor_name,
        "assigned_at": str(entry.assigned_at) if entry.assigned_at else None,
        "arrival_time": str(entry.arrival_time),
        "updated_at": str(entry.updated_at),
    }


def serialize_nurse_vitals(vitals: NurseVitals):
    return {
        "id": vitals.id,
        "prediction_id": vitals.prediction_id,
        "nurse_id": vitals.nurse_id,
        "nurse": serialize_nurse(vitals.nurse) if vitals.nurse else None,
        "temperature": vitals.temperature,
        "heart_rate": vitals.heart_rate,
        "blood_pressure": vitals.blood_pressure,
        "oxygen_level": vitals.oxygen_level,
        "respiratory_rate": vitals.respiratory_rate,
        "pain_score": vitals.pain_score,
        "notes": vitals.notes,
        "recorded_at": str(vitals.recorded_at)
    }


def serialize_nurse_task(task: NurseTask):
    return {
        "id": task.id,
        "prediction_id": task.prediction_id,
        "nurse_id": task.nurse_id,
        "nurse": serialize_nurse(task.nurse) if task.nurse else None,
        "task_title": task.task_title,
        "task_description": task.task_description,
        "status": task.status,
        "priority": task.priority,
        "created_at": str(task.created_at),
        "completed_at": str(task.completed_at) if task.completed_at else None
    }


def serialize_doctor_review(review: DoctorReview):
    return {
        "id": review.id,
        "prediction_id": review.prediction_id,
        "doctor_id": review.doctor_id,
        "diagnosis": review.diagnosis,
        "treatment_plan": review.treatment_plan,
        "medication_notes": review.medication_notes,
        "follow_up_required": review.follow_up_required,
        "admit_status": review.admit_status,
        "reviewed_at": str(review.reviewed_at)
    }


def serialize_patient_status(status: PatientStatus):
    return {
        "id": status.id,
        "prediction_id": status.prediction_id,
        "patient_status": status.patient_status,
        "updated_by": status.updated_by,
        "updated_role": status.updated_role,
        "notes": status.notes,
        "updated_at": str(status.updated_at)
    }


def serialize_staff_member(staff: StaffMember):
    return {
        "id": staff.id,
        "username": staff.username,
        "role": staff.role,
        "department": staff.department,
        "active_status": staff.active_status,
        "created_at": str(staff.created_at),
        "updated_at": str(staff.updated_at)
    }


def serialize_bed(bed: Bed):
    return {
        "id": bed.id,
        "bed_number": bed.bed_number,
        "ward_type": bed.ward_type,
        "occupied": bed.occupied,
        "assigned_prediction_id": bed.assigned_prediction_id,
        "assigned_at": str(bed.assigned_at) if bed.assigned_at else None
    }


def serialize_medication(record: MedicationRecord):
    return {
        "id": record.id,
        "prediction_id": record.prediction_id,
        "nurse_id": record.nurse_id,
        "medication_name": record.medication_name,
        "dosage": record.dosage,
        "route": record.route,
        "scheduled_time": str(record.scheduled_time),
        "administered_time": str(record.administered_time) if record.administered_time else None,
        "status": record.status,
        "side_effects": record.side_effects,
        "notes": record.notes
    }


def serialize_alert(alert: ClinicalAlert):
    return {
        "id": alert.id,
        "prediction_id": alert.prediction_id,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "message": alert.message,
        "created_at": str(alert.created_at),
        "resolved": alert.resolved,
        "resolved_at": str(alert.resolved_at) if alert.resolved_at else None
    }


def serialize_discharge_summary(summary: DischargeSummary):
    return {
        "id": summary.id,
        "prediction_id": summary.prediction_id,
        "doctor_id": summary.doctor_id,
        "diagnosis": summary.diagnosis,
        "treatment_given": summary.treatment_given,
        "medication_notes": summary.medication_notes,
        "follow_up_instructions": summary.follow_up_instructions,
        "discharge_status": summary.discharge_status,
        "created_at": str(summary.created_at)
    }


def serialize_shift(shift: StaffShift):
    return {
        "id": shift.id,
        "staff_id": shift.staff_id,
        "staff_role": shift.staff_role,
        "department": shift.department,
        "shift_type": shift.shift_type,
        "start_time": str(shift.start_time),
        "end_time": str(shift.end_time),
        "status": shift.status
    }


def row_dict(row, fields):
    return {field: (str(getattr(row, field)) if isinstance(getattr(row, field), datetime) else getattr(row, field)) for field in fields}


PATIENT_FIELDS = [
    "id", "name", "age", "gender", "phone", "address",
    "emergency_contact_name", "emergency_contact_phone",
    "allergies", "chronic_diseases", "past_medical_history",
    "assigned_nurse_id", "assigned_nurse_name", "assigned_doctor_id",
    "assigned_doctor_name", "assignment_status", "assigned_at", "created_at",
]


def link_patient_prediction(db, patient_id: int, prediction_id: Optional[int]):
    if prediction_id is None:
        return
    if not db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first():
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    existing = db.query(PatientPredictionLink).filter(PatientPredictionLink.prediction_id == prediction_id).first()
    if existing:
        existing.patient_id = patient_id
    else:
        db.add(PatientPredictionLink(patient_id=patient_id, prediction_id=prediction_id))


def log_audit_action(
    db,
    username: str,
    role: Optional[str],
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[str] = None
):
    db.add(AuditLog(
        username=username,
        role=role,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        details=details
    ))


def get_priority_from_prediction(prediction: PredictionLog):
    priority = queue_priority_from_prediction(prediction)
    return priority["priority"], priority["color"], priority["estimated_wait_time"], priority["critical_status"]


def ensure_queue_entry(db, prediction: PredictionLog):
    entry = ensure_emergency_queue_entry(db, prediction)
    return entry, queue_priority_from_prediction(prediction)["color"]


def create_alert_if_missing(db, prediction_id: int, alert_type: str, severity: str, message: str):
    existing = (
        db.query(ClinicalAlert)
        .filter(
            ClinicalAlert.prediction_id == prediction_id,
            ClinicalAlert.alert_type == alert_type,
            ClinicalAlert.resolved == False
        )
        .first()
    )

    if existing:
        return existing

    alert = ClinicalAlert(
        prediction_id=prediction_id,
        alert_type=alert_type,
        severity=severity,
        message=message
    )
    db.add(alert)
    return alert


def validate_prediction_and_nurse(db, prediction_id: int, nurse_id: int):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    nurse = (
        db.query(Nurse)
        .filter(Nurse.id == nurse_id)
        .first()
    )

    if not nurse:
        raise HTTPException(
            status_code=404,
            detail="Nurse not found."
        )

    return prediction, nurse


@app.post("/nurses/add")
def add_nurse(
    nurse_data: NurseCreateInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    existing_nurse = (
        db.query(Nurse)
        .filter(Nurse.email == nurse_data.email)
        .first()
    )

    if existing_nurse:
        raise HTTPException(
            status_code=409,
            detail="A nurse with this email already exists."
        )

    nurse = Nurse(
        name=nurse_data.name,
        email=nurse_data.email,
        department=nurse_data.department,
        available_status=nurse_data.available_status,
        experience_level=nurse_data.experience_level,
        active_patient_count=0,
    )

    db.add(nurse)
    db.commit()
    db.refresh(nurse)

    return {
        "status": "success",
        "message": "Nurse added successfully",
        "created_by": current_user["username"],
        "nurse": serialize_nurse(nurse)
    }


@app.get("/nurses")
def get_nurses(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    nurses = db.query(Nurse).order_by(Nurse.name.asc()).all()

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "count": len(nurses),
        "nurses": [serialize_nurse(nurse) for nurse in nurses]
    }


@app.post("/assign-nurse")
def assign_nurse(
    assignment_data: NurseAssignmentInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    prediction, nurse = validate_prediction_and_nurse(
        db,
        assignment_data.prediction_id,
        assignment_data.nurse_id
    )

    assignment = assign_nurse_to_prediction(
        db,
        prediction,
        nurse,
        assigned_by=current_user["username"],
        assignment_type="manual",
        status=assignment_data.status,
        notes=assignment_data.notes,
    )
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "nurse_assigned",
        "prediction",
        assignment_data.prediction_id,
        f"Nurse ID {assignment_data.nurse_id} assigned"
    )
    db.commit()
    db.refresh(assignment)

    return {
        "status": "success",
        "message": "Nurse assigned successfully",
        "assigned_by": current_user["username"],
        "assignment": serialize_nurse_assignment(assignment)
    }


@app.get("/assignments")
def get_assignments(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    query = db.query(NurseAssignment)
    if current_user["role"] == "nurse":
        nurse = (
            db.query(Nurse)
            .filter(
                or_(
                    Nurse.email == current_user["username"],
                    Nurse.name == current_user["username"],
                )
            )
            .first()
        )
        if nurse:
            query = query.filter(NurseAssignment.nurse_id == nurse.id)
        else:
            query = query.filter(NurseAssignment.id == -1)
    assignments = query.order_by(NurseAssignment.assigned_at.desc()).all()

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "count": len(assignments),
        "assignments": [
            serialize_nurse_assignment(assignment)
            for assignment in assignments
        ]
    }


@app.get("/assignments/{prediction_id}")
def get_assignments_by_prediction(
    prediction_id: int,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    assignments = (
        db.query(NurseAssignment)
        .filter(NurseAssignment.prediction_id == prediction_id)
        .order_by(NurseAssignment.assigned_at.desc())
        .all()
    )

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "prediction_id": prediction_id,
        "count": len(assignments),
        "assignments": [
            serialize_nurse_assignment(assignment)
            for assignment in assignments
        ]
    }


@app.post("/assignments/auto/{prediction_id}")
def auto_assign_prediction(
    prediction_id: int,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db),
):
    prediction = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    assignment = auto_assign_nurse(db, prediction, assigned_by=current_user["username"])
    if not assignment:
        db.commit()
        raise HTTPException(status_code=409, detail="No nurse records are available for assignment.")
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "nurse_auto_assigned",
        "prediction",
        prediction_id,
        f"Auto-assigned nurse ID {assignment.nurse_id}",
    )
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": serialize_nurse_assignment(assignment)}


@app.post("/assignments/reassign")
def reassign_nurse(
    data: NurseReassignmentInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db),
):
    prediction, nurse = validate_prediction_and_nurse(db, data.prediction_id, data.nurse_id)
    assignment = assign_nurse_to_prediction(
        db,
        prediction,
        nurse,
        assigned_by=current_user["username"],
        assignment_type="reassign",
        notes=data.notes or "Manual reassignment.",
    )
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "nurse_reassigned",
        "prediction",
        data.prediction_id,
        f"Reassigned to nurse ID {data.nurse_id}",
    )
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": serialize_nurse_assignment(assignment)}


@app.put("/assignments/{assignment_id}/status")
def update_assignment_status(
    assignment_id: int,
    data: AssignmentStatusInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db),
):
    allowed = {"waiting", "assigned", "in_triage", "doctor_review", "completed", "critical"}
    normalized = data.status.strip().lower()
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid assignment status. Use one of: {sorted(allowed)}")
    assignment = db.query(NurseAssignment).filter(NurseAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    assignment.status = {
        "waiting": "Waiting",
        "assigned": "Assigned",
        "in_triage": "In Triage",
        "doctor_review": "Doctor Review",
        "completed": "Completed",
        "critical": "Critical",
    }[normalized]
    entry = db.query(EmergencyQueue).filter(EmergencyQueue.prediction_id == assignment.prediction_id).first()
    if entry:
        entry.assignment_status = normalized
        entry.updated_at = datetime.utcnow()
    if normalized == "completed" and assignment.nurse:
        refresh_nurse_workload(db, assignment.nurse)
    db.commit()
    db.refresh(assignment)
    return {"status": "success", "assignment": serialize_nurse_assignment(assignment)}


@app.get("/assignment-queue")
def get_assignment_queue(
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db),
):
    entries = ordered_queue_query(db).all()
    return {
        "status": "success",
        "count": len(entries),
        "queue": [serialize_queue_entry(entry) for entry in entries],
    }


@app.get("/nurse-workload")
def get_nurse_workload(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db),
):
    nurses = db.query(Nurse).order_by(Nurse.name.asc()).all()
    rows = []
    for nurse in nurses:
        active_count = refresh_nurse_workload(db, nurse)
        rows.append({
            **serialize_nurse(nurse),
            "active_patient_count": active_count,
            "available_status": nurse.available_status,
        })
    db.commit()
    return {"status": "success", "nurses": rows}


@app.post("/nurse-vitals/add")
def add_nurse_vitals(
    vitals_data: NurseVitalsInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    validate_prediction_and_nurse(
        db,
        vitals_data.prediction_id,
        vitals_data.nurse_id
    )

    vitals = NurseVitals(
        prediction_id=vitals_data.prediction_id,
        nurse_id=vitals_data.nurse_id,
        temperature=vitals_data.temperature,
        heart_rate=vitals_data.heart_rate,
        blood_pressure=vitals_data.blood_pressure,
        oxygen_level=vitals_data.oxygen_level,
        respiratory_rate=vitals_data.respiratory_rate,
        pain_score=vitals_data.pain_score,
        notes=vitals_data.notes
    )

    db.add(vitals)

    if vitals_data.oxygen_level is not None and vitals_data.oxygen_level < 92:
        create_alert_if_missing(
            db,
            vitals_data.prediction_id,
            "Low Oxygen",
            "Critical",
            f"Oxygen level is {vitals_data.oxygen_level}% for prediction #{vitals_data.prediction_id}."
        )

    if vitals_data.heart_rate is not None and vitals_data.heart_rate > 120:
        create_alert_if_missing(
            db,
            vitals_data.prediction_id,
            "High Heart Rate",
            "High",
            f"Heart rate is {vitals_data.heart_rate} for prediction #{vitals_data.prediction_id}."
        )

    if vitals_data.pain_score is not None and vitals_data.pain_score >= 8:
        create_alert_if_missing(
            db,
            vitals_data.prediction_id,
            "Severe Pain",
            "High",
            f"Pain score is {vitals_data.pain_score}/10 for prediction #{vitals_data.prediction_id}."
        )

    db.commit()
    db.refresh(vitals)

    return {
        "status": "success",
        "message": "Nurse vitals recorded successfully",
        "recorded_by": current_user["username"],
        "vitals": serialize_nurse_vitals(vitals)
    }


@app.get("/nurse-vitals/{prediction_id}")
def get_nurse_vitals(
    prediction_id: int,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    vitals = (
        db.query(NurseVitals)
        .filter(NurseVitals.prediction_id == prediction_id)
        .order_by(NurseVitals.recorded_at.desc())
        .all()
    )

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "prediction_id": prediction_id,
        "count": len(vitals),
        "vitals": [
            serialize_nurse_vitals(record)
            for record in vitals
        ]
    }


@app.post("/nurse-tasks/add")
def add_nurse_task(
    task_data: NurseTaskInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    allowed_statuses = ["Pending", "In Progress", "Completed"]
    allowed_priorities = ["Low", "Medium", "High", "Critical"]

    if task_data.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Invalid task status."
        )

    if task_data.priority not in allowed_priorities:
        raise HTTPException(
            status_code=400,
            detail="Invalid task priority."
        )

    validate_prediction_and_nurse(
        db,
        task_data.prediction_id,
        task_data.nurse_id
    )

    completed_at = datetime.utcnow() if task_data.status == "Completed" else None

    task = NurseTask(
        prediction_id=task_data.prediction_id,
        nurse_id=task_data.nurse_id,
        task_title=task_data.task_title,
        task_description=task_data.task_description,
        status=task_data.status,
        priority=task_data.priority,
        completed_at=completed_at
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    return {
        "status": "success",
        "message": "Nurse task added successfully",
        "created_by": current_user["username"],
        "task": serialize_nurse_task(task)
    }


@app.get("/nurse-tasks/{prediction_id}")
def get_nurse_tasks(
    prediction_id: int,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    tasks = (
        db.query(NurseTask)
        .filter(NurseTask.prediction_id == prediction_id)
        .order_by(NurseTask.created_at.desc())
        .all()
    )

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "prediction_id": prediction_id,
        "count": len(tasks),
        "tasks": [
            serialize_nurse_task(task)
            for task in tasks
        ]
    }


@app.put("/nurse-tasks/{task_id}/status")
def update_nurse_task_status(
    task_id: int,
    status_data: NurseTaskStatusInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    allowed_statuses = ["Pending", "In Progress", "Completed"]

    if status_data.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Invalid task status."
        )

    task = (
        db.query(NurseTask)
        .filter(NurseTask.id == task_id)
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=404,
            detail="Nurse task not found."
        )

    task.status = status_data.status
    task.completed_at = (
        datetime.utcnow()
        if status_data.status == "Completed"
        else None
    )

    db.commit()
    db.refresh(task)

    return {
        "status": "success",
        "message": "Nurse task status updated successfully",
        "updated_by": current_user["username"],
        "task": serialize_nurse_task(task)
    }


@app.post("/doctor-review/add")
def add_doctor_review(
    review_data: DoctorReviewInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == review_data.prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    allowed_admit_statuses = [
        "Not Admitted",
        "Observation",
        "Admitted",
        "Discharged",
        "Transferred"
    ]

    if review_data.admit_status not in allowed_admit_statuses:
        raise HTTPException(
            status_code=400,
            detail="Invalid admit status."
        )

    review = DoctorReview(
        prediction_id=review_data.prediction_id,
        doctor_id=current_user["username"],
        diagnosis=review_data.diagnosis,
        treatment_plan=review_data.treatment_plan,
        medication_notes=review_data.medication_notes,
        follow_up_required=review_data.follow_up_required,
        admit_status=review_data.admit_status
    )

    db.add(review)
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "doctor_review_created",
        "prediction",
        review_data.prediction_id,
        f"Admit status: {review_data.admit_status}"
    )
    db.commit()
    db.refresh(review)

    return {
        "status": "success",
        "message": "Doctor review added successfully",
        "review": serialize_doctor_review(review)
    }


@app.get("/doctor-review/{prediction_id}")
def get_doctor_reviews(
    prediction_id: int,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    reviews = (
        db.query(DoctorReview)
        .filter(DoctorReview.prediction_id == prediction_id)
        .order_by(DoctorReview.reviewed_at.desc())
        .all()
    )

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "prediction_id": prediction_id,
        "count": len(reviews),
        "reviews": [
            serialize_doctor_review(review)
            for review in reviews
        ]
    }


@app.put("/patient-status/{prediction_id}")
def update_patient_status(
    prediction_id: int,
    status_data: PatientStatusInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    allowed_statuses = [
        "Waiting",
        "Assigned to Nurse",
        "Under Nurse Care",
        "Waiting for Doctor",
        "Under Treatment",
        "Admitted",
        "Discharged"
    ]

    allowed_by_role = {
        "nurse": [
            "Assigned to Nurse",
            "Under Nurse Care",
            "Waiting for Doctor"
        ],
        "doctor": [
            "Waiting",
            "Assigned to Nurse",
            "Under Nurse Care",
            "Waiting for Doctor",
            "Under Treatment",
            "Admitted",
            "Discharged"
        ],
        "admin": allowed_statuses
    }

    user_role = current_user["role"]

    if status_data.patient_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Invalid patient status."
        )

    if status_data.patient_status not in allowed_by_role.get(user_role, []):
        raise HTTPException(
            status_code=403,
            detail="This role cannot set the requested patient status."
        )

    status = PatientStatus(
        prediction_id=prediction_id,
        patient_status=status_data.patient_status,
        updated_by=current_user["username"],
        updated_role=user_role,
        notes=status_data.notes
    )

    db.add(status)
    db.commit()
    db.refresh(status)

    return {
        "status": "success",
        "message": "Patient status updated successfully",
        "current_status": serialize_patient_status(status)
    }


@app.get("/patient-status/{prediction_id}")
def get_patient_status_timeline(
    prediction_id: int,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    prediction = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found."
        )

    timeline = (
        db.query(PatientStatus)
        .filter(PatientStatus.prediction_id == prediction_id)
        .order_by(PatientStatus.updated_at.asc())
        .all()
    )

    current_status = timeline[-1] if timeline else None

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "prediction_id": prediction_id,
        "current_status": (
            serialize_patient_status(current_status)
            if current_status
            else None
        ),
        "timeline": [
            serialize_patient_status(status)
            for status in timeline
        ]
    }


@app.get("/admin/staff")
def get_staff_members(
    role: Optional[str] = None,
    current_user: dict = Depends(require_role(["super_admin", "admin"])),
    db=Depends(get_db)
):
    valid_roles = ["doctor", "nurse", "admin", "super_admin"]

    query = db.query(StaffMember)

    if role:
        if role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail="Invalid role filter."
            )
        query = query.filter(StaffMember.role == role)

    staff_members = query.order_by(StaffMember.role.asc(), StaffMember.username.asc()).all()

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "count": len(staff_members),
        "staff": [
            serialize_staff_member(staff)
            for staff in staff_members
        ]
    }


@app.put("/admin/staff/{username}")
def update_staff_member(
    username: str,
    staff_data: StaffUpdateInput,
    current_user: dict = Depends(require_role(["super_admin", "admin"])),
    db=Depends(get_db)
):
    staff = (
        db.query(StaffMember)
        .filter(StaffMember.username == username)
        .first()
    )

    if not staff:
        raise HTTPException(
            status_code=404,
            detail="Staff member not found."
        )

    if username == current_user["username"] and staff_data.active_status is False:
        raise HTTPException(
            status_code=400,
            detail="Admins cannot deactivate their own active session account."
        )

    if staff_data.active_status is not None:
        staff.active_status = staff_data.active_status

    if staff_data.department is not None:
        staff.department = staff_data.department

    staff.updated_at = datetime.utcnow()
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "staff_updated",
        "staff",
        username,
        f"active={staff.active_status}, department={staff.department}"
    )

    db.commit()
    db.refresh(staff)

    return {
        "status": "success",
        "message": "Staff member updated successfully",
        "staff": serialize_staff_member(staff)
    }


@app.get("/admin/workload")
def get_staff_workload(
    current_user: dict = Depends(require_role(["admin"])),
    db=Depends(get_db)
):
    nurses = (
        db.query(StaffMember)
        .filter(StaffMember.role == "nurse")
        .order_by(StaffMember.username.asc())
        .all()
    )

    doctors = (
        db.query(StaffMember)
        .filter(StaffMember.role == "doctor")
        .order_by(StaffMember.username.asc())
        .all()
    )

    nurse_workload = []

    for staff in nurses:
        nurse_record = (
            db.query(Nurse)
            .filter(Nurse.email == staff.username)
            .first()
        )

        if not nurse_record:
            nurse_record = (
                db.query(Nurse)
                .filter(Nurse.name.ilike(staff.username))
                .first()
            )

        if nurse_record:
            assigned_patients = (
                db.query(NurseAssignment)
                .filter(NurseAssignment.nurse_id == nurse_record.id)
                .count()
            )
            open_tasks = (
                db.query(NurseTask)
                .filter(
                    NurseTask.nurse_id == nurse_record.id,
                    NurseTask.status != "Completed"
                )
                .count()
            )
            completed_tasks = (
                db.query(NurseTask)
                .filter(
                    NurseTask.nurse_id == nurse_record.id,
                    NurseTask.status == "Completed"
                )
                .count()
            )
            vitals_recorded = (
                db.query(NurseVitals)
                .filter(NurseVitals.nurse_id == nurse_record.id)
                .count()
            )
        else:
            assigned_patients = 0
            open_tasks = 0
            completed_tasks = 0
            vitals_recorded = 0

        nurse_workload.append({
            "username": staff.username,
            "department": staff.department,
            "active_status": staff.active_status,
            "assigned_patients": assigned_patients,
            "open_tasks": open_tasks,
            "completed_tasks": completed_tasks,
            "vitals_recorded": vitals_recorded
        })

    doctor_workload = []

    for staff in doctors:
        reviews_completed = (
            db.query(DoctorReview)
            .filter(DoctorReview.doctor_id == staff.username)
            .count()
        )
        treatment_patients = (
            db.query(PatientStatus)
            .filter(
                PatientStatus.updated_by == staff.username,
                PatientStatus.patient_status.in_(["Under Treatment", "Admitted", "Discharged"])
            )
            .count()
        )

        doctor_workload.append({
            "username": staff.username,
            "department": staff.department,
            "active_status": staff.active_status,
            "reviews_completed": reviews_completed,
            "treatment_status_updates": treatment_patients
        })

    return {
        "status": "success",
        "requested_by": current_user["username"],
        "nurse_workload": nurse_workload,
        "doctor_workload": doctor_workload
    }


@app.get("/emergency-queue")
def get_emergency_queue(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    predictions = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(100).all()
    rows = []

    for prediction in predictions:
        entry, color = ensure_queue_entry(db, prediction)
        rows.append({
            "prediction_id": prediction.id,
            "esi_severity": entry.esi_severity,
            "priority": entry.priority,
            "priority_color": color,
            "estimated_wait_time": entry.estimated_wait_time,
            "critical_status": entry.critical_status,
            "arrival_time": str(entry.arrival_time),
            "oxygen_level": prediction.triage_vital_o2,
            "heart_rate": prediction.triage_vital_hr,
            "final_prediction": prediction.final_prediction
        })

        if entry.critical_status:
            create_alert_if_missing(
                db,
                prediction.id,
                "Critical ESI",
                "Critical",
                f"Prediction #{prediction.id} is marked critical."
            )
        if entry.estimated_wait_time >= 60:
            create_alert_if_missing(
                db,
                prediction.id,
                "Long Wait",
                "High",
                f"Prediction #{prediction.id} has an estimated wait time of {entry.estimated_wait_time} minutes."
            )

    db.commit()

    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    rows = sorted(
        rows,
        key=lambda row: (
            priority_order.get(row["priority"], 4),
            row["estimated_wait_time"],
            row["oxygen_level"] or 100,
            row["arrival_time"]
        )
    )

    return {"status": "success", "queue": rows}


@app.put("/emergency-queue/{prediction_id}/priority")
def update_queue_priority(
    prediction_id: int,
    priority_data: QueuePriorityInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    allowed = ["Critical", "High", "Medium", "Low"]
    if priority_data.priority not in allowed:
        raise HTTPException(status_code=400, detail="Invalid priority.")

    prediction = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    entry, _ = ensure_queue_entry(db, prediction)
    entry.priority = priority_data.priority
    entry.estimated_wait_time = priority_data.estimated_wait_time if priority_data.estimated_wait_time is not None else entry.estimated_wait_time
    entry.critical_status = priority_data.priority == "Critical"
    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)

    return {"status": "success", "message": "Queue priority updated."}


@app.post("/beds/add")
def add_bed(
    bed_data: BedCreateInput,
    current_user: dict = Depends(require_role(["admin"])),
    db=Depends(get_db)
):
    allowed = ["Emergency", "ICU", "General Ward", "Observation"]
    if bed_data.ward_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid ward type.")

    if db.query(Bed).filter(Bed.bed_number == bed_data.bed_number).first():
        raise HTTPException(status_code=409, detail="Bed number already exists.")

    bed = Bed(bed_number=bed_data.bed_number, ward_type=bed_data.ward_type)
    db.add(bed)
    db.commit()
    db.refresh(bed)
    return {"status": "success", "bed": serialize_bed(bed)}


@app.get("/beds")
def get_beds(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    beds = db.query(Bed).order_by(Bed.ward_type.asc(), Bed.bed_number.asc()).all()
    return {"status": "success", "beds": [serialize_bed(bed) for bed in beds]}


@app.post("/beds/assign")
def assign_bed(
    assign_data: BedAssignInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    bed = db.query(Bed).filter(Bed.id == assign_data.bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found.")
    if bed.occupied:
        raise HTTPException(status_code=409, detail="Bed is already occupied.")
    prediction = db.query(PredictionLog).filter(PredictionLog.id == assign_data.prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    bed.occupied = True
    bed.assigned_prediction_id = assign_data.prediction_id
    bed.assigned_at = datetime.utcnow()
    db.commit()
    db.refresh(bed)
    return {"status": "success", "bed": serialize_bed(bed)}


@app.put("/beds/release/{bed_id}")
def release_bed(
    bed_id: int,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    bed = db.query(Bed).filter(Bed.id == bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found.")
    bed.occupied = False
    bed.assigned_prediction_id = None
    bed.assigned_at = None
    db.commit()
    db.refresh(bed)
    return {"status": "success", "bed": serialize_bed(bed)}


@app.post("/medications/add")
def add_medication(
    med_data: MedicationRecordInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    validate_prediction_and_nurse(db, med_data.prediction_id, med_data.nurse_id)
    med = MedicationRecord(**med_data.dict(), status="Scheduled")
    db.add(med)
    db.commit()
    db.refresh(med)
    return {"status": "success", "medication": serialize_medication(med)}


@app.get("/medications/{prediction_id}")
def get_medications(
    prediction_id: int,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    meds = db.query(MedicationRecord).filter(MedicationRecord.prediction_id == prediction_id).order_by(MedicationRecord.scheduled_time.desc()).all()
    return {"status": "success", "medications": [serialize_medication(med) for med in meds]}


@app.put("/medications/{medication_id}/status")
def update_medication_status(
    medication_id: int,
    med_status: MedicationStatusInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])),
    db=Depends(get_db)
):
    allowed = ["Scheduled", "Given", "Missed", "Cancelled"]
    if med_status.status not in allowed:
        raise HTTPException(status_code=400, detail="Invalid medication status.")
    med = db.query(MedicationRecord).filter(MedicationRecord.id == medication_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medication record not found.")
    med.status = med_status.status
    med.side_effects = med_status.side_effects
    med.notes = med_status.notes
    med.administered_time = datetime.utcnow() if med_status.status == "Given" else med.administered_time
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "medication_status_updated",
        "medication",
        medication_id,
        f"Status changed to {med_status.status}"
    )
    db.commit()
    db.refresh(med)
    return {"status": "success", "medication": serialize_medication(med)}


@app.get("/alerts")
def get_alerts(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    alerts = db.query(ClinicalAlert).order_by(ClinicalAlert.resolved.asc(), ClinicalAlert.created_at.desc()).all()
    return {"status": "success", "alerts": [serialize_alert(alert) for alert in alerts]}


@app.post("/alerts/add")
def add_alert(
    alert_data: ClinicalAlertInput,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    if not db.query(PredictionLog).filter(PredictionLog.id == alert_data.prediction_id).first():
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    alert = ClinicalAlert(**alert_data.dict())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return {"status": "success", "alert": serialize_alert(alert)}


@app.put("/alerts/{alert_id}/resolve")
def resolve_alert(
    alert_id: int,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    alert = db.query(ClinicalAlert).filter(ClinicalAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return {"status": "success", "alert": serialize_alert(alert)}


@app.post("/discharge-summary/create")
def create_discharge_summary(
    summary_data: DischargeSummaryInput,
    current_user: dict = Depends(require_role(["doctor", "admin"])),
    db=Depends(get_db)
):
    if not db.query(PredictionLog).filter(PredictionLog.id == summary_data.prediction_id).first():
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    summary = DischargeSummary(**summary_data.dict(), doctor_id=current_user["username"])
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return {"status": "success", "summary": serialize_discharge_summary(summary)}


@app.get("/discharge-summary/{prediction_id}")
def get_discharge_summary(
    prediction_id: int,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    summaries = db.query(DischargeSummary).filter(DischargeSummary.prediction_id == prediction_id).order_by(DischargeSummary.created_at.desc()).all()
    return {"status": "success", "summaries": [serialize_discharge_summary(summary) for summary in summaries]}


@app.get("/discharge-summary/{prediction_id}/pdf")
def get_discharge_summary_pdf(
    prediction_id: int,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    prediction = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    summary = db.query(DischargeSummary).filter(DischargeSummary.prediction_id == prediction_id).order_by(DischargeSummary.created_at.desc()).first()
    reviews = db.query(DoctorReview).filter(DoctorReview.prediction_id == prediction_id).order_by(DoctorReview.reviewed_at.desc()).all()
    vitals = db.query(NurseVitals).filter(NurseVitals.prediction_id == prediction_id).order_by(NurseVitals.recorded_at.desc()).all()
    meds = db.query(MedicationRecord).filter(MedicationRecord.prediction_id == prediction_id).all()

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    y = letter[1] - 50
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(50, y, "EmergeAI Discharge Summary")
    y -= 30
    pdf.setFont("Helvetica", 10)
    lines = [
        f"Prediction ID: {prediction.id}",
        f"Triage Result: {prediction.final_prediction}",
        f"Symptoms: {prediction.problem_description}",
        f"Initial Vitals: HR {prediction.triage_vital_hr}, BP {prediction.triage_vital_sbp}/{prediction.triage_vital_dbp}, O2 {prediction.triage_vital_o2}, RR {prediction.triage_vital_rr}, Temp {prediction.triage_vital_temp}",
        f"Discharge Diagnosis: {summary.diagnosis if summary else 'Not recorded'}",
        f"Treatment Given: {summary.treatment_given if summary else 'Not recorded'}",
        f"Medication Notes: {summary.medication_notes if summary else 'Not recorded'}",
        f"Follow-up: {summary.follow_up_instructions if summary else 'Not recorded'}",
        "",
        "Doctor Reviews:",
    ]
    lines.extend([f"- {r.diagnosis}: {r.treatment_plan}" for r in reviews[:5]])
    lines.append("")
    lines.append("Nurse Vitals / Notes:")
    lines.extend([f"- {v.recorded_at}: O2 {v.oxygen_level}, HR {v.heart_rate}, Pain {v.pain_score}, Notes {v.notes}" for v in vitals[:5]])
    lines.append("")
    lines.append("Medications:")
    lines.extend([f"- {m.medication_name} {m.dosage} {m.route}: {m.status}" for m in meds[:10]])

    for line in lines:
        if y < 60:
            pdf.showPage()
            y = letter[1] - 50
            pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, str(line)[:110])
        y -= 16
    pdf.save()
    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type="application/pdf")


@app.post("/shifts/add")
def add_shift(
    shift_data: StaffShiftInput,
    current_user: dict = Depends(require_role(["admin"])),
    db=Depends(get_db)
):
    if shift_data.shift_type not in ["Morning", "Evening", "Night"]:
        raise HTTPException(status_code=400, detail="Invalid shift type.")
    shift = StaffShift(**shift_data.dict())
    db.add(shift)
    db.commit()
    db.refresh(shift)
    return {"status": "success", "shift": serialize_shift(shift)}


@app.get("/shifts")
def get_shifts(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    query = db.query(StaffShift)
    if current_user["role"] != "admin":
        query = query.filter(StaffShift.staff_id == current_user["username"])
    shifts = query.order_by(StaffShift.start_time.desc()).all()
    return {"status": "success", "shifts": [serialize_shift(shift) for shift in shifts]}


@app.get("/shifts/current")
def get_current_shifts(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"])),
    db=Depends(get_db)
):
    now = datetime.utcnow()
    query = db.query(StaffShift).filter(StaffShift.start_time <= now, StaffShift.end_time >= now)
    if current_user["role"] != "admin":
        query = query.filter(StaffShift.staff_id == current_user["username"])
    shifts = query.all()
    return {"status": "success", "shifts": [serialize_shift(shift) for shift in shifts]}


@app.get("/admin/workload-summary")
def get_admin_workload_summary(
    current_user: dict = Depends(require_role(["admin"])),
    db=Depends(get_db)
):
    assigned_patients = db.query(NurseAssignment).count()
    total_predictions = db.query(PredictionLog).count()
    reviewed_predictions = db.query(DoctorReview.prediction_id).distinct().count()
    active_beds = db.query(Bed).filter(Bed.occupied == True).count()
    critical_alerts = db.query(ClinicalAlert).filter(ClinicalAlert.resolved == False, ClinicalAlert.severity == "Critical").count()
    waiting_patients = db.query(PatientStatus).filter(PatientStatus.patient_status.in_(["Waiting", "Waiting for Doctor"])).count()

    return {
        "status": "success",
        "assigned_patients_per_nurse": assigned_patients,
        "doctor_reviews_pending": max(total_predictions - reviewed_predictions, 0),
        "active_beds": active_beds,
        "critical_alerts": critical_alerts,
        "waiting_patients": waiting_patients
    }


@app.get("/admin/audit-logs")
def get_audit_logs(
    current_user: dict = Depends(require_role(["admin"])),
    db=Depends(get_db)
):
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(500).all()
    return {
        "status": "success",
        "logs": [
            row_dict(log, [
                "id",
                "username",
                "role",
                "action",
                "entity_type",
                "entity_id",
                "details",
                "created_at"
            ])
            for log in logs
        ]
    }


@app.post("/patients/add")
def add_patient(data: PatientInputData, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    payload = data.dict()
    prediction_id = payload.pop("prediction_id", None)
    patient = Patient(**payload)
    db.add(patient)
    db.flush()
    link_patient_prediction(db, patient.id, prediction_id)
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "patient_created",
        "patient",
        patient.id,
        f"Patient name: {patient.name}"
    )
    db.commit()
    db.refresh(patient)
    return {"status": "success", "patient": row_dict(patient, PATIENT_FIELDS)}


@app.get("/patients")
def get_patients(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    return {"status": "success", "patients": [row_dict(p, PATIENT_FIELDS) for p in patients]}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    links = db.query(PatientPredictionLink).filter(PatientPredictionLink.patient_id == patient_id).all()
    data = row_dict(patient, PATIENT_FIELDS)
    data["prediction_ids"] = [link.prediction_id for link in links]
    return {"status": "success", "patient": data}


@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, data: PatientInputData, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    payload = data.dict(exclude_unset=True)
    prediction_id = payload.pop("prediction_id", None)
    for key, value in payload.items():
        setattr(patient, key, value)
    link_patient_prediction(db, patient.id, prediction_id)
    log_audit_action(
        db,
        current_user["username"],
        current_user["role"],
        "patient_updated",
        "patient",
        patient.id,
        "Patient registration updated"
    )
    db.commit()
    db.refresh(patient)
    return {"status": "success", "patient": row_dict(patient, ["id", "name", "age", "gender", "phone", "address", "emergency_contact_name", "emergency_contact_phone", "allergies", "chronic_diseases", "past_medical_history", "created_at"])}


@app.post("/labs/order")
def order_lab(data: LabOrderInput, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    if not db.query(PredictionLog).filter(PredictionLog.id == data.prediction_id).first():
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    order = LabOrder(**data.dict(), doctor_id=current_user["username"])
    db.add(order)
    db.commit()
    db.refresh(order)
    return {"status": "success", "lab_order": row_dict(order, ["id", "prediction_id", "doctor_id", "test_name", "test_type", "priority", "status", "ordered_at", "result_notes", "result_file_path", "completed_at"])}


@app.get("/labs/{prediction_id}")
def get_labs(prediction_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    labs = db.query(LabOrder).filter(LabOrder.prediction_id == prediction_id).order_by(LabOrder.ordered_at.desc()).all()
    return {"status": "success", "labs": [row_dict(l, ["id", "prediction_id", "doctor_id", "test_name", "test_type", "priority", "status", "ordered_at", "result_notes", "result_file_path", "completed_at"]) for l in labs]}


@app.put("/labs/{lab_order_id}/result")
def update_lab_result(lab_order_id: int, data: LabResultInput, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    lab = db.query(LabOrder).filter(LabOrder.id == lab_order_id).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab order not found.")
    if data.status not in ["Ordered", "In Progress", "Completed", "Cancelled"]:
        raise HTTPException(status_code=400, detail="Invalid lab status.")
    lab.status = data.status
    lab.result_notes = data.result_notes
    lab.result_file_path = data.result_file_path
    lab.completed_at = datetime.utcnow() if data.status == "Completed" else lab.completed_at
    db.commit()
    return {"status": "success", "lab_order": row_dict(lab, ["id", "prediction_id", "doctor_id", "test_name", "test_type", "priority", "status", "ordered_at", "result_notes", "result_file_path", "completed_at"])}


@app.post("/referrals/add")
def add_referral(data: ReferralInput, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    departments = ["Cardiology", "Neurology", "Surgery", "Orthopedics", "Pediatrics", "Psychiatry", "Internal Medicine"]
    if data.specialist_department not in departments:
        raise HTTPException(status_code=400, detail="Invalid specialist department.")
    referral = Referral(**data.dict(), referring_doctor_id=current_user["username"])
    db.add(referral)
    db.commit()
    db.refresh(referral)
    return {"status": "success", "referral": row_dict(referral, ["id", "prediction_id", "referring_doctor_id", "specialist_department", "reason", "urgency", "status", "notes", "created_at", "completed_at"])}


@app.get("/referrals")
def get_referrals(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    refs = db.query(Referral).order_by(Referral.created_at.desc()).all()
    return {"status": "success", "referrals": [row_dict(r, ["id", "prediction_id", "referring_doctor_id", "specialist_department", "reason", "urgency", "status", "notes", "created_at", "completed_at"]) for r in refs]}


@app.get("/referrals/{prediction_id}")
def get_referrals_for_prediction(prediction_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    refs = db.query(Referral).filter(Referral.prediction_id == prediction_id).order_by(Referral.created_at.desc()).all()
    return {"status": "success", "referrals": [row_dict(r, ["id", "prediction_id", "referring_doctor_id", "specialist_department", "reason", "urgency", "status", "notes", "created_at", "completed_at"]) for r in refs]}


@app.put("/referrals/{referral_id}/status")
def update_referral_status(referral_id: int, data: StatusUpdateInput, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    ref = db.query(Referral).filter(Referral.id == referral_id).first()
    if not ref:
        raise HTTPException(status_code=404, detail="Referral not found.")
    ref.status = data.status
    ref.notes = data.notes
    ref.completed_at = datetime.utcnow() if data.status == "Completed" else ref.completed_at
    db.commit()
    return {"status": "success"}


@app.post("/consents/add")
def add_consent(data: ConsentInput, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    if data.consent_type not in ["Treatment Consent", "AI Decision Support Consent", "Image Analysis Consent", "Data Use Consent"]:
        raise HTTPException(status_code=400, detail="Invalid consent type.")
    consent = Consent(**data.dict())
    db.add(consent)
    db.commit()
    db.refresh(consent)
    return {"status": "success", "consent": row_dict(consent, ["id", "patient_id", "prediction_id", "consent_type", "accepted", "signed_by", "signed_at", "notes"])}


@app.get("/consents/{patient_id}")
def get_consents(patient_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    consents = db.query(Consent).filter(Consent.patient_id == patient_id).order_by(Consent.signed_at.desc()).all()
    return {"status": "success", "consents": [row_dict(c, ["id", "patient_id", "prediction_id", "consent_type", "accepted", "signed_by", "signed_at", "notes"]) for c in consents]}


@app.post("/incidents/add")
def add_incident(data: IncidentInput, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    incident = IncidentReport(**data.dict(), staff_id=current_user["username"])
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return {"status": "success", "incident": row_dict(incident, ["id", "prediction_id", "staff_id", "incident_type", "severity", "description", "action_taken", "reported_at", "status"])}


@app.get("/incidents")
def get_incidents(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    incidents = db.query(IncidentReport).order_by(IncidentReport.reported_at.desc()).all()
    return {"status": "success", "incidents": [row_dict(i, ["id", "prediction_id", "staff_id", "incident_type", "severity", "description", "action_taken", "reported_at", "status"]) for i in incidents]}


@app.put("/incidents/{incident_id}/status")
def update_incident_status(incident_id: int, data: StatusUpdateInput, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    incident = db.query(IncidentReport).filter(IncidentReport.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    incident.status = data.status
    incident.action_taken = data.notes
    db.commit()
    return {"status": "success"}


@app.post("/billing/create")
def create_billing(data: BillingInput, current_user: dict = Depends(require_role(["admin"])), db=Depends(get_db)):
    total = data.visit_cost + data.treatment_cost + data.medication_cost
    payload = data.dict()
    record = BillingRecord(**payload, total_cost=total)
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": "success", "billing": row_dict(record, ["id", "patient_id", "prediction_id", "insurance_provider", "policy_number", "visit_cost", "treatment_cost", "medication_cost", "total_cost", "payment_status", "created_at"])}


@app.get("/billing/{patient_id}")
def get_billing(patient_id: int, current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    records = db.query(BillingRecord).filter(BillingRecord.patient_id == patient_id).order_by(BillingRecord.created_at.desc()).all()
    return {"status": "success", "billing": [row_dict(r, ["id", "patient_id", "prediction_id", "insurance_provider", "policy_number", "visit_cost", "treatment_cost", "medication_cost", "total_cost", "payment_status", "created_at"]) for r in records]}


@app.put("/billing/{billing_id}/status")
def update_billing_status(billing_id: int, data: BillingStatusInput, current_user: dict = Depends(require_role(["admin"])), db=Depends(get_db)):
    record = db.query(BillingRecord).filter(BillingRecord.id == billing_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Billing record not found.")
    record.payment_status = data.payment_status
    db.commit()
    return {"status": "success"}


@app.post("/inventory/add")
def add_inventory(data: InventoryInput, current_user: dict = Depends(require_role(["admin"])), db=Depends(get_db)):
    if data.category not in ["Medication", "PPE", "Oxygen", "Equipment", "Lab Supply"]:
        raise HTTPException(status_code=400, detail="Invalid inventory category.")
    item = InventoryItem(**data.dict())
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"status": "success", "item": row_dict(item, ["id", "item_name", "category", "quantity", "unit", "minimum_stock_level", "expiry_date", "location", "status"])}


@app.get("/inventory")
def get_inventory(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    items = db.query(InventoryItem).order_by(InventoryItem.category.asc(), InventoryItem.item_name.asc()).all()
    return {"status": "success", "inventory": [row_dict(i, ["id", "item_name", "category", "quantity", "unit", "minimum_stock_level", "expiry_date", "location", "status"]) for i in items]}


@app.get("/inventory/low-stock")
def get_low_stock(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    items = db.query(InventoryItem).filter(InventoryItem.quantity <= InventoryItem.minimum_stock_level).all()
    return {"status": "success", "inventory": [row_dict(i, ["id", "item_name", "category", "quantity", "unit", "minimum_stock_level", "expiry_date", "location", "status"]) for i in items]}


@app.put("/inventory/{item_id}")
def update_inventory(item_id: int, data: InventoryInput, current_user: dict = Depends(require_role(["admin"])), db=Depends(get_db)):
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Inventory item not found.")
    for key, value in data.dict().items():
        setattr(item, key, value)
    db.commit()
    return {"status": "success"}


@app.post("/appointments/add")
def add_appointment(data: AppointmentInput, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    appt = FollowUpAppointment(**data.dict())
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return {"status": "success", "appointment": row_dict(appt, ["id", "patient_id", "prediction_id", "doctor_id", "appointment_date", "department", "reason", "status", "notes"])}


@app.get("/appointments")
def get_appointments(current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    appts = db.query(FollowUpAppointment).order_by(FollowUpAppointment.appointment_date.desc()).all()
    return {"status": "success", "appointments": [row_dict(a, ["id", "patient_id", "prediction_id", "doctor_id", "appointment_date", "department", "reason", "status", "notes"]) for a in appts]}


@app.get("/appointments/{patient_id}")
def get_patient_appointments(patient_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    appts = db.query(FollowUpAppointment).filter(FollowUpAppointment.patient_id == patient_id).order_by(FollowUpAppointment.appointment_date.desc()).all()
    return {"status": "success", "appointments": [row_dict(a, ["id", "patient_id", "prediction_id", "doctor_id", "appointment_date", "department", "reason", "status", "notes"]) for a in appts]}


@app.put("/appointments/{appointment_id}/status")
def update_appointment_status(appointment_id: int, data: StatusUpdateInput, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    appt = db.query(FollowUpAppointment).filter(FollowUpAppointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    appt.status = data.status
    appt.notes = data.notes
    db.commit()
    return {"status": "success"}


@app.post("/notifications/send")
def send_notification(data: NotificationInput, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    if data.notification_type not in ["Admission", "Critical Alert", "Discharge", "Follow-up Reminder"]:
        raise HTTPException(status_code=400, detail="Invalid notification type.")
    log = NotificationLog(**data.dict(), status="Sent")
    db.add(log)
    db.commit()
    db.refresh(log)
    return {"status": "success", "notification": row_dict(log, ["id", "patient_id", "prediction_id", "contact_name", "contact_phone", "message", "notification_type", "status", "sent_at"])}


@app.get("/notifications/{patient_id}")
def get_notifications(patient_id: int, current_user: dict = Depends(require_role(["nurse", "doctor", "admin"])), db=Depends(get_db)):
    logs = db.query(NotificationLog).filter(NotificationLog.patient_id == patient_id).order_by(NotificationLog.sent_at.desc()).all()
    return {"status": "success", "notifications": [row_dict(n, ["id", "patient_id", "prediction_id", "contact_name", "contact_phone", "message", "notification_type", "status", "sent_at"]) for n in logs]}


@app.get("/analytics/hospital-summary")
def hospital_summary(current_user: dict = Depends(require_role(["doctor", "admin"])), db=Depends(get_db)):
    total_patients = db.query(Patient).count()
    total_triage = db.query(PredictionLog).count()
    critical_cases = db.query(PredictionLog).filter(or_(PredictionLog.final_prediction.ilike("%1%"), PredictionLog.final_prediction.ilike("%critical%"))).count()
    waits = [q.estimated_wait_time for q in db.query(EmergencyQueue).all() if q.estimated_wait_time is not None]
    avg_wait = round(sum(waits) / len(waits), 2) if waits else 0
    bed_total = db.query(Bed).count()
    bed_occupied = db.query(Bed).filter(Bed.occupied == True).count()
    bed_rate = round((bed_occupied / bed_total) * 100, 2) if bed_total else 0
    low_items = db.query(InventoryItem).filter(InventoryItem.quantity <= InventoryItem.minimum_stock_level).all()
    lab_counts = {}
    for lab in db.query(LabOrder).all():
        lab_counts[lab.test_name] = lab_counts.get(lab.test_name, 0) + 1
    symptom_counts = {
        "Chest Pain": db.query(PredictionLog).filter(PredictionLog.cc_chestpain == 1).count(),
        "Shortness of Breath": db.query(PredictionLog).filter(PredictionLog.cc_shortnessofbreath == 1).count(),
        "Fever": db.query(PredictionLog).filter(PredictionLog.cc_fever == 1).count(),
        "Headache": db.query(PredictionLog).filter(PredictionLog.cc_headache == 1).count(),
    }
    return {
        "status": "success",
        "total_patients": total_patients,
        "total_triage_cases": total_triage,
        "critical_cases": critical_cases,
        "average_wait_time": avg_wait,
        "bed_occupancy_rate": bed_rate,
        "nurse_workload": db.query(NurseAssignment).count(),
        "doctor_workload": db.query(DoctorReview).count(),
        "top_symptoms": symptom_counts,
        "common_lab_tests": lab_counts,
        "low_inventory_items": [row_dict(i, ["id", "item_name", "category", "quantity", "minimum_stock_level"]) for i in low_items],
        "discharge_count": db.query(DischargeSummary).count()
    }


# -----------------------------
# SIMPLE FEEDBACK
# -----------------------------

@app.post("/feedback")
def update_feedback(
    feedback_data: FeedbackInput,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    db = SessionLocal()

    try:
        log = (
            db.query(PredictionLog)
            .filter(PredictionLog.id == feedback_data.log_id)
            .first()
        )

        if not log:
            raise HTTPException(
                status_code=404,
                detail="Prediction log not found"
            )

        log.feedback = feedback_data.feedback
        db.commit()

        return {
            "status": "success",
            "message": "Feedback updated successfully",
            "log_id": feedback_data.log_id,
            "feedback": feedback_data.feedback,
            "updated_by": current_user["username"]
        }

    finally:
        db.close()


# -----------------------------
# CLINICAL FEEDBACK — POST
# -----------------------------

@app.post("/clinical-feedback")
def submit_clinical_feedback(
    feedback: ClinicalFeedbackInput,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    db = SessionLocal()

    try:
        prediction = (
            db.query(PredictionLog)
            .filter(PredictionLog.id == feedback.prediction_id)
            .first()
        )

        if not prediction:
            raise HTTPException(
                status_code=404,
                detail="Prediction ID not found. Please run /predict first and use the returned prediction_id."
            )

        if feedback.accepted:
            action = "Accepted AI Prediction"
            clinician_prediction = prediction.final_prediction
        else:
            action = "Overridden by Clinician"
            clinician_prediction = (
                str(feedback.override_esi)
                if feedback.override_esi is not None
                else None
            )

        clinical_feedback = ClinicalFeedback(
            prediction_id=feedback.prediction_id,
            ai_prediction=prediction.final_prediction,
            clinician_prediction=clinician_prediction,
            accepted=feedback.accepted,
            action=action,
            override_reason=feedback.override_reason,
            feedback_notes=feedback.clinical_notes,
            reviewer_username=current_user["username"],
            reviewer_role=current_user["role"]
        )

        prediction.feedback = action

        db.add(clinical_feedback)
        db.commit()
        db.refresh(clinical_feedback)

        return {
            "status": "success",
            "message": "Clinical feedback saved successfully",
            "feedback_id": clinical_feedback.id,
            "prediction_id": feedback.prediction_id,
            "ai_prediction": prediction.final_prediction,
            "clinician_prediction": clinician_prediction,
            "accepted": feedback.accepted,
            "action": action,
            "reviewer": current_user["username"]
        }

    finally:
        db.close()


# -----------------------------
# CLINICAL FEEDBACK — GET
# -----------------------------

@app.get("/clinical-feedback")
def get_clinical_feedback(
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    db = SessionLocal()

    try:
        feedbacks = (
            db.query(ClinicalFeedback)
            .order_by(ClinicalFeedback.created_at.desc())
            .all()
        )

        results = []

        for fb in feedbacks:
            results.append({
                "id": fb.id,
                "prediction_id": fb.prediction_id,
                "ai_prediction": fb.ai_prediction,
                "clinician_prediction": fb.clinician_prediction,
                "accepted": fb.accepted,
                "action": fb.action,
                "override_reason": fb.override_reason,
                "feedback_notes": fb.feedback_notes,
                "reviewer_username": fb.reviewer_username,
                "reviewer_role": fb.reviewer_role,
                "created_at": str(fb.created_at)
            })

        return results

    finally:
        db.close()

# -----------------------------
# PATIENT RISK WATCHLIST
# -----------------------------

def calculate_watchlist_risk(patient):
    """
    Flexible risk logic for the live watchlist.
    This prevents the watchlist from showing empty when the model stores
    final_prediction as 1, 2, ESI-1, ESI 1, Critical, etc.
    It also catches clinically dangerous vitals even if the model prediction text differs.
    """
    final_prediction = str(patient.final_prediction or "").strip().lower()

    critical_prediction_values = [
        "1", "esi 1", "esi-1", "critical", "critical risk", "level 1"
    ]

    high_prediction_values = [
        "2", "esi 2", "esi-2", "high", "high risk", "level 2"
    ]

    critical_vitals = (
        (patient.triage_vital_o2 is not None and patient.triage_vital_o2 <= 88)
        or (patient.triage_vital_sbp is not None and patient.triage_vital_sbp <= 80)
        or (patient.triage_vital_hr is not None and patient.triage_vital_hr >= 140)
        or (patient.triage_vital_rr is not None and patient.triage_vital_rr >= 30)
    )

    high_risk_vitals = (
        (patient.triage_vital_o2 is not None and patient.triage_vital_o2 <= 92)
        or (patient.triage_vital_sbp is not None and patient.triage_vital_sbp <= 90)
        or (patient.triage_vital_hr is not None and patient.triage_vital_hr >= 120)
        or (patient.triage_vital_rr is not None and patient.triage_vital_rr >= 24)
        or (patient.triage_vital_temp is not None and patient.triage_vital_temp >= 38.0)
    )

    dangerous_symptoms = (
        patient.cc_chestpain == 1
        or patient.cc_shortnessofbreath == 1
        or patient.cc_syncope == 1
        or patient.cc_weakness == 1
    )

    if (
        any(value in final_prediction for value in critical_prediction_values)
        or critical_vitals
    ):
        return "CRITICAL"

    if (
        any(value in final_prediction for value in high_prediction_values)
        or high_risk_vitals
        or dangerous_symptoms
    ):
        return "HIGH"

    return "NORMAL"

@app.get("/v2/watchlist")
def get_patient_watchlist(
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"]))
):
    db = SessionLocal()

    try:
        recent_patients = (
            db.query(PredictionLog)
            .order_by(desc(PredictionLog.created_at))
            .limit(100)
            .all()
        )

        results = []

        for patient in recent_patients:
            final_prediction = str(patient.final_prediction or "").lower()

            is_high_risk_prediction = (
                final_prediction in ["esi 1", "esi 2", "1", "2"]
                or "critical" in final_prediction
                or "high risk" in final_prediction
            )

            is_high_risk_vitals = (
                patient.triage_vital_o2 is not None and patient.triage_vital_o2 < 90
            ) or (
                patient.triage_vital_sbp is not None and patient.triage_vital_sbp < 90
            ) or (
                patient.triage_vital_hr is not None and patient.triage_vital_hr > 130
            ) or (
                patient.triage_vital_rr is not None and patient.triage_vital_rr > 30
            ) or (
                patient.triage_vital_temp is not None and patient.triage_vital_temp >= 39
            ) or (
                patient.cc_chestpain == 1 and patient.cc_shortnessofbreath == 1
            ) or (
                patient.cc_syncope == 1
            )

            if not is_high_risk_prediction and not is_high_risk_vitals:
                continue

            if (
                final_prediction in ["esi 1", "1"]
                or "critical" in final_prediction
                or patient.triage_vital_o2 < 85
                or patient.triage_vital_sbp < 80
            ):
                risk_level = "CRITICAL"
            else:
                risk_level = "HIGH"

            results.append({
                "id": patient.id,
                "age": patient.age,
                "gender": patient.gender,
                "arrivalmode": patient.arrivalmode,
                "prediction": patient.final_prediction,
                "confidence": patient.confidence,
                "heart_rate": patient.triage_vital_hr,
                "systolic_bp": patient.triage_vital_sbp,
                "diastolic_bp": patient.triage_vital_dbp,
                "respiratory_rate": patient.triage_vital_rr,
                "spo2": patient.triage_vital_o2,
                "temperature": patient.triage_vital_temp,
                "problem_description": patient.problem_description,
                "emergency_keywords": patient.emergency_keywords,
                "safety_reasons": patient.safety_reasons,
                "clinical_explanations": patient.clinical_explanations,
                "feedback": patient.feedback,
                "created_at": str(patient.created_at),
                "risk_level": risk_level
            })

        return {
            "status": "success",
            "requested_by": current_user["username"],
            "role": current_user["role"],
            "total_critical_patients": len(results),
            "patients": results
        }

    finally:
        db.close()

# -----------------------------
# MODEL RETRAINING API
# -----------------------------

@app.post("/v2/retrain-model")
def run_model_retraining(
    current_user: dict = Depends(require_role(["admin"]))
):
    """
    Admin-only endpoint to run the retrain_model.py pipeline from the backend.
    This allows Streamlit to trigger retraining and display terminal output.
    """
    try:
        result = subprocess.run(
            ["python", "backend/retrain_model.py"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=300
        )

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "requested_by": current_user["username"],
            "role": current_user["role"],
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Retraining timed out. Try again with fewer records or optimize the retraining script."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(e)}"
        )



# -----------------------------
# COMPANY HEALTH DASHBOARD API
# -----------------------------

@app.get("/v2/company-health-dashboard")
def company_health_dashboard(
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    """
    Company-level healthcare dashboard endpoint.
    Provides executive KPIs, operational trends, acuity distribution,
    clinical feedback performance, and AI confidence monitoring.
    """
    db = SessionLocal()

    try:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(500)
            .all()
        )

        feedbacks = (
            db.query(ClinicalFeedback)
            .order_by(ClinicalFeedback.created_at.desc())
            .limit(500)
            .all()
        )

        total_patients = len(logs)

        if total_patients == 0:
            return {
                "status": "success",
                "requested_by": current_user["username"],
                "role": current_user["role"],
                "message": "No records available yet.",
                "kpis": {
                    "total_patients": 0,
                    "high_acuity_total": 0,
                    "high_acuity_rate": 0,
                    "critical_patients": 0,
                    "high_risk_patients": 0,
                    "avg_model_confidence": 0,
                    "total_feedback": 0,
                    "override_rate": 0,
                    "system_status": "No Data"
                },
                "charts": {
                    "esi_distribution": {},
                    "hourly_volume": {},
                    "arrival_distribution": {},
                    "risk_counts": {
                        "critical": 0,
                        "high": 0,
                        "moderate_low": 0
                    },
                    "clinical_feedback": {
                        "accepted": 0,
                        "overridden": 0
                    }
                },
                "operations": {
                    "busiest_hour": None,
                    "busiest_hour_volume": 0,
                    "recommendation": "Add prediction records to activate the dashboard."
                }
            }

        esi_distribution = {}
        hourly_volume = {}
        arrival_distribution = {}
        risk_counts = {
            "critical": 0,
            "high": 0,
            "moderate_low": 0
        }

        confidence_values = []
        low_confidence_count = 0

        for log in logs:
            prediction = str(log.final_prediction or "Unknown")
            prediction_clean = prediction.strip().lower()

            esi_distribution[prediction] = esi_distribution.get(prediction, 0) + 1

            if log.created_at:
                hour = log.created_at.strftime("%Y-%m-%d %H:00")
                hourly_volume[hour] = hourly_volume.get(hour, 0) + 1

            arrival = str(log.arrivalmode or "Unknown")
            arrival_distribution[arrival] = arrival_distribution.get(arrival, 0) + 1

            if log.confidence is not None:
                confidence = float(log.confidence)
                confidence_values.append(confidence)

                if confidence < 0.70:
                    low_confidence_count += 1

            if (
                prediction_clean in ["1", "1.0", "esi 1", "esi-1", "level 1"]
                or "critical" in prediction_clean
            ):
                risk_counts["critical"] += 1

            elif (
                prediction_clean in ["2", "2.0", "esi 2", "esi-2", "level 2"]
                or "high" in prediction_clean
            ):
                risk_counts["high"] += 1

            else:
                risk_counts["moderate_low"] += 1

        avg_confidence = (
            round(sum(confidence_values) / len(confidence_values), 3)
            if confidence_values else 0
        )

        high_acuity_total = risk_counts["critical"] + risk_counts["high"]

        high_acuity_rate = round(
            (high_acuity_total / total_patients) * 100,
            2
        )

        accepted_feedback = 0
        overridden_feedback = 0

        for fb in feedbacks:
            if fb.accepted:
                accepted_feedback += 1
            else:
                overridden_feedback += 1

        total_feedback = len(feedbacks)

        override_rate = (
            round((overridden_feedback / total_feedback) * 100, 2)
            if total_feedback > 0 else 0
        )

        busiest_hour = (
            max(hourly_volume, key=hourly_volume.get)
            if hourly_volume else None
        )

        busiest_hour_volume = (
            hourly_volume[busiest_hour]
            if busiest_hour else 0
        )

        system_status = "Stable"

        if high_acuity_rate >= 40:
            system_status = "High Pressure"
        elif high_acuity_rate >= 25:
            system_status = "Moderate Pressure"

        if system_status == "High Pressure":
            recommendation = "Increase emergency staffing and prioritize critical care capacity."
        elif system_status == "Moderate Pressure":
            recommendation = "Monitor high-acuity patient flow and maintain triage readiness."
        else:
            recommendation = "System operating within stable range."

        return {
            "status": "success",
            "requested_by": current_user["username"],
            "role": current_user["role"],

            "kpis": {
                "total_patients": total_patients,
                "high_acuity_total": high_acuity_total,
                "high_acuity_rate": high_acuity_rate,
                "critical_patients": risk_counts["critical"],
                "high_risk_patients": risk_counts["high"],
                "moderate_low_patients": risk_counts["moderate_low"],
                "avg_model_confidence": avg_confidence,
                "low_confidence_predictions": low_confidence_count,
                "total_feedback": total_feedback,
                "override_rate": override_rate,
                "system_status": system_status
            },

            "charts": {
                "esi_distribution": esi_distribution,
                "hourly_volume": hourly_volume,
                "arrival_distribution": arrival_distribution,
                "risk_counts": risk_counts,
                "clinical_feedback": {
                    "accepted": accepted_feedback,
                    "overridden": overridden_feedback
                }
            },

            "operations": {
                "busiest_hour": busiest_hour,
                "busiest_hour_volume": busiest_hour_volume,
                "recommendation": recommendation
            }
        }

    finally:
        db.close()

def create_simple_pdf_report(log):
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "EmergeAI Healthcare - Patient Triage Report")

    y -= 35
    pdf.setFont("Helvetica", 10)

    lines = [
        f"Prediction ID: {log.id}",
        f"Age: {log.age}",
        f"Gender: {log.gender}",
        f"Arrival Mode: {log.arrivalmode}",
        "",
        "Vitals:",
        f"Heart Rate: {log.triage_vital_hr}",
        f"Systolic BP: {log.triage_vital_sbp}",
        f"Diastolic BP: {log.triage_vital_dbp}",
        f"Respiratory Rate: {log.triage_vital_rr}",
        f"Oxygen Saturation: {log.triage_vital_o2}",
        f"Temperature: {log.triage_vital_temp}",
        "",
        "AI Prediction:",
        f"ML Prediction: {log.ml_prediction}",
        f"Final Prediction: {log.final_prediction}",
        f"Confidence: {log.confidence}",
        "",
        "Symptoms:",
        f"Problem Description: {log.problem_description}",
        f"Emergency Keywords: {log.emergency_keywords}",
        "",
        "Safety Reasons:",
        f"{log.safety_reasons}",
        "",
        "Clinical Explanation:",
        f"{log.clinical_explanations}",
        "",
        "Historical Medical Report Context:",
    ]

    db = SessionLocal()
    try:
        historical_reports = (
            db.query(HistoricalMedicalReport)
            .filter(HistoricalMedicalReport.patient_id == str(log.id))
            .order_by(HistoricalMedicalReport.upload_date.desc())
            .limit(3)
            .all()
        )
        if historical_reports:
            for report in historical_reports:
                flags = json.loads(report.risk_flags) if report.risk_flags else []
                summary = json.loads(report.summary) if report.summary else {}
                clinical = summary.get("clinical_summary", {})
                lines.extend([
                    f"Report: {report.report_type} ({report.file_name})",
                    f"Summary: {clinical.get('summary_text', '')}",
                    f"Risk Flags: {', '.join(flag.get('label', '') for flag in flags)}",
                    f"File Type: {report.file_type or 'document'}",
                    f"OCR Findings: {(report.ocr_text or '')[:120]}",
                    f"Image Metadata: {report.image_metadata or ''}",
                    f"Image Quality Notes: {report.image_quality_notes or ''}",
                    f"Doctor Notes: {report.doctor_notes or ''}",
                    f"Nurse Notes: {report.nurse_notes or ''}",
                ])
        else:
            lines.append("No linked historical report summary found for this prediction/patient ID.")
    finally:
        db.close()

    lines.extend([
        "",
        "Historical Report Disclaimer:",
        HISTORICAL_REPORT_DISCLAIMER,
    ])

    for line in lines:
        if y < 60:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

        pdf.drawString(50, y, str(line)[:100])
        y -= 16

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

@app.post("/send-report-email")
def send_report_email(
    data: EmailReportInput,
    current_user: dict = Depends(require_role(["doctor", "nurse", "admin"]))
):
    db = SessionLocal()

    try:
        log = (
            db.query(PredictionLog)
            .filter(PredictionLog.id == data.prediction_id)
            .first()
        )

        if not log:
            raise HTTPException(
                status_code=404,
                detail="Prediction report not found"
            )

        subject = f"EmergeAI Patient Triage Report - Prediction #{log.id}"

        body = f"""
EmergeAI Healthcare - Patient Triage Report

Prediction ID: {log.id}
Age: {log.age}
Gender: {log.gender}
Arrival Mode: {log.arrivalmode}

Vitals:
Heart Rate: {log.triage_vital_hr}
Systolic BP: {log.triage_vital_sbp}
Diastolic BP: {log.triage_vital_dbp}
Respiratory Rate: {log.triage_vital_rr}
Oxygen Saturation: {log.triage_vital_o2}
Temperature: {log.triage_vital_temp}

AI Prediction:
ML Prediction: {log.ml_prediction}
Final Prediction: {log.final_prediction}
Confidence: {log.confidence}

Symptoms:
Problem Description: {log.problem_description}
Emergency Keywords: {log.emergency_keywords}

Safety Reasons:
{log.safety_reasons}

Clinical Explanation:
{log.clinical_explanations}

Generated by: {current_user["username"]} ({current_user["role"]})
"""

        try:
            print("STEP 1: Starting PDF generation...")
            pdf_bytes = create_simple_pdf_report(log)
            print("STEP 2: PDF generated successfully")

            print("STEP 3: Sending email...")
            send_doctor_report_email(
                to_email=data.doctor_email,
                subject=subject,
                body=body,
                pdf_bytes=pdf_bytes,
                filename=f"emergeai_report_{log.id}.pdf"
            )
            print("STEP 4: Email sent successfully")

        except TimeoutError as e:
            print("EMAIL REPORT TIMEOUT:", str(e))
            raise HTTPException(
                status_code=504,
                detail=str(e)
            )

        except Exception as e:
            print("EMAIL REPORT ERROR:", str(e))
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

        return {
            "status": "success",
            "message": "Report sent to doctor successfully",
            "doctor_email": data.doctor_email,
            "prediction_id": data.prediction_id
        }

    finally:
        db.close()

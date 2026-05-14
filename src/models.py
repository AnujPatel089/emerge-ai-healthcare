from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    Boolean,
    ForeignKey
)

from sqlalchemy.orm import relationship
from datetime import datetime

from src.database import Base


# =====================================================
# PREDICTION LOG TABLE
# =====================================================

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    # -------------------------------------------------
    # PATIENT DEMOGRAPHICS
    # -------------------------------------------------

    age = Column(Integer)
    gender = Column(String)
    race = Column(String)
    ethnicity = Column(String)
    arrivalmode = Column(String)

    # -------------------------------------------------
    # VITAL SIGNS
    # -------------------------------------------------

    triage_vital_hr = Column(Integer)
    triage_vital_sbp = Column(Integer)
    triage_vital_dbp = Column(Integer)
    triage_vital_rr = Column(Integer)
    triage_vital_o2 = Column(Integer)
    triage_vital_temp = Column(Float)

    # -------------------------------------------------
    # SYMPTOMS
    # -------------------------------------------------

    cc_chestpain = Column(Integer)
    cc_shortnessofbreath = Column(Integer)
    cc_headache = Column(Integer)
    cc_fever = Column(Integer)
    cc_abdominalpain = Column(Integer)
    cc_dizziness = Column(Integer)
    cc_syncope = Column(Integer)
    cc_weakness = Column(Integer)

    # -------------------------------------------------
    # AI PREDICTIONS
    # -------------------------------------------------

    ml_prediction = Column(String)
    final_prediction = Column(String)

    confidence = Column(Float, nullable=True)

    # -------------------------------------------------
    # EXPLANATIONS
    # -------------------------------------------------

    safety_reasons = Column(Text)
    clinical_explanations = Column(Text)

    # -------------------------------------------------
    # NLP / LLM FIELDS
    # -------------------------------------------------

    problem_description = Column(Text, nullable=True)

    llm_ready_text = Column(Text, nullable=True)

    matched_terms = Column(Text, nullable=True)

    emergency_keywords = Column(Text, nullable=True)

    # -------------------------------------------------
    # COMPUTER VISION
    # -------------------------------------------------

    cv_analysis = Column(Text, nullable=True)

    # -------------------------------------------------
    # AUDIT INFO
    # -------------------------------------------------

    source = Column(String, default="API Prediction")

    feedback = Column(String, default="Pending")

    assigned_nurse_id = Column(Integer, nullable=True, index=True)
    assigned_nurse_name = Column(String, nullable=True)
    assigned_doctor_id = Column(String, nullable=True, index=True)
    assigned_doctor_name = Column(String, nullable=True)
    assignment_status = Column(String, default="waiting", index=True)
    assigned_at = Column(DateTime, nullable=True)
    triage_started_at = Column(DateTime, nullable=True)
    doctor_review_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # -------------------------------------------------
    # RELATIONSHIP
    # -------------------------------------------------

    clinical_feedback = relationship(
        "ClinicalFeedback",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    nurse_assignments = relationship(
        "NurseAssignment",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    nurse_vitals = relationship(
        "NurseVitals",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    nurse_tasks = relationship(
        "NurseTask",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    doctor_reviews = relationship(
        "DoctorReview",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    status_timeline = relationship(
        "PatientStatus",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    queue_entries = relationship(
        "EmergencyQueue",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    bed_assignments = relationship(
        "Bed",
        back_populates="prediction"
    )

    medication_records = relationship(
        "MedicationRecord",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    clinical_alerts = relationship(
        "ClinicalAlert",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )

    discharge_summaries = relationship(
        "DischargeSummary",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )


# =====================================================
# REGISTERED APPLICATION USERS
# =====================================================

class AppUser(Base):
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    username = Column(String, nullable=False, unique=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    requested_role = Column(String, nullable=False, default="patient")
    role = Column(String, nullable=False, default="patient")
    status = Column(String, nullable=False, default="active")
    account_status = Column(String, nullable=False, default="pending")
    approval_level = Column(String, nullable=True)
    rejected_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_by = Column(String, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    rejected_by = Column(String, nullable=True)
    rejected_at = Column(DateTime, nullable=True)


# =====================================================
# CLINICAL FEEDBACK TABLE
# =====================================================

class ClinicalFeedback(Base):
    __tablename__ = "clinical_feedback"

    id = Column(Integer, primary_key=True, index=True)

    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False
    )

    # -------------------------------------------------
    # AI VS CLINICIAN
    # -------------------------------------------------

    ai_prediction = Column(String, nullable=False)

    clinician_prediction = Column(String, nullable=True)

    accepted = Column(Boolean, default=False)

    # -------------------------------------------------
    # REVIEW ACTIONS
    # -------------------------------------------------

    action = Column(String, nullable=False)

    override_reason = Column(Text, nullable=True)

    feedback_notes = Column(Text, nullable=True)

    # -------------------------------------------------
    # REVIEWER INFO
    # -------------------------------------------------

    reviewer_username = Column(String, nullable=False)

    reviewer_role = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # -------------------------------------------------
    # RELATIONSHIP
    # -------------------------------------------------

    prediction = relationship(
        "PredictionLog",
        back_populates="clinical_feedback"
    )


# =====================================================
# NURSE TABLE
# =====================================================

class Nurse(Base):
    __tablename__ = "nurses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    department = Column(String, nullable=False)
    available_status = Column(Boolean, default=True)
    active_patient_count = Column(Integer, default=0)
    experience_level = Column(String, default="normal")

    assignments = relationship(
        "NurseAssignment",
        back_populates="nurse",
        cascade="all, delete-orphan"
    )

    vitals = relationship(
        "NurseVitals",
        back_populates="nurse",
        cascade="all, delete-orphan"
    )

    tasks = relationship(
        "NurseTask",
        back_populates="nurse",
        cascade="all, delete-orphan"
    )


# =====================================================
# NURSE ASSIGNMENT TABLE
# =====================================================

class NurseAssignment(Base):
    __tablename__ = "nurse_assignments"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False,
        index=True
    )
    nurse_id = Column(
        Integer,
        ForeignKey("nurses.id"),
        nullable=False,
        index=True
    )
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Assigned")
    notes = Column(Text, nullable=True)
    assigned_by = Column(String, nullable=True)
    assignment_type = Column(String, default="manual")
    priority_level = Column(String, default="Medium")

    prediction = relationship(
        "PredictionLog",
        back_populates="nurse_assignments"
    )

    nurse = relationship(
        "Nurse",
        back_populates="assignments"
    )


# =====================================================
# NURSE VITALS TABLE
# =====================================================

class NurseVitals(Base):
    __tablename__ = "nurse_vitals"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False,
        index=True
    )
    nurse_id = Column(
        Integer,
        ForeignKey("nurses.id"),
        nullable=False,
        index=True
    )
    temperature = Column(Float, nullable=True)
    heart_rate = Column(Integer, nullable=True)
    blood_pressure = Column(String, nullable=True)
    oxygen_level = Column(Integer, nullable=True)
    respiratory_rate = Column(Integer, nullable=True)
    pain_score = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship(
        "PredictionLog",
        back_populates="nurse_vitals"
    )

    nurse = relationship(
        "Nurse",
        back_populates="vitals"
    )


# =====================================================
# NURSE TASK TABLE
# =====================================================

class NurseTask(Base):
    __tablename__ = "nurse_tasks"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False,
        index=True
    )
    nurse_id = Column(
        Integer,
        ForeignKey("nurses.id"),
        nullable=False,
        index=True
    )
    task_title = Column(String, nullable=False)
    task_description = Column(Text, nullable=True)
    status = Column(String, default="Pending")
    priority = Column(String, default="Medium")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    prediction = relationship(
        "PredictionLog",
        back_populates="nurse_tasks"
    )

    nurse = relationship(
        "Nurse",
        back_populates="tasks"
    )


# =====================================================
# DOCTOR REVIEW TABLE
# =====================================================

class DoctorReview(Base):
    __tablename__ = "doctor_reviews"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False,
        index=True
    )
    doctor_id = Column(String, nullable=False, index=True)
    diagnosis = Column(Text, nullable=False)
    treatment_plan = Column(Text, nullable=False)
    medication_notes = Column(Text, nullable=True)
    follow_up_required = Column(Boolean, default=False)
    admit_status = Column(String, default="Not Admitted")
    reviewed_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship(
        "PredictionLog",
        back_populates="doctor_reviews"
    )


# =====================================================
# PATIENT STATUS TABLE
# =====================================================

class PatientStatus(Base):
    __tablename__ = "patient_statuses"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction_logs.id"),
        nullable=False,
        index=True
    )
    patient_status = Column(String, nullable=False, index=True)
    updated_by = Column(String, nullable=False)
    updated_role = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship(
        "PredictionLog",
        back_populates="status_timeline"
    )


# =====================================================
# STAFF MEMBER TABLE
# =====================================================

class StaffMember(Base):
    __tablename__ = "staff_members"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True, index=True)
    role = Column(String, nullable=False, index=True)
    department = Column(String, nullable=True)
    active_status = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class EmergencyQueue(Base):
    __tablename__ = "emergency_queue"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, unique=True, index=True)
    esi_severity = Column(String, nullable=False)
    priority = Column(String, default="Medium")
    estimated_wait_time = Column(Integer, default=30)
    critical_status = Column(Boolean, default=False)
    assignment_status = Column(String, default="waiting", index=True)
    assigned_nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=True, index=True)
    assigned_nurse_name = Column(String, nullable=True)
    assigned_doctor_id = Column(String, nullable=True)
    assigned_doctor_name = Column(String, nullable=True)
    assigned_at = Column(DateTime, nullable=True)
    triage_started_at = Column(DateTime, nullable=True)
    doctor_review_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    arrival_time = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("PredictionLog", back_populates="queue_entries")
    assigned_nurse = relationship("Nurse")


class Bed(Base):
    __tablename__ = "beds"

    id = Column(Integer, primary_key=True, index=True)
    bed_number = Column(String, nullable=False, unique=True, index=True)
    ward_type = Column(String, nullable=False)
    occupied = Column(Boolean, default=False)
    assigned_prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    assigned_at = Column(DateTime, nullable=True)

    prediction = relationship("PredictionLog", back_populates="bed_assignments")


class MedicationRecord(Base):
    __tablename__ = "medication_records"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=False, index=True)
    medication_name = Column(String, nullable=False)
    dosage = Column(String, nullable=False)
    route = Column(String, nullable=False)
    scheduled_time = Column(DateTime, nullable=False)
    administered_time = Column(DateTime, nullable=True)
    status = Column(String, default="Scheduled")
    side_effects = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    prediction = relationship("PredictionLog", back_populates="medication_records")
    nurse = relationship("Nurse")


class ClinicalAlert(Base):
    __tablename__ = "clinical_alerts"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

    prediction = relationship("PredictionLog", back_populates="clinical_alerts")


class DischargeSummary(Base):
    __tablename__ = "discharge_summaries"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    doctor_id = Column(String, nullable=False)
    diagnosis = Column(Text, nullable=False)
    treatment_given = Column(Text, nullable=False)
    medication_notes = Column(Text, nullable=True)
    follow_up_instructions = Column(Text, nullable=True)
    discharge_status = Column(String, default="Discharged")
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("PredictionLog", back_populates="discharge_summaries")


class StaffShift(Base):
    __tablename__ = "staff_shifts"

    id = Column(Integer, primary_key=True, index=True)
    staff_id = Column(String, nullable=False, index=True)
    staff_role = Column(String, nullable=False, index=True)
    department = Column(String, nullable=True)
    shift_type = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, default="Scheduled")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    emergency_contact_name = Column(String, nullable=True)
    emergency_contact_phone = Column(String, nullable=True)
    allergies = Column(Text, nullable=True)
    chronic_diseases = Column(Text, nullable=True)
    past_medical_history = Column(Text, nullable=True)
    assigned_nurse_id = Column(Integer, nullable=True, index=True)
    assigned_nurse_name = Column(String, nullable=True)
    assigned_doctor_id = Column(String, nullable=True, index=True)
    assigned_doctor_name = Column(String, nullable=True)
    assignment_status = Column(String, default="waiting", index=True)
    assigned_at = Column(DateTime, nullable=True)
    triage_started_at = Column(DateTime, nullable=True)
    doctor_review_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PatientPredictionLink(Base):
    __tablename__ = "patient_prediction_links"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, unique=True, index=True)
    linked_at = Column(DateTime, default=datetime.utcnow)


class LabOrder(Base):
    __tablename__ = "lab_orders"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    test_name = Column(String, nullable=False)
    test_type = Column(String, nullable=True)
    priority = Column(String, default="Routine")
    status = Column(String, default="Ordered")
    ordered_at = Column(DateTime, default=datetime.utcnow)
    result_notes = Column(Text, nullable=True)
    result_file_path = Column(String, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class Referral(Base):
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    referring_doctor_id = Column(String, nullable=False, index=True)
    specialist_department = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    urgency = Column(String, default="Routine")
    status = Column(String, default="Pending")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class Consent(Base):
    __tablename__ = "consents"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    consent_type = Column(String, nullable=False)
    accepted = Column(Boolean, default=False)
    signed_by = Column(String, nullable=False)
    signed_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    staff_id = Column(String, nullable=False, index=True)
    incident_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    action_taken = Column(Text, nullable=True)
    reported_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Open")


class BillingRecord(Base):
    __tablename__ = "billing_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    insurance_provider = Column(String, nullable=True)
    policy_number = Column(String, nullable=True)
    visit_cost = Column(Float, default=0)
    treatment_cost = Column(Float, default=0)
    medication_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    payment_status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id = Column(Integer, primary_key=True, index=True)
    item_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    quantity = Column(Integer, default=0)
    unit = Column(String, nullable=True)
    minimum_stock_level = Column(Integer, default=0)
    expiry_date = Column(DateTime, nullable=True)
    location = Column(String, nullable=True)
    status = Column(String, default="Available")


class FollowUpAppointment(Base):
    __tablename__ = "follow_up_appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    appointment_date = Column(DateTime, nullable=False)
    department = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    status = Column(String, default="Scheduled")
    notes = Column(Text, nullable=True)


class NotificationLog(Base):
    __tablename__ = "notification_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    contact_name = Column(String, nullable=False)
    contact_phone = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String, nullable=False)
    status = Column(String, default="Sent")
    sent_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, index=True)
    role = Column(String, nullable=True)
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=True)
    entity_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class HistoricalMedicalReport(Base):
    __tablename__ = "historical_medical_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    patient_name = Column(String, nullable=False)
    uploaded_by = Column(String, nullable=False, index=True)
    uploaded_by_role = Column(String, nullable=False, index=True)
    report_type = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    image_metadata = Column(Text, nullable=True)
    image_quality_notes = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    risk_flags = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    doctor_notes = Column(Text, nullable=True)
    nurse_notes = Column(Text, nullable=True)
    upload_notes = Column(Text, nullable=True)


class TriageUploadedFile(Base):
    __tablename__ = "triage_uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    patient_name = Column(String, nullable=False)
    uploaded_by = Column(String, nullable=False, index=True)
    uploaded_by_role = Column(String, nullable=False, index=True)
    triage_session_id = Column(String, nullable=True, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    report_type = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    image_metadata = Column(Text, nullable=True)
    image_quality_notes = Column(Text, nullable=True)
    detected_conditions = Column(Text, nullable=True)
    risk_flags = Column(Text, nullable=True)
    clinical_summary = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True)

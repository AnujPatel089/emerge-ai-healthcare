"""
api_extensions.py
New v2 routes — registered in main.py via:

    from src.api_extensions import router as v2_router
    app.include_router(v2_router)

Endpoints added:
  POST /v2/analyze-image-dl         Deep-learning image analysis (EfficientNet-B0)
  POST /v2/triage                   Multi-modal triage scoring (ESI 1-5)
  GET  /v2/report/{log_id}          Download PDF clinical report
  GET  /v2/dashboard/summary        Aggregate metrics for dashboard
"""

import json
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func

from src.database import SessionLocal
from src.models import PredictionLog, ClinicalFeedback, TriageUploadedFile
from src.audit_logger import save_prediction_log, convert_to_text
from src.image_analyzer import analyze_medical_image
from src.dl_module import get_dl_analyzer
from src.triage_engine import compute_triage, from_payload
from src.report_generator import generate_report
from src.auth import get_current_user


router = APIRouter(prefix="/v2", tags=["v2-multimodal"])


# -----------------------------
# ROLE CHECKER (mirrors main.py)
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


def _json_loads(value, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


# -----------------------------
# INPUT MODELS
# -----------------------------

class TriageInput(BaseModel):
    # Link to an existing PredictionLog row (from /predict)
    # so vitals + symptoms are pulled automatically.
    log_id: Optional[int] = None

    # OR supply vitals inline if not linking to an existing log
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    respiratory_rate: Optional[int] = None
    spo2: Optional[int] = None
    temperature: Optional[float] = None
    consciousness: Optional[str] = "alert"

    # Only needed when not linking via log_id
    symptoms: Optional[List[Dict[str, Any]]] = None

    # Paste in the output of /extract-symptoms
    nlp_findings: Optional[Dict[str, Any]] = None

    # Link an image log_id returned by /v2/analyze-image-dl
    image_log_id: Optional[int] = None

    # Patient context
    age: Optional[int] = None
    sex: Optional[str] = None
    patient_notes: Optional[str] = None


# -----------------------------
# 1. DEEP LEARNING IMAGE ANALYSIS
# -----------------------------

@router.post("/analyze-image-dl")
def analyze_image_dl(
    image: UploadFile = File(...),
    patient_notes: Optional[str] = Form(None),
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"]))
):
    """
    Runs two analyzers on the uploaded image:
      - Your existing analyze_medical_image (classical CV)
      - EfficientNet-B0 DL model (wound class + infection severity)

    Saves result into PredictionLog.cv_analysis.
    Returns log_id so it can be linked into POST /v2/triage.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image file.")

    image_bytes = image.file.read()

    # Classical CV — your existing module expects a file-like object
    classical_result = analyze_medical_image(io.BytesIO(image_bytes))

    # Deep learning
    dl_result = get_dl_analyzer().analyze(image_bytes)

    # Merge both into one dict stored in cv_analysis
    merged_analysis = {
        "classical_cv":          classical_result,
        "deep_learning":         dl_result,
        "wound_class":           dl_result.get("wound_class"),
        "infection_severity":    dl_result.get("infection_severity"),
        "severity_score":        dl_result.get("severity_score"),
        "wound_confidence":      dl_result.get("wound_confidence"),
        "infection_confidence":  dl_result.get("infection_confidence"),
        "wound_distribution":    dl_result.get("wound_distribution"),
        "infection_distribution": dl_result.get("infection_distribution"),
        "model":                 dl_result.get("model"),
        "filename":              image.filename,
        "analyzed_by":           current_user["username"],
        "analyzed_at":           datetime.utcnow().isoformat(),
    }

    # Save using your existing save_prediction_log.
    # ml_prediction = "IMAGE_ANALYSIS" makes it easy to filter in history.
    log_id = save_prediction_log(
        patient_data={
            "problem_description": f"Image analysis: {image.filename}",
            "cv_analysis":         merged_analysis,
            "llm_ready_text":      patient_notes,
        },
        ml_prediction="IMAGE_ANALYSIS",
        final_prediction=dl_result.get("wound_class", "unknown"),
        safety_reasons=[],
        clinical_explanations=(
            f"Wound: {dl_result.get('wound_class')} | "
            f"Infection: {dl_result.get('infection_severity')} | "
            f"Severity score: {dl_result.get('severity_score', 0):.3f}"
        ),
        confidence=dl_result.get("wound_confidence"),
        source=current_user["username"],
        feedback="Image Analysis"
    )

    if log_id is None:
        raise HTTPException(
            500,
            "Image analysis completed but failed to save to PostgreSQL."
        )

    return {
        "status":               "success",
        "processed_by":         current_user["username"],
        "role":                 current_user["role"],
        "log_id":               log_id,
        "filename":             image.filename,
        "wound_class":          dl_result.get("wound_class"),
        "wound_confidence":     dl_result.get("wound_confidence"),
        "infection_severity":   dl_result.get("infection_severity"),
        "severity_score":       dl_result.get("severity_score"),
        "wound_distribution":   dl_result.get("wound_distribution"),
        "infection_distribution": dl_result.get("infection_distribution"),
        "classical_cv":         classical_result,
        "model":                dl_result.get("model"),
    }


# -----------------------------
# 2. MULTI-MODAL TRIAGE SCORING
# -----------------------------

@router.post("/triage")
def triage_multimodal(
    data: TriageInput,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"]))
):
    """
    Fuses vitals + symptoms + NLP + image into ESI 1-5 triage score.

    Option A — link an existing prediction log:
        { "log_id": 42, "nlp_findings": {...}, "image_log_id": 7 }
        Vitals and symptoms are pulled automatically from PredictionLog row 42.

    Option B — supply everything inline:
        { "heart_rate": 118, "spo2": 91, "symptoms": [...], ... }
    """
    db = SessionLocal()
    try:
        payload: Dict[str, Any] = {}

        # -----------------------------------------------
        # Option A: pull vitals/symptoms from existing log
        # -----------------------------------------------
        if data.log_id:
            base_log = db.query(PredictionLog).filter(
                PredictionLog.id == data.log_id
            ).first()

            if not base_log:
                raise HTTPException(
                    404,
                    f"log_id {data.log_id} not found. Run /predict first."
                )

            payload["vitals"] = {
                "heart_rate":       base_log.triage_vital_hr,
                "systolic_bp":      base_log.triage_vital_sbp,
                "respiratory_rate": base_log.triage_vital_rr,
                "spo2":             base_log.triage_vital_o2,
                "temperature":      base_log.triage_vital_temp,
                "consciousness":    data.consciousness or "alert",
            }

            # Convert binary symptom columns to structured list
            symptom_map = {
                "chest_pain":          base_log.cc_chestpain,
                "shortness_of_breath": base_log.cc_shortnessofbreath,
                "headache":            base_log.cc_headache,
                "fever":               base_log.cc_fever,
                "abdominal_pain":      base_log.cc_abdominalpain,
                "dizziness":           base_log.cc_dizziness,
                "syncope":             base_log.cc_syncope,
                "weakness":            base_log.cc_weakness,
            }
            payload["structured_symptoms"] = [
                {"name": name, "severity": "moderate"}
                for name, flag in symptom_map.items()
                if flag == 1
            ]

            payload["age"] = data.age or base_log.age
            payload["sex"] = data.sex or base_log.gender

        # -----------------------------------------------
        # Option B: use inline values
        # -----------------------------------------------
        else:
            payload["vitals"] = {
                "heart_rate":       data.heart_rate,
                "systolic_bp":      data.systolic_bp,
                "respiratory_rate": data.respiratory_rate,
                "spo2":             data.spo2,
                "temperature":      data.temperature,
                "consciousness":    data.consciousness or "alert",
            }
            payload["structured_symptoms"] = data.symptoms or []
            payload["age"] = data.age
            payload["sex"] = data.sex

        # NLP findings always come from the request body
        payload["nlp_findings"] = data.nlp_findings

        # -----------------------------------------------
        # Pull image analysis from DB if image_log_id given
        # -----------------------------------------------
        if data.image_log_id:
            img_log = db.query(PredictionLog).filter(
                PredictionLog.id == data.image_log_id,
                PredictionLog.ml_prediction == "IMAGE_ANALYSIS"
            ).first()

            if img_log and img_log.cv_analysis:
                try:
                    cv_data = json.loads(img_log.cv_analysis)
                    payload["image_analysis"] = {
                        "severity_score":     cv_data.get("severity_score", 0),
                        "infection_severity": cv_data.get("infection_severity"),
                        "wound_class":        cv_data.get("wound_class"),
                        "cv_flags":           (
                            cv_data.get("classical_cv", {}).get("flags", {})
                        ),
                    }
                except (json.JSONDecodeError, AttributeError):
                    pass  # image data unreadable — continue without it

        # -----------------------------------------------
        # Compute triage score
        # -----------------------------------------------
        triage_input = from_payload(payload)
        triage_result = compute_triage(triage_input)

        triage_summary = (
            f"ESI {triage_result['esi_level']} | "
            f"Risk: {triage_result['composite_risk']}/100 | "
            f"{triage_result['esi_label']}"
        )

        triage_log_id = save_prediction_log(
            patient_data={
                "age":                payload.get("age"),
                "problem_description": data.patient_notes or "Multi-modal triage",
                "llm_ready_text":     json.dumps(
                    triage_result.get("contributions", {})
                ),
                "cv_analysis":        payload.get("image_analysis"),
            },
            ml_prediction=f"ESI_{triage_result['esi_level']}",
            final_prediction=f"ESI_{triage_result['esi_level']}",
            safety_reasons=triage_result.get("red_flags", []),
            clinical_explanations=triage_summary,
            confidence=triage_result["composite_risk"] / 100,
            source=current_user["username"],
            feedback="Triage Score"
        )

        if triage_log_id is None:
            raise HTTPException(
                500,
                "Triage scored but failed to save to PostgreSQL."
            )

        return {
            "status":                    "success",
            "processed_by":              current_user["username"],
            "role":                      current_user["role"],
            "triage_log_id":             triage_log_id,
            "linked_prediction_log_id":  data.log_id,
            "linked_image_log_id":       data.image_log_id,
            "esi_level":                 triage_result["esi_level"],
            "esi_label":                 triage_result["esi_label"],
            "composite_risk":            triage_result["composite_risk"],
            "total_score":               triage_result["total_score"],
            "contributions":             triage_result["contributions"],
            "red_flags":                 triage_result["red_flags"],
            "components":                triage_result["components"],
        }

    finally:
        db.close()


# -----------------------------
# 3. PDF CLINICAL REPORT
# -----------------------------

@router.get("/report/{log_id}")
def get_clinical_report(
    log_id: int,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    """
    Generates and downloads a PDF clinical report for any log_id
    from the prediction_logs table.
    """
    db = SessionLocal()
    try:
        log = db.query(PredictionLog).filter(
            PredictionLog.id == log_id
        ).first()

        if not log:
            raise HTTPException(404, f"log_id {log_id} not found.")

        def safe_parse(value):
            if not value:
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        # Image analysis
        cv_data = safe_parse(log.cv_analysis)
        image_analysis = None
        if isinstance(cv_data, dict):
            image_analysis = {
                "wound_class":        cv_data.get("wound_class"),
                "wound_confidence":   cv_data.get("wound_confidence"),
                "infection_severity": cv_data.get("infection_severity"),
                "severity_score":     cv_data.get("severity_score"),
                "model":              cv_data.get("model", "classical CV"),
            }

        # Triage reconstruction
        triage = {
            "esi_level":      3,
            "esi_label":      log.clinical_explanations or "See record",
            "composite_risk": round((log.confidence or 0) * 100, 1),
            "total_score":    0,
            "contributions":  {},
            "red_flags":      safe_parse(log.safety_reasons) or [],
        }
        if log.ml_prediction and log.ml_prediction.startswith("ESI_"):
            try:
                triage["esi_level"] = int(log.ml_prediction.split("_")[1])
            except (IndexError, ValueError):
                pass

        # Vitals
        vitals = {
            "heart_rate":       log.triage_vital_hr,
            "systolic_bp":      log.triage_vital_sbp,
            "respiratory_rate": log.triage_vital_rr,
            "spo2":             log.triage_vital_o2,
            "temperature":      log.triage_vital_temp,
        }

        # Symptoms from binary columns
        symptom_map = {
            "chest_pain":          log.cc_chestpain,
            "shortness_of_breath": log.cc_shortnessofbreath,
            "headache":            log.cc_headache,
            "fever":               log.cc_fever,
            "abdominal_pain":      log.cc_abdominalpain,
            "dizziness":           log.cc_dizziness,
            "syncope":             log.cc_syncope,
            "weakness":            log.cc_weakness,
        }
        symptoms_structured = [
            {"name": name, "severity": "present"}
            for name, flag in symptom_map.items()
            if flag == 1
        ]

        patient = {
            "mrn":          str(log_id),
            "name":         "—",
            "age":          log.age,
            "sex":          log.gender,
            "arrival_time": str(log.created_at),
        }

        upload = db.query(TriageUploadedFile).filter(
            (TriageUploadedFile.triage_session_id == str(log_id)) |
            (TriageUploadedFile.uploaded_by == current_user["username"])
        ).order_by(TriageUploadedFile.uploaded_at.desc()).first()
        historical_context = None
        if upload:
            historical_context = {
                "file_name": upload.file_name,
                "file_type": upload.file_type,
                "report_type": upload.report_type,
                "ocr_text": upload.ocr_text,
                "image_metadata": _json_loads(upload.image_metadata, {}),
                "image_quality_notes": _json_loads(upload.image_quality_notes, []),
                "detected_conditions": _json_loads(upload.detected_conditions, {}),
                "risk_flags": _json_loads(upload.risk_flags, []),
                "clinical_summary": _json_loads(upload.clinical_summary, {}),
            }

        output_path = generate_report(
            prediction_id=str(log_id),
            patient=patient,
            vitals=vitals,
            symptoms={
                "structured": symptoms_structured,
                "nlp": {
                    "urgency_terms": safe_parse(log.emergency_keywords) or [],
                    "entities":      safe_parse(log.matched_terms) or [],
                    "negations":     [],
                },
            },
            image_analysis=image_analysis,
            triage=triage,
            shap_values=None,
            image_bytes=None,
            historical_context=historical_context,
            clinician=current_user["username"],
        )

    finally:
        db.close()

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"emergeai_report_{log_id}.pdf",
    )


# -----------------------------
# 4. DASHBOARD SUMMARY
# -----------------------------

@router.get("/dashboard/summary")
def dashboard_summary(
    hours: int = 24,
    current_user: dict = Depends(require_role(["nurse", "doctor", "admin"]))
):
    """
    Returns aggregate stats for the clinical dashboard.
    """
    db = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=hours)

        total = db.query(func.count(PredictionLog.id)).filter(
            PredictionLog.created_at >= since
        ).scalar() or 0

        image_total = db.query(func.count(PredictionLog.id)).filter(
            PredictionLog.created_at >= since,
            PredictionLog.ml_prediction == "IMAGE_ANALYSIS"
        ).scalar() or 0

        triage_logs = db.query(PredictionLog).filter(
            PredictionLog.created_at >= since,
            PredictionLog.ml_prediction.like("ESI_%")
        ).all()

        esi_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        risk_scores: List[float] = []

        for row in triage_logs:
            try:
                level = int(row.ml_prediction.split("_")[1])
                esi_counts[level] = esi_counts.get(level, 0) + 1
                if row.confidence is not None:
                    risk_scores.append(row.confidence * 100)
            except (IndexError, ValueError):
                continue

        avg_risk = (
            round(sum(risk_scores) / len(risk_scores), 1)
            if risk_scores else 0
        )

        accepted = db.query(func.count(ClinicalFeedback.id)).filter(
            ClinicalFeedback.accepted == True,
            ClinicalFeedback.created_at >= since
        ).scalar() or 0

        overridden = db.query(func.count(ClinicalFeedback.id)).filter(
            ClinicalFeedback.accepted == False,
            ClinicalFeedback.created_at >= since
        ).scalar() or 0

        return {
            "status":               "success",
            "window_hours":         hours,
            "total_predictions":    total,
            "total_image_analyses": image_total,
            "total_triage_scores":  len(triage_logs),
            "esi_distribution":     esi_counts,
            "avg_composite_risk":   avg_risk,
            "high_acuity_count":    esi_counts[1] + esi_counts[2],
            "clinical_feedback": {
                "accepted":   accepted,
                "overridden": overridden,
            },
        }

    finally:
        db.close()

"""
audit_logger.py

Save every prediction into PostgreSQL.
"""

import json
from src.database import SessionLocal
from src.models import PredictionLog


def convert_to_text(value):
    if value is None:
        return None

    if isinstance(value, (list, dict)):
        return json.dumps(value)

    return str(value)


def save_prediction_log(
    patient_data,
    ml_prediction,
    final_prediction,
    safety_reasons,
    clinical_explanations,
    confidence=None,
    source="API Prediction",
    feedback="Pending"
):
    db = SessionLocal()

    try:
        log = PredictionLog(
            age=patient_data.get("age"),
            gender=patient_data.get("gender"),
            race=patient_data.get("race"),
            ethnicity=patient_data.get("ethnicity"),
            arrivalmode=patient_data.get("arrivalmode"),

            triage_vital_hr=patient_data.get("triage_vital_hr"),
            triage_vital_sbp=patient_data.get("triage_vital_sbp"),
            triage_vital_dbp=patient_data.get("triage_vital_dbp"),
            triage_vital_rr=patient_data.get("triage_vital_rr"),
            triage_vital_o2=patient_data.get("triage_vital_o2"),
            triage_vital_temp=patient_data.get("triage_vital_temp"),

            cc_chestpain=patient_data.get("cc_chestpain"),
            cc_shortnessofbreath=patient_data.get("cc_shortnessofbreath"),
            cc_headache=patient_data.get("cc_headache"),
            cc_fever=patient_data.get("cc_fever"),
            cc_abdominalpain=patient_data.get("cc_abdominalpain"),
            cc_dizziness=patient_data.get("cc_dizziness"),
            cc_syncope=patient_data.get("cc_syncope"),
            cc_weakness=patient_data.get("cc_weakness"),

            ml_prediction=str(ml_prediction),
            final_prediction=str(final_prediction),
            confidence=confidence,

            safety_reasons=convert_to_text(safety_reasons),
            clinical_explanations=convert_to_text(clinical_explanations),

            problem_description=patient_data.get("problem_description"),
            llm_ready_text=patient_data.get("llm_ready_text"),
            matched_terms=convert_to_text(patient_data.get("matched_terms")),
            emergency_keywords=convert_to_text(patient_data.get("emergency_keywords")),
            cv_analysis=convert_to_text(patient_data.get("cv_analysis")),

            source=source,
            feedback=feedback
        )

        db.add(log)
        db.commit()
        db.refresh(log)

        return log.id

    except Exception as e:
        db.rollback()
        print("Database logging error:", e)
        return None

    finally:
        db.close()
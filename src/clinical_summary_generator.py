"""
Generate clinician-facing summaries from historical report analysis.
"""

from __future__ import annotations


DISCLAIMER = (
    "This tool analyzes uploaded historical medical reports and medical images for "
    "educational and decision-support purposes only. It does not provide a final "
    "diagnosis. A licensed healthcare professional must verify all findings."
)


def generate_clinical_summary(
    patient_name: str,
    report_type: str,
    analysis: dict,
    risk_flags: list[dict],
    notes: str = "",
    image_assessment: dict | None = None,
) -> dict:
    conditions = analysis.get("detected_conditions", {})

    def terms(category: str) -> str:
        values = conditions.get(category, [])
        return ", ".join(values) if values else "No keyword mention detected."

    known = sorted(conditions.keys())
    high_flags = [flag["label"] for flag in risk_flags if flag["level"] in {"High Risk", "Allergy Alert"}]

    recommended_questions = [
        "What current medications is the patient taking, and when was the last dose?",
        "Any current chest pain, shortness of breath, fainting, fever, or neurological symptoms?",
        "Any known medication, latex, contrast, or food allergies?",
        "Any recent emergency visits, hospital admissions, surgeries, or ICU stays?",
        "Any recent abnormal blood tests, glucose readings, kidney issues, or infections?",
    ]

    image_assessment = image_assessment or {}
    image_quality_notes = image_assessment.get("image_quality_notes", [])
    ocr_text = image_assessment.get("ocr_text") or ""

    triage_notes = (
        "Historical report findings should be used as supporting context beside current vitals, "
        "symptoms, ESI prediction, and clinician assessment. Do not override triage solely from history."
    )

    narrative = (
        f"{patient_name or 'Patient'} has uploaded a {report_type.lower()} report. "
        f"Keyword analysis detected possible historical indicators for: {', '.join(known) if known else 'no major category'}. "
        f"Current risk flags for clinician review: {', '.join(high_flags) if high_flags else 'no high-risk flag detected'}. "
        "Doctor/nurse should verify these findings with the patient and current chart."
    )
    if image_assessment:
        narrative += " Image assessment includes metadata and quality checks only; image content needs clinician review."

    return {
        "disclaimer": DISCLAIMER,
        "patient_background": narrative,
        "uploaded_report_image_type": report_type,
        "known_conditions": known,
        "current_risk_factors": [flag["label"] for flag in risk_flags],
        "medication_history": terms("Medications"),
        "allergy_alerts": terms("Allergies"),
        "previous_hospital_visits": terms("Emergency Visits"),
        "image_quality_notes": image_quality_notes or ["No image quality warning detected."],
        "ocr_findings": ocr_text[:1000] if ocr_text else image_assessment.get("ocr_message", "No OCR text available."),
        "possible_emergency_concerns": [
            concern for concern in ["Heart Risk", "Respiratory Risk", "Kidney Risk", "Neurological Symptoms", "Infection History"]
            if concern in conditions
        ],
        "recommended_questions": recommended_questions,
        "triage_notes": triage_notes,
        "uploaded_notes": notes or "",
        "summary_text": narrative,
    }

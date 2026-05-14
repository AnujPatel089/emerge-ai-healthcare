"""
Risk flag generation from historical report mention analysis.

Flags are possible risk indicators for clinician review, not final diagnoses.
"""

from __future__ import annotations


def generate_risk_flags(analysis: dict) -> list[dict]:
    conditions = analysis.get("detected_conditions", {})
    flags: list[dict] = []

    has = lambda name: name in conditions

    if has("Heart Risk") and ("chest pain" in conditions.get("Heart Risk", []) or "coronary artery disease" in conditions.get("Heart Risk", [])):
        flags.append({"level": "High Risk", "label": "High Cardiac Risk", "reason": "Detected historical cardiac-risk mentions needing clinician review."})

    if has("Diabetes"):
        level = "High Risk" if any(term in conditions["Diabetes"] for term in ["high glucose", "hyperglycemia", "insulin"]) else "Medium Risk"
        flags.append({"level": level, "label": "Metabolic Risk", "reason": "Detected possible diabetes or glucose-control history."})

    if has("Respiratory Risk"):
        level = "High Risk" if any(term in conditions["Respiratory Risk"] for term in ["copd", "low oxygen", "shortness of breath"]) else "Medium Risk"
        flags.append({"level": level, "label": "Respiratory Risk", "reason": "Detected respiratory history indicators."})

    if has("Allergies"):
        flags.append({"level": "Allergy Alert", "label": "Allergy Alert", "reason": "Detected possible allergy mention; verify before medication orders."})

    if "icu" in conditions.get("Emergency Visits", []) or "intensive care" in conditions.get("Emergency Visits", []):
        flags.append({"level": "High Risk", "label": "High Emergency Risk", "reason": "Detected previous ICU or intensive-care mention."})

    if has("Kidney Risk"):
        flags.append({"level": "Medium Risk", "label": "Renal Risk", "reason": "Detected possible kidney disease or renal-function concern."})

    if not flags:
        flags.append({"level": "Low Risk", "label": "No Major Historical Flag Detected", "reason": "No major keyword-based historical risk flag was detected."})

    return flags


def generate_image_risk_flags(report_type: str, notes: str = "", ocr_text: str = "", image_quality_notes: list[str] | None = None) -> list[dict]:
    """Generate image-related review flags from image metadata, notes, and OCR text."""
    text = f"{report_type} {notes} {ocr_text}".lower()
    quality_notes = image_quality_notes or []
    flags: list[dict] = []

    high_terms = ["bleeding", "stroke", "tumor", "fracture", "pneumonia", "infection", "abnormal"]
    if any(term in text for term in high_terms):
        flags.append({
            "level": "High Risk",
            "label": "High Review Needed",
            "reason": "OCR or notes detected medical terms that need clinician review.",
        })

    if quality_notes:
        flags.append({
            "level": "Image Quality Warning",
            "label": "Low Image Quality",
            "reason": "; ".join(quality_notes[:3]),
        })

    if report_type.lower() in {"x-ray", "ct scan", "mri", "medical image"}:
        flags.append({
            "level": "Clinician Review Required",
            "label": "Clinician Imaging Review Required",
            "reason": "Uploaded imaging requires professional interpretation.",
        })

    if "chest pain" in text and report_type.lower() in {"x-ray", "medical image"}:
        flags.append({
            "level": "Medium Risk",
            "label": "Possible Cardio-Respiratory Review Needed",
            "reason": "Notes mention chest pain with an imaging report.",
        })

    if "wound" in text or "infection" in text:
        flags.append({
            "level": "Medium Risk",
            "label": "Possible Infection/Wound Review Needed",
            "reason": "Notes or OCR mention possible wound or infection concern.",
        })

    return flags

"""
Rule-based medical history mention detection.

This module intentionally uses cautious language: it detects mentions and
possible historical indicators, not confirmed diagnoses.
"""

from __future__ import annotations

import re


KEYWORD_CATEGORIES = {
    "Diabetes": ["diabetes", "diabetic", "insulin", "metformin", "high glucose", "hyperglycemia"],
    "Hypertension": ["hypertension", "high blood pressure", "elevated bp", "antihypertensive"],
    "Heart Risk": ["chest pain", "myocardial infarction", "heart attack", "cad", "coronary artery disease", "angina", "ecg abnormal"],
    "Respiratory Risk": ["asthma", "copd", "shortness of breath", "wheezing", "low oxygen", "pneumonia"],
    "Allergies": ["allergy", "allergic", "penicillin allergy", "latex allergy", "medication allergy"],
    "Kidney Risk": ["kidney disease", "renal failure", "ckd", "creatinine high", "dialysis"],
    "Previous Surgery": ["surgery", "surgical", "appendectomy", "bypass", "stent", "operation", "post-operative"],
    "Medications": ["prescription", "medication", "tablet", "capsule", "dose", "mg", "antibiotic", "anticoagulant"],
    "Abnormal Labs": ["abnormal", "elevated", "low hemoglobin", "high creatinine", "troponin", "positive culture"],
    "Emergency Visits": ["emergency visit", "ed visit", "er visit", "icu", "intensive care", "admission"],
    "Smoking History": ["smoking", "smoker", "tobacco", "pack-year"],
    "Infection History": ["infection", "sepsis", "pneumonia", "cellulitis", "uti", "positive culture"],
    "Pregnancy Notes": ["pregnant", "pregnancy", "gestational", "prenatal", "postpartum"],
    "Neurological Symptoms": ["stroke", "seizure", "weakness", "slurred speech", "confusion", "neurologic"],
}


def _find_matches(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    matches = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, lowered):
            matches.append(keyword)
    return matches


def analyze_medical_history(text: str) -> dict:
    conditions = {}
    for category, keywords in KEYWORD_CATEGORIES.items():
        matches = _find_matches(text, keywords)
        if matches:
            conditions[category] = matches

    all_matches = sorted({match for matches in conditions.values() for match in matches})
    return {
        "detected_conditions": conditions,
        "detected_terms": all_matches,
        "has_history_mentions": bool(conditions),
    }


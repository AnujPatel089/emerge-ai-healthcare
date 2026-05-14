"""
Clinical triage rules for synthetic data generation, model fallback, and UI display.

These rules are educational decision support logic, not medical advice. Lower ESI
means higher urgency.
"""

from __future__ import annotations

import math
from typing import Any


ESI_LABELS = {
    1: "Immediate / Critical",
    2: "Emergency",
    3: "Urgent",
    4: "Semi-Urgent",
    5: "Non-Urgent",
}

ESI_COLORS = {
    1: "#b91c1c",
    2: "#ea580c",
    3: "#ca8a04",
    4: "#2563eb",
    5: "#16a34a",
}


def _flag(patient: dict[str, Any], *names: str) -> int:
    for name in names:
        if int(patient.get(name, 0) or 0) == 1:
            return 1
    return 0


def _num(patient: dict[str, Any], name: str, default: float) -> float:
    value = patient.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def calculate_news2_score(patient: dict[str, Any]) -> int:
    """Approximate NEWS2-style risk score from vital signs."""
    rr = _num(patient, "Respiratory_Rate", _num(patient, "triage_vital_rr", 18))
    o2 = _num(patient, "Oxygen_Saturation", _num(patient, "triage_vital_o2", 98))
    sbp = _num(patient, "Systolic_BP", _num(patient, "triage_vital_sbp", 120))
    hr = _num(patient, "Heart_Rate", _num(patient, "triage_vital_hr", 80))
    temp = _num(patient, "Temperature", _num(patient, "triage_vital_temp", 98.6))

    score = 0
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    if o2 <= 91:
        score += 3
    elif 92 <= o2 <= 93:
        score += 2
    elif 94 <= o2 <= 95:
        score += 1

    if sbp <= 90:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110 or sbp >= 220:
        score += 1

    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif 41 <= hr <= 50 or 91 <= hr <= 110:
        score += 1

    if temp <= 95.0:
        score += 3
    elif 95.1 <= temp <= 96.8 or temp >= 100.4:
        score += 1

    return int(score)


def rule_based_esi(patient: dict[str, Any]) -> tuple[int, list[str]]:
    """Return ESI 1-5 plus human-readable rule reasons."""
    news2 = calculate_news2_score(patient)
    age = _num(patient, "Age", _num(patient, "age", 45))
    hr = _num(patient, "Heart_Rate", _num(patient, "triage_vital_hr", 80))
    sbp = _num(patient, "Systolic_BP", _num(patient, "triage_vital_sbp", 120))
    o2 = _num(patient, "Oxygen_Saturation", _num(patient, "triage_vital_o2", 98))
    chest_pain = _flag(patient, "Chest_Pain", "cc_chestpain")
    shortness_of_breath = _flag(patient, "Shortness_of_Breath", "cc_shortnessofbreath")
    fever = _flag(patient, "Fever", "cc_fever")

    reasons: list[str] = []
    score = news2

    if o2 < 90:
        score += 4
        reasons.append("Oxygen saturation below 90% increases emergency severity.")
    elif o2 < 94:
        score += 2
        reasons.append("Reduced oxygen saturation increases triage priority.")

    if chest_pain:
        score += 3
        reasons.append("Chest pain is treated as a high-risk presenting symptom.")
    if shortness_of_breath:
        score += 3
        reasons.append("Shortness of breath increases risk of respiratory compromise.")
    if chest_pain and shortness_of_breath:
        score += 3
        reasons.append("Chest pain with shortness of breath is a red-flag combination.")
    if hr > 130:
        score += 3
        reasons.append("Heart rate above 130 suggests possible instability.")
    elif hr > 110:
        score += 1
        reasons.append("Elevated heart rate contributes to urgency.")
    if sbp < 90:
        score += 4
        reasons.append("Very low systolic blood pressure suggests shock risk.")
    if fever and (hr > 110 or age >= 65):
        score += 1
        reasons.append("Fever with tachycardia or older age raises concern for infection severity.")

    if news2 >= 9 or o2 < 85 or sbp < 80:
        esi = 1
    elif score >= 11:
        esi = 2
    elif score >= 6:
        esi = 3
    elif score >= 3:
        esi = 4
    else:
        esi = 5

    if not reasons:
        reasons.append("No critical rule triggered; vitals and symptoms appear lower acuity.")
    return int(esi), reasons


def estimate_icu_risk(patient: dict[str, Any], esi_level: int | None = None) -> float:
    """Estimate ICU risk as a probability from vitals, symptoms, age, and ESI."""
    if esi_level is None:
        esi_level, _ = rule_based_esi(patient)

    age = _num(patient, "Age", _num(patient, "age", 45))
    hr = _num(patient, "Heart_Rate", _num(patient, "triage_vital_hr", 80))
    sbp = _num(patient, "Systolic_BP", _num(patient, "triage_vital_sbp", 120))
    rr = _num(patient, "Respiratory_Rate", _num(patient, "triage_vital_rr", 18))
    o2 = _num(patient, "Oxygen_Saturation", _num(patient, "triage_vital_o2", 98))
    news2 = calculate_news2_score(patient)

    linear = (
        -5.2
        + (esi_level == 1) * 3.0
        + (esi_level == 2) * 1.5
        + (o2 < 90) * 1.5
        + (sbp < 90) * 1.1
        + (hr > 125) * 0.7
        + (rr > 28) * 0.7
        + (news2 >= 7) * 0.8
        + (age >= 75) * 0.35
    )
    return round(1 / (1 + math.exp(-linear)), 3)


def estimate_readmission_risk(patient: dict[str, Any], esi_level: int | None = None, icu_risk: float | None = None) -> float:
    """Estimate 30-day readmission risk from age, comorbidities, acuity, and ICU risk."""
    if esi_level is None:
        esi_level, _ = rule_based_esi(patient)
    if icu_risk is None:
        icu_risk = estimate_icu_risk(patient, esi_level)

    age = _num(patient, "Age", _num(patient, "age", 45))
    diabetes = _flag(patient, "Diabetes")
    hypertension = _flag(patient, "Hypertension")
    smoking = _flag(patient, "Smoking")

    linear = (
        -3.1
        + (age / 100) * 1.55
        + diabetes * 0.45
        + hypertension * 0.3
        + smoking * 0.25
        + (esi_level <= 2) * 0.35
        + icu_risk * 1.1
    )
    return round(1 / (1 + math.exp(-linear)), 3)


def recommended_wait_time_minutes(esi_level: int) -> int:
    """Typical target wait time by ESI level."""
    return {1: 0, 2: 10, 3: 45, 4: 90, 5: 150}.get(int(esi_level), 60)


from __future__ import annotations

from typing import Any


def evaluate_clinical_guardrails(features: dict[str, Any], confidence: float | None = None, prediction_source: str = "ml") -> dict[str, Any]:
    hr = int(features.get("triage_vital_hr") or features.get("heart_rate") or 0)
    rr = int(features.get("triage_vital_rr") or features.get("respiratory_rate") or 0)
    o2 = int(features.get("triage_vital_o2") or features.get("oxygen_saturation") or 100)
    temp = float(features.get("triage_vital_temp") or features.get("temperature") or 37)
    chest = int(features.get("cc_chestpain") or features.get("chest_pain") or 0)
    fever = int(features.get("cc_fever") or features.get("fever") or 0)

    reasons = []
    severity = "healthy"
    if o2 < 90:
        severity = "critical"
        reasons.append("Low oxygen saturation creates possible risk; clinician review required.")
    if chest and hr >= 120:
        severity = "critical" if severity == "critical" else "warning"
        reasons.append("Chest pain with high heart rate creates possible urgent risk.")
    if rr and (rr < 8 or rr > 30):
        severity = "critical" if severity == "critical" else "warning"
        reasons.append("Abnormal respiratory rate requires urgent clinician review.")
    if (fever or temp >= 38.5) and o2 < 94:
        severity = "critical" if severity == "critical" else "warning"
        reasons.append("Fever with reduced oxygenation creates possible risk.")
    if confidence is not None and confidence < 0.55:
        severity = "warning" if severity == "healthy" else severity
        reasons.append("Low confidence prediction requires clinician review.")
    if prediction_source == "fallback_rules":
        severity = "warning" if severity == "healthy" else severity
        reasons.append("Fallback rule-based triage used. Clinician review required.")

    return {
        "guardrail_triggered": bool(reasons),
        "severity": severity,
        "safe_message": " ".join(reasons) if reasons else "No safety guardrail triggered. AI-supported triage only.",
        "recommended_action": "Clinician review required." if reasons else "Continue standard clinical review.",
    }

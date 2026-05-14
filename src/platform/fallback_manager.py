from __future__ import annotations

from typing import Any

from src.platform.clinical_guardrails import evaluate_clinical_guardrails


def fallback_rule_based_triage(features: dict[str, Any]) -> dict[str, Any]:
    hr = int(features.get("triage_vital_hr") or 0)
    rr = int(features.get("triage_vital_rr") or 16)
    o2 = int(features.get("triage_vital_o2") or 98)
    temp = float(features.get("triage_vital_temp") or 37)
    chest = int(features.get("cc_chestpain") or 0)
    shortness = int(features.get("cc_shortnessofbreath") or 0)
    fever = int(features.get("cc_fever") or 0)

    esi = 4
    reasons = []
    if o2 < 85 or (rr > 35 or rr < 6):
        esi = 1
        reasons.append("Severe oxygen or respiratory abnormality creates possible critical risk.")
    elif o2 < 90 or (chest and hr >= 120) or (shortness and o2 < 94):
        esi = 2
        reasons.append("Possible high-risk cardiopulmonary pattern detected.")
    elif rr > 28 or rr < 10 or (fever and o2 < 94) or hr >= 120:
        esi = 3
        reasons.append("Abnormal vitals create possible urgent risk.")
    elif 90 <= hr <= 110 and 12 <= rr <= 22 and o2 >= 95 and temp < 38:
        esi = 4
        reasons.append("Vitals appear relatively stable, but clinician review required.")
    else:
        esi = 3
        reasons.append("Mixed vitals require clinician review.")

    guardrails = evaluate_clinical_guardrails(features, confidence=None, prediction_source="fallback_rules")
    return {
        "prediction_source": "fallback_rules",
        "final_prediction": f"ESI {esi}",
        "ml_prediction": f"ESI {esi}",
        "confidence": None,
        "safety_reasons": reasons + [guardrails["safe_message"]],
        "clinical_explanations": [
            "Fallback rule-based triage used. Clinician review required.",
            "AI-supported triage only. This is not a final medical diagnosis.",
        ],
        "guardrails": guardrails,
    }

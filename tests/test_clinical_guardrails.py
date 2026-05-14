from src.platform.clinical_guardrails import evaluate_clinical_guardrails


def test_low_oxygen_triggers_critical_review():
    result = evaluate_clinical_guardrails({"triage_vital_o2": 86})
    assert result["guardrail_triggered"] is True
    assert result["severity"] == "critical"
    assert "clinician review required" in result["safe_message"].lower()

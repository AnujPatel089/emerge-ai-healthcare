from src.platform.fallback_manager import fallback_rule_based_triage


def test_missing_model_uses_rule_based_fallback():
    result = fallback_rule_based_triage({"triage_vital_o2": 88, "triage_vital_rr": 24})
    assert result["prediction_source"] == "fallback_rules"
    assert "Clinician review required" in result["clinical_explanations"][0]

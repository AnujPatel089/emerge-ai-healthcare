"""
shap_explainer.py

Simple SHAP-style explanation helper for Emergency Triage AI.
This version gives clinician-friendly explanations based on input values.
"""

def generate_clinical_explanation(patient, ml_prediction, final_prediction, safety_reasons):
    explanations = []

    if patient["triage_vital_o2"] < 92:
        explanations.append("Low oxygen saturation increased urgency.")

    if patient["triage_vital_hr"] > 120:
        explanations.append("High heart rate increased urgency.")

    if patient["triage_vital_sbp"] < 90:
        explanations.append("Low systolic blood pressure increased urgency.")

    if patient["triage_vital_rr"] > 24:
        explanations.append("High respiratory rate increased urgency.")

    if patient["triage_vital_temp"] >= 38:
        explanations.append("Fever may indicate infection or inflammation.")

    if patient["cc_chestpain"] == 1:
        explanations.append("Chest pain is a high-risk symptom.")

    if patient["cc_shortnessofbreath"] == 1:
        explanations.append("Shortness of breath is a high-risk symptom.")

    if patient["cc_syncope"] == 1:
        explanations.append("Syncope/fainting can indicate serious risk.")

    if patient["age"] >= 65:
        explanations.append("Older age may increase clinical risk.")

    if safety_reasons:
        explanations.append("Safety rules were applied to reduce clinical risk.")

    if not explanations:
        explanations.append("Prediction was mainly based on model pattern recognition from patient data.")

    return explanations
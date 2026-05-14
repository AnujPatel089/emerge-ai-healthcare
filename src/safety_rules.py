"""
safety_rules.py

Purpose:
Add clinical safety rules on top of the ML model.

In healthcare AI, safety rules help prevent dangerous under-triage.
"""

def apply_safety_rules(patient, model_prediction):
    """
    patient: dictionary of patient input
    model_prediction: ML predicted ESI level

    Lower ESI number = more urgent.
    """

    reasons = []
    final_prediction = model_prediction

    # Rule 1: Very low oxygen
    if patient["triage_vital_o2"] < 90:
        final_prediction = min(final_prediction, 2.0)
        reasons.append("Oxygen saturation is below 90%.")

    # Rule 2: Chest pain + shortness of breath
    if patient["cc_chestpain"] == 1 and patient["cc_shortnessofbreath"] == 1:
        final_prediction = min(final_prediction, 2.0)
        reasons.append("Chest pain with shortness of breath detected.")

    # Rule 3: Very low blood pressure
    if patient["triage_vital_sbp"] < 90:
        final_prediction = min(final_prediction, 2.0)
        reasons.append("Systolic blood pressure is very low.")

    # Rule 4: Very high heart rate
    if patient["triage_vital_hr"] > 130:
        final_prediction = min(final_prediction, 2.0)
        reasons.append("Heart rate is above 130.")

    # Rule 5: Unresponsive patient
    if patient.get("cc_unresponsive", 0) == 1:
        final_prediction = 1.0
        reasons.append("Patient is unresponsive.")

    if not reasons:
        reasons.append("No high-risk safety rule triggered.")

    return final_prediction, reasons
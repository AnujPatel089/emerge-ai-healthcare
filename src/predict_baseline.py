"""
predict_baseline.py

Purpose:
Test ML triage prediction + safety rule layer.
"""

import pandas as pd
import joblib

from src.safety_rules import apply_safety_rules


# Load trained model
model = joblib.load("models/triage_baseline_model.pkl")


# Sample patient input
patient = {
    "age": 55,
    "gender": "Male",
    "race": "White",
    "ethnicity": "Non-Hispanic",
    "arrivalmode": "Ambulance",

    "triage_vital_hr": 125,
    "triage_vital_sbp": 90,
    "triage_vital_dbp": 60,
    "triage_vital_rr": 24,
    "triage_vital_o2": 88,
    "triage_vital_temp": 38.5,

    "cc_chestpain": 1,
    "cc_shortnessofbreath": 1,
    "cc_headache": 0,
    "cc_fever": 1,
    "cc_abdominalpain": 0,
    "cc_dizziness": 1,
    "cc_syncope": 0,
    "cc_weakness": 1,
}

# Convert dictionary to dataframe
input_df = pd.DataFrame([patient])

# ML prediction
model_prediction = model.predict(input_df)[0]

# Probability
probabilities = model.predict_proba(input_df)[0]
classes = model.classes_

# Safety rule prediction
final_prediction, reasons = apply_safety_rules(patient, model_prediction)

print("\nML Predicted ESI Level:")
print(model_prediction)

print("\nFinal ESI After Safety Rules:")
print(final_prediction)

print("\nSafety Reasons:")
for reason in reasons:
    print("-", reason)

print("\nPrediction Probabilities:")
for cls, prob in zip(classes, probabilities):
    print(f"ESI {cls}: {prob:.2f}")
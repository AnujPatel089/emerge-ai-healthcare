"""
feature_schema.py

Stores and loads model feature columns.
This prevents Streamlit prediction errors caused by missing or extra columns.
"""

import joblib
import os


FEATURE_PATH = "models/feature_columns.pkl"


def save_feature_columns(columns):
    os.makedirs("models", exist_ok=True)
    joblib.dump(list(columns), FEATURE_PATH)


def load_feature_columns():
    return joblib.load(FEATURE_PATH)
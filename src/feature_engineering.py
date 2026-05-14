"""
Feature engineering for the EmergAI synthetic emergency triage dataset.
"""

from __future__ import annotations

import pandas as pd

from src.triage_rules import calculate_news2_score


def add_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically useful derived fields without mutating the caller's DataFrame."""
    out = df.copy()

    out["Blood_Pressure"] = out["Systolic_BP"].astype(int).astype(str) + "/" + out["Diastolic_BP"].astype(int).astype(str)
    out["Mean_Arterial_Pressure"] = (
        out["Diastolic_BP"] + (out["Systolic_BP"] - out["Diastolic_BP"]) / 3
    ).round(1)
    out["Pulse_Pressure"] = out["Systolic_BP"] - out["Diastolic_BP"]
    out["Shock_Index"] = (out["Heart_Rate"] / out["Systolic_BP"]).round(2)
    out["Comorbidity_Count"] = out[["Diabetes", "Hypertension", "Smoking"]].sum(axis=1)
    out["Abnormal_Vitals_Count"] = (
        (out["Systolic_BP"] < 90).astype(int)
        + (out["Heart_Rate"] > 120).astype(int)
        + (out["Respiratory_Rate"] > 24).astype(int)
        + (out["Oxygen_Saturation"] < 92).astype(int)
        + ((out["Temperature"] < 96.0) | (out["Temperature"] >= 101.0)).astype(int)
    )
    out["NEWS2_Score"] = out.apply(lambda row: calculate_news2_score(row.to_dict()), axis=1)
    out["Critical_Vital_Flag"] = (
        (out["Triage_Level"] <= 2)
        | (out["Oxygen_Saturation"] < 90)
        | (out["Systolic_BP"] < 90)
        | (out["Shock_Index"] >= 1.0)
    ).astype(int)
    return out


def model_feature_columns() -> list[str]:
    """Columns used by the new ESI ML model."""
    return [
        "Age",
        "Gender",
        "Systolic_BP",
        "Diastolic_BP",
        "Heart_Rate",
        "Respiratory_Rate",
        "Oxygen_Saturation",
        "Temperature",
        "Diabetes",
        "Hypertension",
        "Smoking",
        "Chest_Pain",
        "Shortness_of_Breath",
        "Fever",
        "Mean_Arterial_Pressure",
        "Pulse_Pressure",
        "Shock_Index",
        "Comorbidity_Count",
        "Abnormal_Vitals_Count",
        "NEWS2_Score",
        "Critical_Vital_Flag",
    ]


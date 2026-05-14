"""
Validation checks for synthetic emergency triage data.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Gender",
    "Blood_Pressure",
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
    "Symptom_Description",
    "Triage_Level",
    "ICU_Required",
    "Wait_Time_Minutes",
    "Hospital_Stay_Days",
    "Readmission_Risk",
]


def validate_emergency_dataset(df: pd.DataFrame, min_records: int = 10_000) -> list[str]:
    errors: list[str] = []
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    if missing:
        errors.append(f"Missing required columns: {missing}")
    if len(df) < min_records:
        errors.append(f"Dataset must contain at least {min_records:,} records.")
    if "Patient_ID" in df and df["Patient_ID"].duplicated().any():
        errors.append("Patient_ID must be unique.")
    if not missing and df[REQUIRED_COLUMNS].isna().any().any():
        errors.append("Required columns contain missing values.")

    checks = {
        "Age": df.get("Age", pd.Series(dtype=float)).between(1, 100),
        "Systolic_BP": df.get("Systolic_BP", pd.Series(dtype=float)).between(65, 230),
        "Diastolic_BP": df.get("Diastolic_BP", pd.Series(dtype=float)).between(35, 135),
        "Heart_Rate": df.get("Heart_Rate", pd.Series(dtype=float)).between(35, 190),
        "Respiratory_Rate": df.get("Respiratory_Rate", pd.Series(dtype=float)).between(6, 50),
        "Oxygen_Saturation": df.get("Oxygen_Saturation", pd.Series(dtype=float)).between(70, 100),
        "Temperature": df.get("Temperature", pd.Series(dtype=float)).between(95, 106),
        "Triage_Level": df.get("Triage_Level", pd.Series(dtype=float)).between(1, 5),
        "Wait_Time_Minutes": df.get("Wait_Time_Minutes", pd.Series(dtype=float)).between(0, 360),
        "Hospital_Stay_Days": df.get("Hospital_Stay_Days", pd.Series(dtype=float)).between(0, 30),
        "Readmission_Risk": df.get("Readmission_Risk", pd.Series(dtype=float)).between(0, 1),
    }
    for name, valid in checks.items():
        if len(valid) != len(df) or not valid.all():
            errors.append(f"{name} has values outside expected healthcare range.")

    if "Triage_Level" in df:
        critical = df["Triage_Level"] <= 2
        low_acuity = df["Triage_Level"] >= 4
        if critical.any() and low_acuity.any():
            if df.loc[critical, "Oxygen_Saturation"].mean() >= df.loc[low_acuity, "Oxygen_Saturation"].mean():
                errors.append("Critical patients should have lower oxygen saturation on average.")
            if df.loc[critical, "Heart_Rate"].mean() <= df.loc[low_acuity, "Heart_Rate"].mean():
                errors.append("Critical patients should have higher heart rate on average.")
            if df.loc[critical, "Wait_Time_Minutes"].mean() >= df.loc[low_acuity, "Wait_Time_Minutes"].mean():
                errors.append("Critical patients should have shorter wait times than non-urgent patients.")

    if "ICU_Required" in df and df["ICU_Required"].nunique() > 1:
        if df.loc[df["ICU_Required"] == 1, "NEWS2_Score"].mean() <= df.loc[df["ICU_Required"] == 0, "NEWS2_Score"].mean():
            errors.append("ICU patients should have higher NEWS2 severity scores.")

    if "Age" in df:
        older = df["Age"] >= 65
        if older.any() and (~older).any():
            if df.loc[older, "Readmission_Risk"].mean() <= df.loc[~older, "Readmission_Risk"].mean():
                errors.append("Older patients should have higher readmission risk on average.")

    if errors:
        raise ValueError("Emergency data validation failed:\n- " + "\n- ".join(errors))

    return [
        "Required columns present.",
        "Minimum record count met.",
        "No missing values in required columns.",
        "Clinical ranges passed.",
        "Acuity, ICU, wait-time, and readmission relationships passed.",
    ]


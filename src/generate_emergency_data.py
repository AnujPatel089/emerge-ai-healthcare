"""
Generate realistic synthetic emergency department data for EmergAI.

Run:
    python -m src.generate_emergency_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faker import Faker

from src.data_validation import validate_emergency_dataset
from src.feature_engineering import add_medical_features
from src.triage_rules import estimate_icu_risk, estimate_readmission_risk, recommended_wait_time_minutes, rule_based_esi


DEFAULT_RECORDS = 10_000
DEFAULT_SEED = 42
DEFAULT_DATA_PATH = Path("data/emergency_synthetic_data.csv")
DEFAULT_SUMMARY_PATH = Path("reports/emergency_synthetic_summary.txt")
DEFAULT_PLOT_PATH = Path("reports/emergency_synthetic_distributions.png")


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _ages(rng: np.random.Generator, n: int) -> np.ndarray:
    groups = rng.choice(["child", "young", "adult", "older", "elderly"], n, p=[0.08, 0.18, 0.35, 0.25, 0.14])
    out = np.zeros(n, dtype=int)
    ranges = {
        "child": (1, 17),
        "young": (18, 34),
        "adult": (35, 54),
        "older": (55, 74),
        "elderly": (75, 96),
    }
    for group, (low, high) in ranges.items():
        mask = groups == group
        out[mask] = rng.integers(low, high + 1, mask.sum())
    return out


def _condition_weights(age: int, diabetes: int, hypertension: int, smoking: int) -> np.ndarray:
    weights = np.array([0.15, 0.13, 0.14, 0.12, 0.09, 0.08, 0.08, 0.07, 0.06, 0.04, 0.04], dtype=float)
    if age >= 55 or diabetes or hypertension or smoking:
        weights[0] += 0.09
    if age >= 65 or smoking:
        weights[1] += 0.07
    if age >= 65 or hypertension:
        weights[4] += 0.05
    if age < 18:
        weights[9] += 0.08
    return weights / weights.sum()


def _symptom_text(fake: Faker, rng: np.random.Generator, age: int, condition: str, esi: int) -> str:
    severity = {
        1: "severe",
        2: "significant",
        3: "moderate",
        4: "mild",
        5: "minor",
    }[int(esi)]
    onset = rng.choice(["30 minutes", "2 hours", "6 hours", "1 day", "3 days", "1 week"])
    associated = {
        "chest pain": ["shortness of breath", "nausea", "diaphoresis", "left arm discomfort"],
        "shortness of breath": ["wheezing", "cough", "fatigue", "chest tightness"],
        "fever": ["chills", "body aches", "cough", "poor oral intake"],
        "abdominal pain": ["nausea", "vomiting", "diarrhea", "reduced appetite"],
        "neurologic deficit": ["slurred speech", "arm weakness", "facial droop", "confusion"],
        "trauma": ["localized swelling", "bleeding", "limited movement", "dizziness"],
        "syncope": ["lightheadedness", "palpitations", "weakness", "brief loss of consciousness"],
        "headache": ["photophobia", "nausea", "blurred vision", "neck stiffness"],
        "vomiting": ["dehydration", "abdominal cramps", "dizziness", "fever"],
        "minor injury": ["bruising", "small laceration", "localized pain", "normal movement"],
        "psychiatric crisis": ["anxiety", "agitation", "insomnia", "safety concerns"],
    }
    context = rng.choice([
        "arrived by private vehicle",
        "brought by family",
        "referred from urgent care",
        "symptoms worsening at home",
        "limited relief with home care",
        fake.city() + " EMS report available",
    ])
    return f"{age}-year-old reports {severity} {condition} for {onset} with {rng.choice(associated[condition])}; {context}."


def generate_emergency_dataset(n_records: int = DEFAULT_RECORDS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    age = _ages(rng, n_records)
    gender = rng.choice(["Female", "Male", "Nonbinary"], n_records, p=[0.505, 0.485, 0.01])
    smoking_prob = _clip(0.10 + (age >= 25) * 0.08 + (age >= 55) * 0.05, 0.03, 0.34)
    smoking = rng.binomial(1, smoking_prob)
    diabetes_prob = _clip(0.02 + np.maximum(age - 20, 0) * 0.0045 + smoking * 0.03, 0.01, 0.43)
    hypertension_prob = _clip(0.03 + np.maximum(age - 25, 0) * 0.007 + diabetes_prob * 0.25, 0.01, 0.64)
    diabetes = rng.binomial(1, diabetes_prob)
    hypertension = rng.binomial(1, hypertension_prob)

    conditions = np.array([
        rng.choice(
            [
                "chest pain",
                "shortness of breath",
                "fever",
                "abdominal pain",
                "neurologic deficit",
                "trauma",
                "syncope",
                "headache",
                "vomiting",
                "minor injury",
                "psychiatric crisis",
            ],
            p=_condition_weights(age[i], diabetes[i], hypertension[i], smoking[i]),
        )
        for i in range(n_records)
    ])

    chest_pain = (conditions == "chest pain").astype(int)
    shortness_of_breath = (conditions == "shortness of breath").astype(int)
    fever = (conditions == "fever").astype(int)

    preliminary_severity = rng.normal(0, 0.8, n_records)
    preliminary_severity += (age >= 75) * 0.7 + (age >= 60) * 0.25
    preliminary_severity += diabetes * 0.2 + hypertension * 0.2
    preliminary_severity += chest_pain * 0.9 + shortness_of_breath * 0.9 + fever * 0.25
    preliminary_severity += (conditions == "neurologic deficit") * 1.1 + (conditions == "syncope") * 0.55
    provisional_esi = np.select(
        [
            preliminary_severity >= 2.1,
            preliminary_severity >= 1.0,
            preliminary_severity >= -0.15,
            preliminary_severity >= -1.0,
        ],
        [1, 2, 3, 4],
        default=5,
    ).astype(int)

    acuity = 6 - provisional_esi
    systolic = rng.normal(126, 14, n_records) + hypertension * 17 + (age >= 65) * 8
    systolic -= (provisional_esi == 1) * rng.normal(25, 10, n_records)
    systolic -= (provisional_esi == 2) * rng.normal(8, 6, n_records)
    systolic = _clip(systolic, 72, 225).round().astype(int)

    diastolic = rng.normal(78, 9, n_records) + hypertension * 9 + (age >= 65) * 3
    diastolic -= (provisional_esi == 1) * rng.normal(12, 6, n_records)
    diastolic = _clip(diastolic, 42, 132).round().astype(int)

    heart_rate = rng.normal(78, 11, n_records) + acuity * 8
    heart_rate += fever * rng.normal(13, 4, n_records) + (conditions == "trauma") * rng.normal(7, 3, n_records)
    heart_rate += (provisional_esi == 1) * rng.normal(18, 7, n_records)
    heart_rate = _clip(heart_rate, 42, 185).round().astype(int)

    respiratory_rate = rng.normal(16, 2.2, n_records) + acuity * 1.2
    respiratory_rate += shortness_of_breath * rng.normal(5.5, 2, n_records)
    respiratory_rate += (provisional_esi == 1) * rng.normal(4, 2, n_records)
    respiratory_rate = _clip(respiratory_rate, 8, 45).round().astype(int)

    oxygen = rng.normal(98, 1.1, n_records) - acuity * 1.3
    oxygen -= shortness_of_breath * rng.normal(5.8, 2.4, n_records)
    oxygen -= chest_pain * (provisional_esi <= 2) * rng.normal(2.4, 1.1, n_records)
    oxygen -= (provisional_esi == 1) * rng.normal(5.4, 2.4, n_records)
    oxygen = _clip(oxygen, 72, 100).round().astype(int)

    temperature = rng.normal(98.4, 0.55, n_records) + fever * rng.normal(2.7, 0.7, n_records)
    temperature = _clip(temperature, 95.0, 105.3).round(1)

    records: list[dict[str, object]] = []
    for i in range(n_records):
        patient = {
            "Age": int(age[i]),
            "Gender": gender[i],
            "Systolic_BP": int(systolic[i]),
            "Diastolic_BP": int(diastolic[i]),
            "Heart_Rate": int(heart_rate[i]),
            "Respiratory_Rate": int(respiratory_rate[i]),
            "Oxygen_Saturation": int(oxygen[i]),
            "Temperature": float(temperature[i]),
            "Diabetes": int(diabetes[i]),
            "Hypertension": int(hypertension[i]),
            "Smoking": int(smoking[i]),
            "Chest_Pain": int(chest_pain[i]),
            "Shortness_of_Breath": int(shortness_of_breath[i]),
            "Fever": int(fever[i]),
        }
        esi, _ = rule_based_esi(patient)
        icu_risk = estimate_icu_risk(patient, esi)
        icu_required = int(rng.random() < icu_risk)
        wait = max(0, int(round(rng.normal(recommended_wait_time_minutes(esi), {1: 2, 2: 8, 3: 20, 4: 30, 5: 45}[esi]))))
        stay_base = {1: 5.8, 2: 3.2, 3: 1.4, 4: 0.45, 5: 0.12}[esi]
        stay = float(_clip(rng.gamma(1.8, max(stay_base, 0.1) / 1.8) + icu_required * rng.normal(3.5, 1.1), 0, 24))
        readmission = estimate_readmission_risk(patient, esi, icu_risk)
        records.append(
            {
                "Patient_ID": f"ER-{seed}-{i + 1:06d}",
                **patient,
                "Symptom_Description": _symptom_text(fake, rng, int(age[i]), str(conditions[i]), esi),
                "Presenting_Condition": conditions[i],
                "Triage_Level": esi,
                "ICU_Required": icu_required,
                "Wait_Time_Minutes": min(wait, 360),
                "Hospital_Stay_Days": round(stay, 1),
                "Readmission_Risk": readmission,
            }
        )

    df = add_medical_features(pd.DataFrame(records))
    ordered = [
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
        "Presenting_Condition",
        "Mean_Arterial_Pressure",
        "Pulse_Pressure",
        "Shock_Index",
        "Comorbidity_Count",
        "Abnormal_Vitals_Count",
        "NEWS2_Score",
        "Critical_Vital_Flag",
    ]
    return df[ordered]


def save_summary(df: pd.DataFrame, messages: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        "EmergAI Synthetic Emergency Dataset Summary",
        "=" * 52,
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        "",
        "Validation",
        "-" * 52,
        *[f"- {message}" for message in messages],
        "",
        "ESI Distribution (%)",
        "-" * 52,
        df["Triage_Level"].value_counts(normalize=True).sort_index().mul(100).round(1).to_string(),
        "",
        "ICU Required by ESI (%)",
        "-" * 52,
        df.groupby("Triage_Level")["ICU_Required"].mean().mul(100).round(1).to_string(),
        "",
        "Average Wait Time by ESI",
        "-" * 52,
        df.groupby("Triage_Level")["Wait_Time_Minutes"].mean().round(1).to_string(),
        "",
        "Numeric Summary",
        "-" * 52,
        df[[
            "Age",
            "Systolic_BP",
            "Heart_Rate",
            "Respiratory_Rate",
            "Oxygen_Saturation",
            "Temperature",
            "Wait_Time_Minutes",
            "Hospital_Stay_Days",
            "Readmission_Risk",
            "NEWS2_Score",
        ]].describe().round(2).to_string(),
    ]
    path.write_text("\n".join(summary), encoding="utf-8")


def save_visualizations(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EmergAI Synthetic Emergency Data", fontsize=16, fontweight="bold")

    df["Triage_Level"].value_counts().sort_index().plot(kind="bar", ax=axes[0, 0], color="#3b6ea8")
    axes[0, 0].set_title("ESI Distribution")
    df["ICU_Required"].map({0: "No ICU", 1: "ICU"}).value_counts().plot(kind="bar", ax=axes[0, 1], color="#b35c44")
    axes[0, 1].set_title("ICU Required")
    df.groupby("Triage_Level")["Wait_Time_Minutes"].mean().plot(kind="bar", ax=axes[0, 2], color="#6d7f2f")
    axes[0, 2].set_title("Average Wait Time by ESI")
    axes[1, 0].hist(df["Age"], bins=30, color="#4f8f6f", edgecolor="white")
    axes[1, 0].set_title("Age")
    sample = df.sample(min(2500, len(df)), random_state=DEFAULT_SEED)
    axes[1, 1].scatter(sample["Age"], sample["Readmission_Risk"], s=8, alpha=0.35, color="#c4783a")
    axes[1, 1].set_title("Readmission Risk by Age")
    df.groupby("Triage_Level")["Oxygen_Saturation"].mean().plot(kind="bar", ax=axes[1, 2], color="#9a5c93")
    axes[1, 2].set_title("Average Oxygen by ESI")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic emergency triage data.")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--plots", type=Path, default=DEFAULT_PLOT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.records < DEFAULT_RECORDS:
        raise ValueError(f"Generate at least {DEFAULT_RECORDS:,} records.")
    df = generate_emergency_dataset(args.records, args.seed)
    messages = validate_emergency_dataset(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    save_summary(df, messages, args.summary)
    save_visualizations(df, args.plots)
    print(f"Saved dataset: {args.output}")
    print(f"Saved summary: {args.summary}")
    print(f"Saved visualizations: {args.plots}")
    print("\n".join(messages))


if __name__ == "__main__":
    main()


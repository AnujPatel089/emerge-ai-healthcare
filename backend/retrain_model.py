"""
Model Retraining Pipeline
Emergency Triage AI
"""

import sys
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)

import shutil
import joblib
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from src.database import SessionLocal
from src.models import PredictionLog


MODEL_PATH = "models/triage_xgboost_balanced.pkl"
BACKUP_DIR = "models/backups"


FEATURE_COLUMNS = [
    "age",
    "gender",
    "race",
    "ethnicity",
    "arrivalmode",
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "cc_chestpain",
    "cc_shortnessofbreath",
    "cc_headache",
    "cc_fever",
    "cc_abdominalpain",
    "cc_dizziness",
    "cc_syncope",
    "cc_weakness"
]


def load_training_data_from_postgres():
    db = SessionLocal()

    try:
        logs = db.query(PredictionLog).all()

        rows = []

        for log in logs:

            if log.final_prediction is None:
                continue

            rows.append({
                "age": log.age,
                "gender": log.gender,
                "race": log.race,
                "ethnicity": log.ethnicity,
                "arrivalmode": log.arrivalmode,

                "triage_vital_hr": log.triage_vital_hr,
                "triage_vital_sbp": log.triage_vital_sbp,
                "triage_vital_dbp": log.triage_vital_dbp,
                "triage_vital_rr": log.triage_vital_rr,
                "triage_vital_o2": log.triage_vital_o2,
                "triage_vital_temp": log.triage_vital_temp,

                "cc_chestpain": log.cc_chestpain,
                "cc_shortnessofbreath": log.cc_shortnessofbreath,
                "cc_headache": log.cc_headache,
                "cc_fever": log.cc_fever,
                "cc_abdominalpain": log.cc_abdominalpain,
                "cc_dizziness": log.cc_dizziness,
                "cc_syncope": log.cc_syncope,
                "cc_weakness": log.cc_weakness,

                "target": str(log.final_prediction)
            })

        return pd.DataFrame(rows)

    finally:
        db.close()


def clean_target(value):

    value = str(value)

    value = value.replace("ESI", "")
    value = value.replace("esi", "")
    value = value.strip()

    try:
        return int(float(value)) - 1

    except Exception:
        return None


def preprocess_data(df):

    df = df.copy()

    df["target"] = df["target"].apply(clean_target)

    df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    categorical_columns = [
        "gender",
        "race",
        "ethnicity",
        "arrivalmode"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype(str)

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False
    )

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def backup_current_model():

    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = (
        f"{BACKUP_DIR}/"
        f"triage_xgboost_backup_{timestamp}.pkl"
    )

    if os.path.exists(MODEL_PATH):

        shutil.copy(MODEL_PATH, backup_path)

        print(f"Old model backed up to: {backup_path}")

    return backup_path


def retrain_model():

    print("=" * 60)
    print("Emergency Triage AI — Model Retraining")
    print("=" * 60)

    print("\nLoading PostgreSQL training data...")

    df = load_training_data_from_postgres()

    if df.empty:

        print("\nNo training data found.")
        return

    print(f"\nLoaded records: {len(df)}")

    if len(df) < 20:

        print(
            "\nNot enough records to retrain safely."
        )

        print("Need at least 20 records.")

        return

    X, y = preprocess_data(df)

    print("\nDataset prepared successfully.")

    print(f"Features shape: {X.shape}")
    print(f"Target classes: {y.unique()}")

    if y.nunique() < 2:

        print(
            "\nNeed at least 2 different ESI classes to train."
        )

        return

    print("\nSplitting train/test dataset...")

    class_counts = y.value_counts()
    can_stratify = y.nunique() > 1 and class_counts.min() >= 2

    stratify_value = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_value
    )

    print("\nTraining new XGBoost model...")

    new_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(y.unique()),
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
    )

    new_model.fit(X_train, y_train)

    print("\nModel training completed.")

    y_pred = new_model.predict(X_test)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    new_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("NEW MODEL ACCURACY")
    print("=" * 60)

    print(round(new_accuracy, 4))

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
    )

    print("\nBacking up old model...")

    backup_current_model()

    print("\nSaving new model...")

    joblib.dump(new_model, MODEL_PATH)

    feature_path = (
        "models/retrained_feature_columns.pkl"
    )

    joblib.dump(
        list(X.columns),
        feature_path
    )

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nNew model saved to:")
    print(MODEL_PATH)

    print(f"\nFeature columns saved to:")
    print(feature_path)


if __name__ == "__main__":
    retrain_model()
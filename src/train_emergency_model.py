"""
Train the EmergAI ESI model on the new synthetic emergency dataset.

Run:
    python -m src.generate_emergency_data
    python -m src.train_emergency_model
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from src.data_validation import validate_emergency_dataset
from src.feature_engineering import model_feature_columns


DATA_PATH = Path("data/emergency_synthetic_data.csv")
MODEL_PATH = Path("models/emergency_triage_model.pkl")
FEATURE_PATH = Path("models/emergency_feature_columns.pkl")
METRICS_PATH = Path("reports/emergency_model_metrics.txt")
CONFUSION_MATRIX_PATH = Path("reports/emergency_confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = Path("reports/emergency_feature_importance.png")


def build_model(random_state: int = 42) -> Pipeline:
    numeric_features = [
        "Age",
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
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    if HAS_XGBOOST:
        classifier = XGBClassifier(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=260,
            max_depth=14,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def save_feature_importance(model: Pipeline, output_path: Path) -> None:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return

    names = get_transformed_feature_names(model)
    importances = pd.DataFrame(
        {"Feature": names, "Importance": classifier.feature_importances_}
    ).sort_values("Importance", ascending=False).head(18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(importances["Feature"][::-1], importances["Importance"][::-1], color="#3b6ea8")
    plt.title("Emergency ESI Model Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} was not found. Run: python -m src.generate_emergency_data"
        )

    df = pd.read_csv(DATA_PATH)
    validate_emergency_dataset(df)

    features = model_feature_columns()
    X = df[features]
    y = df["Triage_Level"].astype(int) - 1 if HAS_XGBOOST else df["Triage_Level"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report_labels = [1, 2, 3, 4, 5]
    if HAS_XGBOOST:
        y_eval = y_test + 1
        y_pred_eval = y_pred + 1
    else:
        y_eval = y_test
        y_pred_eval = y_pred

    accuracy = accuracy_score(y_eval, y_pred_eval)
    report = classification_report(y_eval, y_pred_eval, labels=report_labels, digits=3)
    matrix = confusion_matrix(y_eval, y_pred_eval, labels=report_labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURE_PATH)

    metrics_text = "\n".join(
        [
            "EmergAI Emergency ESI Model Metrics",
            "=" * 48,
            f"Model type: {'XGBoost' if HAS_XGBOOST else 'RandomForest'}",
            f"Training rows: {len(X_train):,}",
            f"Test rows: {len(X_test):,}",
            f"Accuracy: {accuracy:.4f}",
            "",
            "Classification Report",
            "-" * 48,
            report,
        ]
    )
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=report_labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Emergency ESI Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()

    save_feature_importance(model, FEATURE_IMPORTANCE_PATH)

    print(metrics_text)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved feature columns: {FEATURE_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()


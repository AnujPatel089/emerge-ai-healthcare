"""
train_baseline.py

Baseline Emergency Triage AI Model

Goal:
Predict ESI triage level using core patient information.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------

df = pd.read_csv("data/triage.csv")

print("Original Shape:", df.shape)


# ---------------------------------------------------
# STEP 2: Select important columns
# ---------------------------------------------------

selected_columns = [

    # Target
    "esi",

    # Demographics
    "age",
    "gender",
    "race",
    "ethnicity",

    # Arrival information
    "arrivalmode",

    # Triage vitals
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",

    # Common symptoms / complaints
    "cc_chestpain",
    "cc_shortnessofbreath",
    "cc_headache",
    "cc_fever",
    "cc_abdominalpain",
    "cc_dizziness",
    "cc_syncope",
    "cc_weakness",
]

df = df[selected_columns]

print("Selected Shape:", df.shape)


# ---------------------------------------------------
# STEP 3: Remove missing target values
# ---------------------------------------------------

df = df.dropna(subset=["esi"])

print("After removing missing ESI:", df.shape)


# ---------------------------------------------------
# STEP 4: Separate features and target
# ---------------------------------------------------

X = df.drop(columns=["esi"])
y = df["esi"]


# ---------------------------------------------------
# STEP 5: Identify column types
# ---------------------------------------------------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ---------------------------------------------------
# STEP 6: Create preprocessing pipelines
# ---------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ---------------------------------------------------
# STEP 7: Create model pipeline
# ---------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# ---------------------------------------------------
# STEP 8: Train-test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ---------------------------------------------------
# STEP 9: Train model
# ---------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ---------------------------------------------------
# STEP 10: Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# STEP 11: Evaluation
# ---------------------------------------------------

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# STEP 12: Save model
# ---------------------------------------------------

joblib.dump(model, "models/triage_baseline_model.pkl")

print("\nModel saved successfully!")
"""
train_xgboost_balanced.py

Balanced XGBoost model.
Purpose:
Improve rare urgent classes like ESI 1.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


df = pd.read_csv("data/triage.csv")

selected_columns = [
    "esi", "age", "gender", "race", "ethnicity", "arrivalmode",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
    "triage_vital_rr", "triage_vital_o2", "triage_vital_temp",
    "cc_chestpain", "cc_shortnessofbreath", "cc_headache", "cc_fever",
    "cc_abdominalpain", "cc_dizziness", "cc_syncope", "cc_weakness",
]

df = df[selected_columns]
df = df.dropna(subset=["esi"])

X = df.drop(columns=["esi"])
y = df["esi"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "str"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

classifier = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Training balanced XGBoost model...")

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    os.path.join("models", "triage_xgboost_balanced.pkl")
)
joblib.dump(label_encoder, "models/esi_label_encoder.pkl")

print("\nBalanced XGBoost model saved successfully!")
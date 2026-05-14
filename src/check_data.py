"""
check_data.py

Purpose:
Understand the triage dataset before training the AI model.
"""

import pandas as pd

# Load CSV dataset
df = pd.read_csv("data/triage.csv")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nTarget column distribution:")
print(df["esi"].value_counts(dropna=False))

print("\nMissing values in important columns:")
important_cols = ["esi", "age", "gender", "race", "ethnicity", "lang"]
print(df[important_cols].isnull().sum())

print("\nData types:")
print(df[important_cols].dtypes)
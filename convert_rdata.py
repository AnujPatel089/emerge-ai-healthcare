"""
convert_rdata.py

Purpose:
Convert RData (.rdata/.RData) healthcare dataset into CSV format
for machine learning and AI development.
"""

import pyreadr
import pandas as pd


# ---------------------------------------------------
# STEP 1: File path to RData file
# ---------------------------------------------------

file_path = r"E:\Personal Project\Sem 2\emergency-triage-ai\data\5v_cleandf.rdata"


# ---------------------------------------------------
# STEP 2: Read RData file
# ---------------------------------------------------

result = pyreadr.read_r(file_path)


# ---------------------------------------------------
# STEP 3: Show available objects inside RData
# ---------------------------------------------------

print("\nAvailable objects inside RData file:")
print(result.keys())


# ---------------------------------------------------
# STEP 4: Extract dataframe
# ---------------------------------------------------

# Try automatic extraction
try:
    df = result[None]
except:
    # If automatic extraction fails,
    # manually select first object
    first_key = list(result.keys())[0]
    df = result[first_key]


# ---------------------------------------------------
# STEP 5: Show dataset preview
# ---------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# ---------------------------------------------------
# STEP 6: Save CSV
# ---------------------------------------------------

output_path = r"E:\Personal Project\Sem 2\emergency-triage-ai\data\triage.csv"

df.to_csv(output_path, index=False)

print("\nCSV conversion completed successfully!")
print(f"CSV saved at:\n{output_path}")
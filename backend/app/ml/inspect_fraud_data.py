
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "fraud_detection"
    / "fraud_oracle.csv"
)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# Check fraud distribution
target = "FraudFound_P"

if target in df.columns:
    print("\nFraud distribution:")
    print(df[target].value_counts())

    print("\nFraud percentage:")
    print(df[target].value_counts(normalize=True) * 100)
else:
    print(f"\nTarget column '{target}' not found!")

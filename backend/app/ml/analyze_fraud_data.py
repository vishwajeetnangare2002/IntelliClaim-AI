
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

df = pd.read_csv(DATA_PATH)

print("\n========== CATEGORICAL FEATURE ANALYSIS ==========")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))

print("\n========== NUMERICAL FEATURE ANALYSIS ==========")

print(df.describe().T)

print("\n========== POSSIBLE PLACEHOLDERS ==========")

for column in df.columns:
    if df[column].dtype == "object":
        suspicious = df[column].astype(str).str.strip().str.lower().isin(
            ["?", "unknown", "null", "nan", ""]
        )

        if suspicious.any():
            print(f"{column}: {suspicious.sum()} suspicious values")

print("\n========== FRAUD BY IMPORTANT FEATURES ==========")

for column in [
    "VehicleCategory",
    "VehiclePrice",
    "Fault",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "PastNumberOfClaims",
]:
    print(f"\n{column}")
    print(pd.crosstab(
        df[column],
        df["FraudFound_P"],
        normalize="index"
    ).round(3))

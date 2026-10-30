
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

print("\n========== AGE ZERO RECORDS ==========")

age_zero = df[df["Age"] == 0]

print("Count:", len(age_zero))
print(age_zero[["Age", "Sex", "MaritalStatus", "FraudFound_P"]].head(10))

print("\n========== INVALID CLAIM DATE VALUES ==========")

invalid_dates = df[
    (df["DayOfWeekClaimed"] == "0") |
    (df["MonthClaimed"] == "0")
]

print("Count:", len(invalid_dates))
print(invalid_dates[
    ["DayOfWeekClaimed", "MonthClaimed", "Year", "FraudFound_P"]
])

print("\n========== POLICY NUMBER CHECK ==========")

print("Unique policy numbers:", df["PolicyNumber"].nunique())
print("Total records:", len(df))
print("Sequential identifiers:",
      df["PolicyNumber"].is_monotonic_increasing)

print("\n========== FRAUD BY YEAR ==========")

print(pd.crosstab(
    df["Year"],
    df["FraudFound_P"],
    normalize="index"
).round(3))

print("\n========== FRAUD BY REPRESENTATIVE ==========")

rep_summary = df.groupby("RepNumber").agg(
    total_claims=("FraudFound_P", "count"),
    fraud_cases=("FraudFound_P", "sum"),
    fraud_rate=("FraudFound_P", "mean")
)

print(rep_summary.sort_values("fraud_rate", ascending=False))

print("\n========== INVESTIGATION COMPLETE ==========")

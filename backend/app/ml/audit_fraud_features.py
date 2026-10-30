
from pathlib import Path
import pandas as pd



# 1. PATH CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "fraud_detection"
    / "fraud_oracle.csv"
)



# 2. LOAD DATASET


df = pd.read_csv(DATA_PATH)

print("\n========== FRAUD FEATURE LEAKAGE AUDIT ==========")

print("Dataset shape:", df.shape)



# 3. DEFINE FEATURE AVAILABILITY


feature_categories = {

    "Potentially Available at Claim Submission": [
        "Month",
        "WeekOfMonth",
        "DayOfWeek",
        "Make",
        "AccidentArea",
        "Sex",
        "MaritalStatus",
        "Age",
        "PolicyType",
        "VehicleCategory",
        "VehiclePrice",
        "Deductible",
        "DriverRating",
        "AgeOfVehicle",
        "AgeOfPolicyHolder",
        "NumberOfCars",
        "BasePolicy"
    ],

    "Requires Timing / Business Process Verification": [
        "DayOfWeekClaimed",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "PastNumberOfClaims",
        "AddressChange_Claim"
    ],

    "Potentially Available Only During Investigation": [
        "Fault",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "NumberOfSuppliments"
    ],

    "Excluded Identifier Columns": [
        "PolicyNumber",
        "RepNumber"
    ],

    "Target Column": [
        "FraudFound_P"
    ]
}



# 4. PRINT FEATURE CLASSIFICATION


for category, features in feature_categories.items():

    print("\n" + "=" * 60)
    print(category)
    print("=" * 60)

    for feature in features:

        if feature in df.columns:
            print("-", feature)



# 5. CHECK FEATURE AVAILABILITY


print("\n========== FEATURE AVAILABILITY CHECK ==========")

for category, features in feature_categories.items():

    available_features = [
        feature for feature in features
        if feature in df.columns
    ]

    print(f"\n{category}: {len(available_features)} features")



# 6. CHECK POTENTIAL LEAKAGE FEATURES


potential_leakage = (
    feature_categories[
        "Potentially Available Only During Investigation"
    ]
)

print("\n========== POTENTIAL LEAKAGE FEATURES ==========")
for feature in potential_leakage:

    if feature in df.columns:

        print(f"\nFeature: {feature}")

        print("Unique values:")
        print(df[feature].value_counts().head(10))



# 7. FINAL SUMMARY


print("\n========== AUDIT SUMMARY ==========")

print("Total dataset columns:", len(df.columns))

print(
    "Target column:",
    "FraudFound_P"
)

print(
    "Potential investigation-stage features:",
    len(potential_leakage)
)

print("\nIMPORTANT:")
print(
    "Feature availability classifications are preliminary. "
    "Confirm them against the intended claim-processing workflow."
)

print(
    "Do not treat statistical association as proof of fraud "
    "or as proof of data leakage."
)

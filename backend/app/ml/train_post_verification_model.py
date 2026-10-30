
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "fraud_detection"
    / "fraud_oracle.csv"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "post_verification_fraud_model.joblib"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# Clean data
df["Age"] = df["Age"].replace(0, np.nan)

for column in ["DayOfWeekClaimed", "MonthClaimed"]:
    df[column] = df[column].replace("0", "Unknown")


# Post-verification features
features = [
    # Early submission features
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
    "BasePolicy",

    # Additional timing and verification features
    "DayOfWeekClaimed",
    "MonthClaimed",
    "WeekOfMonthClaimed",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AddressChange_Claim",

    # Investigation information assumed available
    # by this post-verification stage
    "Fault",
    "PoliceReportFiled",
    "WitnessPresent",
    "NumberOfSuppliments",
]

target = "FraudFound_P"

X = df[features].copy()
y = df[target]


# Identify feature types
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Total features:", len(features))
print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# Preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# Split: 64% train, 16% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val,
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# Build model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ),
    ]
)


# Train
print("\nTraining post-verification fraud model...")

pipeline.fit(X_train, y_train)


# Select threshold using validation data
val_probabilities = pipeline.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities,
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = f1_scores.argmax()

best_threshold = float(thresholds[best_index])
best_val_f1 = float(f1_scores[best_index])

validation_pr_auc = average_precision_score(
    y_val,
    val_probabilities,
)

print("\nSelected threshold:", round(best_threshold, 4))
print("Validation F1-score:", round(best_val_f1, 4))
print("Validation PR-AUC:", round(validation_pr_auc, 4))


# Evaluate on untouched test data
test_probabilities = pipeline.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

print("Test PR-AUC:", round(test_pr_auc, 4))
print("Test ROC-AUC:", round(test_roc_auc, 4))


# Save model package
model_package = {
    "model": pipeline,
    "threshold": best_threshold,
    "model_name": "Random Forest",
    "features": features,
    "validation_pr_auc": float(validation_pr_auc),
    "validation_f1": best_val_f1,
    "pr_auc": float(test_pr_auc),
    "roc_auc": float(test_roc_auc),
}

joblib.dump(model_package, MODEL_PATH)

print("\nPost-verification model saved successfully!")
print("Model path:", MODEL_PATH)

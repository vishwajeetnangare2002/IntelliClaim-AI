
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)



# 1. PATH CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "fraud_detection"
    / "fraud_oracle.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "models"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "early_fraud_model.joblib"



# 2. LOAD DATASET


df = pd.read_csv(DATA_PATH)

print("\n========== EARLY FRAUD DATASET ==========")

print("Dataset shape:", df.shape)



# 3. DATA CLEANING


df["Age"] = df["Age"].replace(0, np.nan)



# 4. SELECT EARLY-STAGE FEATURES


TARGET = "FraudFound_P"

early_features = [
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
]

X = df[early_features].copy()

y = df[TARGET]

print("\n========== FEATURE INFORMATION ==========")

print("Total features:", len(early_features))

print("\nSelected features:")

for feature in early_features:
    print("-", feature)



# 5. IDENTIFY FEATURE TYPES


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))



# 6. DATA PREPROCESSING


numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)



# 7. TRAIN / VALIDATION / TEST SPLIT


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nTraining fraud distribution:")

print(y_train.value_counts())



# 8. DEFINE RANDOM FOREST MODEL


classifier = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)



# 9. CREATE ML PIPELINE


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)



# 10. TRAIN MODEL


print("\n========== MODEL TRAINING ==========")

pipeline.fit(X_train, y_train)

print("Early fraud model training completed.")



# 11. VALIDATION AND THRESHOLD TUNING


val_probabilities = pipeline.predict_proba(X_val)[:, 1]

val_ap_score = average_precision_score(
    y_val,
    val_probabilities
)

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-10
    )
)

best_index = np.argmax(f1_scores)

best_threshold = float(thresholds[best_index])

best_validation_f1 = float(f1_scores[best_index])

print("\n========== VALIDATION RESULTS ==========")

print("Selected threshold:", round(best_threshold, 4))

print("Validation F1-score:", round(best_validation_f1, 4))

print("Validation PR-AUC:", round(float(val_ap_score), 4))



# 12. TEST MODEL


test_probabilities = pipeline.predict_proba(X_test)[:, 1]

predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_ap_score = average_precision_score(
    y_test,
    test_probabilities
)

test_roc_score = roc_auc_score(
    y_test,
    test_probabilities
)

print("\n========== TEST RESULTS ==========")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("Test PR-AUC:", round(test_ap_score, 4))

print("Test ROC-AUC:", round(test_roc_score, 4))



# 13. SAVE MODEL


model_package = {
    "model": pipeline,
    "threshold": best_threshold,
    "model_name": "Random Forest - Early Fraud Screening",
    "features": early_features,
    "validation_pr_auc": float(val_ap_score),
    "validation_f1": best_validation_f1,
    "test_pr_auc": float(test_ap_score),
    "test_roc_auc": float(test_roc_score)
}

joblib.dump(
    model_package,
    MODEL_PATH
)



# 14. FINAL SUMMARY


print("\n" + "=" * 55)

print("EARLY FRAUD MODEL TRAINING COMPLETED")

print("=" * 55)

print("Model: Random Forest")

print("Threshold:", round(best_threshold, 4))

print("Validation PR-AUC:", round(val_ap_score, 4))

print("Test PR-AUC:", round(test_ap_score, 4))

print("Test ROC-AUC:", round(test_roc_score, 4))

print("\nModel saved to:")

print(MODEL_PATH)

print(
    "\nNOTE: Experimental model trained using historical "
    "public insurance data. It is not validated for "
    "real insurance decisions."
)

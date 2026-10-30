
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

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

MODEL_PATH = MODEL_DIR / "fraud_detection_model.joblib"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET ==========")
print("Dataset shape:", df.shape)


# ==========================================
# 3. DATA CLEANING
# ==========================================

# Treat zero age as missing
df["Age"] = df["Age"].replace(0, np.nan)

# Replace invalid date categories
df["DayOfWeekClaimed"] = (
    df["DayOfWeekClaimed"].replace("0", "Unknown")
)

df["MonthClaimed"] = (
    df["MonthClaimed"].replace("0", "Unknown")
)


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

TARGET = "FraudFound_P"

excluded_columns = [
    TARGET,
    "PolicyNumber",
    "RepNumber"
]

X = df.drop(columns=excluded_columns)
y = df[TARGET]

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ==========================================
# 5. DATA PREPROCESSING
# ==========================================

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


# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# First split: 80% train + validation, 20% test

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 80% training, 20% validation
# Overall split: 64% / 16% / 20%

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


# ==========================================
# 7. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )

}


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", model_name)
    print("=" * 55)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # --------------------------------------
    # Train model
    # --------------------------------------

    pipeline.fit(X_train, y_train)

    # --------------------------------------
    # Validation predictions
    # --------------------------------------

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    # Calculate validation PR-AUC
    val_ap_score = average_precision_score(
        y_val,
        val_probabilities
    )

    # --------------------------------------
    # Threshold tuning on validation data
    # --------------------------------------

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

    print("\nSelected threshold:", round(best_threshold, 4))

    print(
        "Validation F1-score:",
        round(best_validation_f1, 4)
    )

    print(
        "Validation PR-AUC:",
        round(float(val_ap_score), 4)
    )

    # --------------------------------------
    # Final evaluation on test data
    # --------------------------------------

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    ap_score = average_precision_score(
        y_test,
        test_probabilities
    )

    roc_score = roc_auc_score(
        y_test,
        test_probabilities
    )

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

    print(
        "Test PR-AUC (Average Precision):",
        round(ap_score, 4)
    )

    print(
        "Test ROC-AUC:",
        round(roc_score, 4)
    )

    # --------------------------------------
    # Store results
    # --------------------------------------

    results[model_name] = {
        "pipeline": pipeline,
        "validation_pr_auc": float(val_ap_score),
        "pr_auc": float(ap_score),
        "roc_auc": float(roc_score),
        "threshold": best_threshold,
        "validation_f1": best_validation_f1
    }


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

# IMPORTANT:
# Select model using validation PR-AUC.
# Do not use test PR-AUC for model selection.

best_model_name = max(
    results,
    key=lambda name: results[name]["validation_pr_auc"]
)

best_result = results[best_model_name]

best_model = best_result["pipeline"]
best_threshold = best_result["threshold"]


# ==========================================
# 10. SAVE MODEL AND THRESHOLD
# ==========================================

model_package = {
    "model": best_model,
    "threshold": best_threshold,
    "model_name": best_model_name,
    "validation_pr_auc": best_result["validation_pr_auc"],
    "pr_auc": best_result["pr_auc"],
    "roc_auc": best_result["roc_auc"]
}

joblib.dump(
    model_package,
    MODEL_PATH
)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("FRAUD DETECTION TRAINING COMPLETED")
print("=" * 55)

print("Selected model:", best_model_name)

print(
    "Selected threshold:",
    round(best_threshold, 4)
)

print(
    "Validation PR-AUC:",
    round(best_result["validation_pr_auc"], 4)
)

print(
    "Test PR-AUC:",
    round(best_result["pr_auc"], 4)
)

print(
    "Test ROC-AUC:",
    round(best_result["roc_auc"], 4)
)

print("\nModel saved to:")
print(MODEL_PATH)

print(
    "\nNOTE: Experimental model trained on historical "
    "public dataset records. It is not validated for "
    "real insurance decisions."
)

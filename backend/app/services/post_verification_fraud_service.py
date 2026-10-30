
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "models"
    / "post_verification_fraud_model.joblib"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


def predict_post_verification_fraud(claim_data: dict) -> dict:

    missing_features = [
        feature
        for feature in features
        if feature not in claim_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    input_data = pd.DataFrame(
        [{feature: claim_data[feature] for feature in features}]
    )

    fraud_score = float(
        model.predict_proba(input_data)[0][1]
    )

    fraud_flag = int(fraud_score >= threshold)

    return {
        "fraud_flag": fraud_flag,
        "fraud_status": (
            "Potential Fraud - Requires Investigation"
            if fraud_flag
            else "No Fraud Flag"
        ),
        "fraud_score": round(fraud_score, 4),
        "decision_threshold": round(float(threshold), 4),
        "note": (
            "Experimental prediction using a historical dataset. "
            "This score is not proof of fraud and must not be used "
            "as an automatic claim rejection decision."
        ),
    }

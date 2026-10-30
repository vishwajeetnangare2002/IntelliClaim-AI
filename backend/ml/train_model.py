from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler



PROJECT_ROOT = Path(__file__).resolve().parents[2]

data_path = PROJECT_ROOT / "backend" / "data" / "claim_risk_data.csv"

model_path = PROJECT_ROOT / "backend" / "ml" / "claim_risk_model.joblib"


def train_model():
    df=pd.read_csv(data_path)
    
    x=df[
        ["claim_amount","incident_type","previous_claims","days_to_report",]        
    ]

    y = df["risk_label"]
    
    # Preprocess numerical and categorical columns
    
    numerical_features = ["claim_amount","previous_claims","days_to_report",]
    
    categorical_features = ["incident_type"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_features),
            ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features,),    
        ]
    )
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    
    print("\n--- Model Evaluation ---")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    
    
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Lower Risk (0)", "Higher Risk (1)"],
            zero_division=0,
        )
    )
    
    
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    
if __name__ == "__main__":
    train_model()
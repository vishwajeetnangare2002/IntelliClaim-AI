from pathlib import Path

import joblib
import pandas as pd

project_root=Path(__file__).resolve().parents[3]

model_path=(project_root/"backend"/"ml"/"claim_risk_model.joblib")

model=joblib.load(model_path)

def predict_claim_risk(claim_amount:float,incident_type:str,previous_claims:int,days_to_report:int,):
    
    input_data=pd.DataFrame(
     [
        {
         "claim_amount": claim_amount,
         "incident_type": incident_type,
         "previous_claims": previous_claims,
         "days_to_report": days_to_report,
        }
     ]
  )

    prediction=int(model.predict(input_data)[0])
    
    probability=model.predict_proba(input_data)[0][1]
    return {
        "risk_label": prediction,
        "risk_category": ( "Higher Risk" if prediction == 1 else "Lower Risk"),
        "risk_probability": round(float(probability), 4),
                "note": "Experimental prediction from synthetic training data. "
                        "Not suitable for real claim decisions."
            }
            
        
        
        
    
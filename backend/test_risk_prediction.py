from backend.app.services.risk_prediction_service import predict_claim_risk

result = predict_claim_risk(
    claim_amount=45000,
    incident_type="Vehicle Accident",
    previous_claims=3,
    days_to_report=20,
)

print(result)
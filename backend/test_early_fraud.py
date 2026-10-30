
from backend.app.services.early_fraud_service import (
    predict_early_fraud,
)

sample_claim = {
    "Month": "Dec",
    "WeekOfMonth": 3,
    "DayOfWeek": "Friday",
    "Make": "Honda",
    "AccidentArea": "Urban",
    "Sex": "Male",
    "MaritalStatus": "Single",
    "Age": 30,
    "PolicyType": "Sedan - Collision",
    "VehicleCategory": "Sedan",
    "VehiclePrice": "20,000 to 29,000",
    "Deductible": 400,
    "DriverRating": 1,
    "AgeOfVehicle": "3 years",
    "AgeOfPolicyHolder": "26 to 30",
    "NumberOfCars": "1 vehicle",
    "BasePolicy": "Collision",
}

result = predict_early_fraud(sample_claim)

print("\nEarly Fraud Prediction:")
print(result)

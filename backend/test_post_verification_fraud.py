
from backend.app.services.post_verification_fraud_service import (
    predict_post_verification_fraud,
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

    "DayOfWeekClaimed": "Friday",
    "MonthClaimed": "Dec",
    "WeekOfMonthClaimed": 3,
    "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30",
    "PastNumberOfClaims": "none",
    "AddressChange_Claim": "no change",

    "Fault": "Policy Holder",
    "PoliceReportFiled": "Yes",
    "WitnessPresent": "Yes",
    "NumberOfSuppliments": "none",
}

result = predict_post_verification_fraud(sample_claim)

print("\nPost-Verification Fraud Prediction:")
print(result)

import pandas as pd
import joblib

# Load Model
model = joblib.load(
    "outputs/vendor_xgboost_model.pkl"
)

# Vendor Data
'''vendor = pd.DataFrame([{
    "vendornumber": 1128,
    "brand": 1233,
    "purchaseprice": 26.27,
    "actualprice": 36.99,
    "volume": 1750.0,
    "totalpurchasequantity": 145080,
    "totalpurchasedollars": 3811251.6,
    "totalexcisetax": 260999.2,
    "freightcost": 68601.68,
    "stockturnover": 0.9791080783016268
}])'''

vendor = pd.DataFrame([{
    "vendornumber": 999,
    "brand": 100,
    "purchaseprice": 10,
    "actualprice": 15,
    "volume": 750,
    "totalpurchasequantity": 1000,
    "totalpurchasedollars": 1000,
    "totalexcisetax": 50,
    "freightcost": 100,
    "stockturnover": 0.1
}])

prediction = model.predict(vendor)

probability = model.predict_proba(vendor)

print(
    "\nPrediction:",
    prediction[0]
)

print(
    "Low Sales Probability:",
    round(probability[0][0] * 100, 2),
    "%"
)

print(
    "High Sales Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)


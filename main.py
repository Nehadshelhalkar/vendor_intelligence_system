import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import VendorInput
from app.model_loader import model

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Vendor Intelligence API Running"}


@app.post("/predict")
def predict(data: VendorInput):

    features = [[
        data.vendornumber,
        data.brand,
        data.purchaseprice,
        data.actualprice,
        data.volume,
        data.totalpurchasequantity,
        data.totalpurchasedollars,
        data.totalexcisetax,
        data.freightcost,
        data.stockturnover
    ]]

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    low_prob = round(float(probability[0][0]) * 100, 2)
    high_prob = round(float(probability[0][1]) * 100, 2)

    # Risk Classification
    if high_prob < 30:
        risk = "High Risk"
    elif high_prob < 70:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "prediction": int(prediction[0]),
        "low_sales_probability": low_prob,
        "high_sales_probability": high_prob,
        "risk": risk
    }

@app.get("/dashboard-summary")
def dashboard_summary():

    df = pd.read_csv(
        "data/processed/vendor_sales_summary_processed.csv"
    )

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return {
        "total_vendors": int(len(df)),
        "total_sales": float(df["totalsalesdollars"].sum()),
        "total_purchase": float(df["totalpurchasedollars"].sum()),
        "average_profit_margin": float(df["profitmargin"].mean())
    }


@app.get("/vendors")
def vendors():

    df = pd.read_csv(
        "data/processed/vendor_sales_summary_processed.csv"
    )

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df.head(100).to_dict(orient="records")


@app.get("/feature-importance")
def feature_importance():

    df = pd.read_csv(
        "outputs/feature_importance.csv"
    )

    return df.to_dict(orient="records")
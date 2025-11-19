from pathlib import Path
from typing import Literal


import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# MODELS

class Transaction(BaseModel):
    transaction_id: str
    user_id: int

    amount: float
    currency: str
    country: str
    merchant_category: str
    time_of_day: int # 0-23

    device_trust_score: float # 0.0 - 1.0
    num_tx_last_24h: int
    avg_amount_last_24h: float

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    action: Literal["allow", "review", "block"]


# Loading modeel
MODEL_PATH = Path("fraud/models/model.pkl")
print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Fraud Detection API", version="0.1.0")


# Helper function
"""
Rules: 
    < 0.3 = allow
    0.3 - 0.7 = review
    > 0.7 = block
"""
def decide_action(fraud_score: float) -> str:
    if fraud_score < 0.3:
        return "allow"
    elif fraud_score < 0.7:
        return "review"
    else:
        return "block"

# Routes

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(tx: Transaction):
    """
    rules: score > 0.5 = fraud
    """
    
    # single row DataFrame of same features used in trraining
    features = pd.DataFrame(
        [
            {
                "amount": tx.amount,
                "country": tx.country,
                "merchant_category": tx.merchant_category,
                "time_of_day": tx.time_of_day,
                "device_trust_score": tx.device_trust_score,
                "num_tx_last_24h": tx.num_tx_last_24h,
                "avg_amount_last_24h": tx.avg_amount_last_24h,
            }
        ]
    )

    # Get probability
    proba = model.predict_proba(features)[0, 1]
    fraud_score = float(proba)
    is_fraud = fraud_score > 0.5
    action = decide_action(fraud_score)

    return PredictionResponse(
            transaction_id = tx.transaction_id,
            fraud_score = fraud_score,
            is_fraud = is_fraud,
            action = action,
    )



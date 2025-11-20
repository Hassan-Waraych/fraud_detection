from pathlib import Path
from typing import Literal
from datetime import datetime
import json

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


# paths
MODEL_PATH = Path("fraud/models/model.pkl")
LOG_PATH = Path("logs/predictions.csv")

#Loading model
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

def log_prediction(tx: Transaction, fraud_score: float, is_fraud: bool, action: str):
    # Add prediction info to logs/predictions.csv
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_id": tx.transaction_id,
        "user_id": tx.user_id,
        "amount": tx.amount,
        "currency": tx.currency,
        "country": tx.country,
        "merchant_category": tx.merchant_category,
        "time_of_day": tx.time_of_day,
        "device_trust_score": tx.device_trust_score,
        "num_tx_last_24h": tx.num_tx_last_24h,
        "avg_amount_last_24h": tx.avg_amount_last_24h,
        "fraud_score": fraud_score,
        "is_fraud": int(is_fraud),
        "action": action,
        "raw_request": json.dumps(tx.model_dump()),
    }

    # create file w/ header
    file_exists = LOG_PATH.exists()
    import csv
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
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

    #Log prediction
    log_prediction(tx, fraud_score, is_fraud, action)

    return PredictionResponse(
            transaction_id = tx.transaction_id,
            fraud_score = fraud_score,
            is_fraud = is_fraud,
            action = action,
    )



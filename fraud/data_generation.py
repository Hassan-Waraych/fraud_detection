
import numpy as np
import pandas as pd

def generate_transactions(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # basic configs for data
    user_ids = np.arange(1, 5001)  # 5000 users / data points
    countries = ["US", "CA", "GB", "FR", "DE", "IN", "BR"] 
    merchant_categories = [
        "groceries",
        "electronics",
        "fashion",
        "gaming",
        "travel",
        "restaurants",
        "gambling",
    ]

    rows = []

    for i in range(n_samples):
        user_id = int(rng.choice(user_ids))

        # home country of user
        home_country = rng.choice(countries)

        # most transactions are from home country (85%)
        if rng.random() < 0.85:
            tx_country = home_country
        else:
            tx_country = rng.choice([c for c in countries if c != home_country])

        amount = float(rng.gamma(shape=2.0, scale=50.0))  # skewed towards smaller amounts

        currency = "USD"  # standard for now
        merchant_category = rng.choice(merchant_categories) #random category
        time_of_day = int(rng.integers(0, 24))  # hour 0..23

        device_trust_score = float(rng.uniform(0.0, 1.0))

        num_tx_last_24h = int(rng.poisson(lam=2.0))  # avg 2 transactions/day
        avg_amount_last_24h = float(
            max(1.0, rng.normal(loc=60.0, scale=30.0))
        )  # simple normal with min clamp

        # Fraud probability Rules
        fraud_prob = 0.02  # 2% base

        # Big transactions are riskier
        if amount > 500:
            fraud_prob += 0.05
        if amount > 2000:
            fraud_prob += 0.15
        if amount > 5000:
            fraud_prob += 0.25

        # Transactions from non-home country are riskier
        if tx_country != home_country:
            fraud_prob += 0.20

        #  low device trust score (new/suspicious device)
        if device_trust_score < 0.2:
            fraud_prob += 0.25

        # Many transactions in short time window
        if num_tx_last_24h > 10:
            fraud_prob += 0.20

        # Certain merchant categories are higher risk
        if merchant_category in ["gambling", "gaming", "travel"]:
            fraud_prob += 0.05

        # Cap fraud probability at 0.95 so it stays valid
        fraud_prob = min(fraud_prob, 0.95)

        is_fraud = rng.random() < fraud_prob

        row = {
            "transaction_id": f"tx_{i+1}",
            "user_id": user_id,
            "home_country": home_country,
            "country": tx_country,
            "amount": amount,
            "currency": currency,
            "merchant_category": merchant_category,
            "time_of_day": time_of_day,
            "device_trust_score": device_trust_score,
            "num_tx_last_24h": num_tx_last_24h,
            "avg_amount_last_24h": avg_amount_last_24h,
            "is_fraud": int(is_fraud),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def main():
    df = generate_transactions(n_samples=50000)
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index = False)
    print(f"Saved {len(df)} transactions to {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()

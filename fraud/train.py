import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path: str = "data/transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

"""
sklearn pipeline which uses one-hot encoding for categories,
scales numeric features and trains a RandomForest classifier
"""
def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


def train_and_evaluate(df: pd.DataFrame):
    # Features
    feature_cols = [
        "amount",
        "country",
        "merchant_category",
        "time_of_day",
        "device_trust_score",
        "num_tx_last_24h",
        "avg_amount_last_24h",
    ]

    target_col = "is_fraud" # our target column

    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = ["country", "merchant_category"]
    numeric_cols = [
        "amount",
        "time_of_day",
        "device_trust_score",
        "num_tx_last_24h",
        "avg_amount_last_24h",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = build_pipeline(categorical_cols, numeric_cols) # Pipeline obj
    clf.fit(X_train, y_train) # Fit into training data

    # Evaluate on Test Data
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) # run accuracy score check on correctly classified samples
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = None

    print(f"Test Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC:      {auc:.4f}")
    else:
        print("ROC AUC:      could not be computed (only one class present).")

    return clf


def save_model(model: Pipeline, path: str = "fraud/models/model.pkl"):
    # Ensure directory exists
    model_path = Path(path)
    os.makedirs(model_path.parent, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


def main():
    df = load_data("data/transactions.csv")
    print(f"Loaded {len(df)} rows from data/transactions.csv")
    print(df.head())

    model = train_and_evaluate(df)
    save_model(model)


if __name__ == "__main__":
    main()


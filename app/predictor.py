import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_churn_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_churn(data: dict) -> dict:
    df = pd.DataFrame([data])
    

    # Match training-time feature engineering
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["avg_charge_per_tenure"] = df["avg_charge_per_tenure"].clip(upper=1000)
    df["family_flag"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["is_long_term"] = (df["tenure"] >= 24).astype(int)

    # Match training-time cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }

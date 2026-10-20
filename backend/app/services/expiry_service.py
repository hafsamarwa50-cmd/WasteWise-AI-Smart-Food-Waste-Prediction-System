from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "ml" / "artifacts" / "expiry_risk_model.joblib"

MODEL_VERSION = "1.0.0"


class ExpiryRiskService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Expiry-risk model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, data):
        # Never recommend selling a batch past its expiry date.
        if data.days_to_expiry < 0:
            return {
                "risk_label": "expired",
                "risk_probability": 1.0,
                "model_version": MODEL_VERSION,
                "sale_blocked": True,
                "message": (
                    "Batch is past expiry. Do not recommend it for sale."
                ),
            }

        features = pd.DataFrame([{
            "category": data.category,
            "initial_quantity": data.initial_quantity,
            "days_to_expiry": data.days_to_expiry,
            "avg_daily_sales_7d": data.avg_daily_sales_7d,
            "unit_price": data.unit_price,
            "unit_cost": data.unit_cost,
            "discount_pct": data.discount_pct,
        }])

        probability = float(
            self.model.predict_proba(features)[0][1]
        )
        prediction = int(probability >= 0.5)

        return {
            "risk_label": "high" if prediction == 1 else "low",
            "risk_probability": round(probability, 4),
            "model_version": MODEL_VERSION,
            "sale_blocked": False,
            "message": (
                "High waste-risk score. Review stock and demand."
                if prediction == 1
                else "Lower predicted waste risk. Continue monitoring."
            ),
        }


expiry_risk_service = ExpiryRiskService()

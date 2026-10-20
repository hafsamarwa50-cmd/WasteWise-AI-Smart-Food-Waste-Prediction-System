from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

inventory = pd.read_csv(
    DATA_DIR / "inventory_batches.csv"
)

print("=== EXPIRY-RISK LABEL INVESTIGATION ===")

print("\nColumns:")
print(inventory.columns.tolist())

print("\nExpiry-risk labels:")
print(inventory["expiry_risk_label"].value_counts())

print("\nDays to expiry by risk label:")
print(
    inventory.groupby("expiry_risk_label")["days_to_expiry"]
    .agg(["count", "min", "max", "mean", "median"])
    .round(2)
)

print("\nAverage remaining quantity by risk label:")
print(
    inventory.groupby("expiry_risk_label")["quantity_remaining"]
    .mean()
    .round(2)
)

print("\nRisk labels for batches already past expiry:")
print(
    inventory.loc[
        inventory["days_to_expiry"] < 0,
        ["days_to_expiry", "expiry_risk_label"]
    ].value_counts().sort_index()
)

print("\nInvestigation completed.")
print("These checks describe relationships; they do not prove")
print("how the labels were originally generated.")

print("\n=== SAMPLE HIGH-RISK RECORDS ===")

print(
    inventory.loc[
        inventory["expiry_risk_label"] == 1,
        [
            "days_to_expiry",
            "quantity_remaining",
            "avg_daily_sales_7d",
            "expected_demand_before_expiry",
            "waste_units_actual_simulated",
            "expiry_risk_label"
        ]
    ].head(10).to_string(index=False)
)

print("\n=== SAMPLE LOW-RISK EXPIRED RECORDS ===")

print(
    inventory.loc[
        (inventory["days_to_expiry"] < 0)
        & (inventory["expiry_risk_label"] == 0),
        [
            "days_to_expiry",
            "quantity_remaining",
            "avg_daily_sales_7d",
            "expected_demand_before_expiry",
            "waste_units_actual_simulated",
            "expiry_risk_label"
        ]
    ].head(10).to_string(index=False)
)

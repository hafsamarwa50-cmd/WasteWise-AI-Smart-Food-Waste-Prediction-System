from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

file_path = ROOT / "data" / "processed" / "inventory_batches.csv"
df = pd.read_csv(file_path)

print("=== DATASET CHECK ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n=== EXPIRY DAYS ===")
print(df["days_to_expiry"].describe())

print("\n=== EXISTING LABEL VS EXPIRY ===")
print(
    pd.crosstab(
        df["days_to_expiry"] < 0,
        df["expiry_risk_label"],
        rownames=["Already expired"],
        colnames=["Existing risk label"]
    )
)

print("\n=== LABEL VS REMAINING STOCK ===")
print(
    df.groupby("expiry_risk_label")["quantity_remaining"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\n=== LABEL VS SIMULATED WASTE ===")
print(
    df.groupby("expiry_risk_label")["waste_units_actual_simulated"]
    .agg(["mean", "median", "min", "max"])
)

print("\n=== LABEL VS EXPECTED DEMAND ===")
print(
    df.groupby("expiry_risk_label")["expected_demand_before_expiry"]
    .agg(["mean", "median", "min", "max"])
)

print("\n=== INVESTIGATION COMPLETE ===")

print("\n=== PROPOSED RISK RULE CHECK ===")

not_expired = df[df["days_to_expiry"] >= 0].copy()

not_expired["proposed_expiry_risk"] = (
    not_expired["quantity_remaining"]
    > not_expired["expected_demand_before_expiry"]
).astype(int)

print(
    "\nProposed labels for non-expired batches:"
)
print(not_expired["proposed_expiry_risk"].value_counts().sort_index())

print(
    "\nExisting vs proposed labels:"
)
print(
    pd.crosstab(
        not_expired["expiry_risk_label"],
        not_expired["proposed_expiry_risk"],
        rownames=["Existing label"],
        colnames=["Proposed label"]
    )
)

print("\nNon-expired batches:", len(not_expired))
print("Expired batches excluded:", (df["days_to_expiry"] < 0).sum())

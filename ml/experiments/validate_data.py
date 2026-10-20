from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

inventory_path = DATA_DIR / "inventory_batches.csv"
discount_path = DATA_DIR / "discount_scenarios_simulated.csv"

inventory = pd.read_csv(inventory_path)
discount = pd.read_csv(discount_path)

print("=== INVENTORY BATCH VALIDATION ===")

print("\nDataset shape:", inventory.shape)

print("\nExpiry-risk label counts:")
print(inventory["expiry_risk_label"].value_counts(dropna=False))

print("\nDays to expiry validation:")
calculated_days = (
    pd.to_datetime(inventory["expiry_date"])
    - pd.to_datetime(inventory["snapshot_date"])
).dt.days

matches = calculated_days.eq(inventory["days_to_expiry"])

print("Rows matching calculated days:", matches.sum())
print("Rows with mismatched days:", (~matches).sum())

print("\nNegative remaining quantities:")
print((inventory["quantity_remaining"] < 0).sum())

print("\nRemaining quantity greater than initial quantity:")
print(
    (
        inventory["quantity_remaining"]
        > inventory["initial_quantity"]
    ).sum()
)

print("\n=== SIMULATED DISCOUNT VALIDATION ===")

print("\nDataset shape:", discount.shape)

print("\nDiscount levels:")
print(sorted(discount["discount_pct"].unique()))

print("\nMissing values:", discount.isna().sum().sum())
print("Duplicate rows:", discount.duplicated().sum())

print("\nDays to expiry range:")
print(
    discount["days_to_expiry"].min(),
    "to",
    discount["days_to_expiry"].max()
)

print("\nAverage simulated outcomes by discount:")
print(
    discount.groupby("discount_pct")[
        [
            "estimated_units_sold_simulated",
            "estimated_revenue_simulated",
            "estimated_gross_margin_simulated"
        ]
    ].mean().round(2)
)

print("\nValidation checks completed.")
print("Note: Discount outcomes are simulated, not real-world results.")


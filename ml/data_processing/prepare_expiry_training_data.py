from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

input_path = ROOT / "data" / "processed" / "inventory_batches.csv"
output_path = ROOT / "data" / "processed" / "expiry_training_data.csv"

df = pd.read_csv(input_path)

# Exclude batches already past expiry from this training target.
eligible = df.loc[df["days_to_expiry"] >= 0].copy()

# Proposed operational risk target:
# 1 = remaining stock exceeds expected demand before expiry.
# 0 = expected demand is sufficient for remaining stock.
eligible["proposed_expiry_risk"] = (
    eligible["quantity_remaining"]
    > eligible["expected_demand_before_expiry"]
).astype(int)

eligible.to_csv(output_path, index=False)

print("Expiry-risk training data prepared.")
print("Output:", output_path)
print("Rows:", len(eligible))
print("\nProposed target distribution:")
print(eligible["proposed_expiry_risk"].value_counts().sort_index())
print("\nOriginal labels preserved:", "expiry_risk_label" in eligible.columns)
print("Expired batches excluded:", (df["days_to_expiry"] < 0).sum())

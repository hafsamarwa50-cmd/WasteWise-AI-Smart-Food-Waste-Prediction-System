from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"

# Create report folder if it does not exist
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load processed datasets
    inventory = pd.read_csv(DATA_DIR / "perishable_inventory.csv")
    batches = pd.read_csv(DATA_DIR / "inventory_batches.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    discounts = pd.read_csv(
        DATA_DIR / "discount_scenarios_simulated.csv"
    )

    # 1. Dataset overview
    print("\n===== DATASET OVERVIEW =====")
    for name, df in [
        ("Perishable Inventory", inventory),
        ("Inventory Batches", batches),
        ("Products", products),
        ("Discount Scenarios", discounts),
    ]:
        print(f"\n{name}")
        print("Shape:", df.shape)
        print("Missing values:", df.isnull().sum().sum())
        print("Duplicate rows:", df.duplicated().sum())

    # 2. Sales and food waste
    print("\n===== SALES AND FOOD WASTE =====")
    print("Total units sold:", inventory["units_sold"].sum())
    print("Total food waste:", inventory["waste_units"].sum())
    print("Stockout records:", inventory["stockout_flag"].sum())

    # 3. Summary by product category
    print("\n===== CATEGORY SUMMARY =====")
    category_summary = inventory.groupby("category").agg(
        total_units_sold=("units_sold", "sum"),
        total_waste=("waste_units", "sum"),
        average_discount_pct=("discount_pct", "mean"),
    ).round(2)

    print(category_summary)
    category_summary.to_csv(
        REPORT_DIR / "category_summary.csv"
    )

    # 4. Expiry risk labels
    print("\n===== EXPIRY RISK DISTRIBUTION =====")
    risk_counts = batches["expiry_risk_label"].value_counts().sort_index()
    print(risk_counts)

    # 5. Simulated discount scenario summary
    print("\n===== SIMULATED DISCOUNT ANALYSIS =====")
    discount_summary = discounts.groupby("discount_pct").agg(
        average_units_sold=(
            "estimated_units_sold_simulated", "mean"
        ),
        average_revenue=(
            "estimated_revenue_simulated", "mean"
        ),
        average_gross_margin=(
            "estimated_gross_margin_simulated", "mean"
        ),
    ).round(2)

    print(discount_summary)
    discount_summary.to_csv(
        REPORT_DIR / "discount_summary_simulated.csv"
    )

    # 6. Chart: units sold vs food waste by category
    category_summary[
        ["total_units_sold", "total_waste"]
    ].plot(kind="bar", figsize=(9, 5))

    plt.title("Units Sold vs Food Waste by Category")
    plt.xlabel("Product Category")
    plt.ylabel("Units")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "sales_vs_waste.png",
        dpi=150
    )
    plt.close()

    # 7. Chart: expiry risk distribution
    risk_counts.plot(kind="bar", figsize=(7, 4))

    plt.title("Expiry Risk Label Distribution")
    plt.xlabel("Risk Label (0 = Low, 1 = High)")
    plt.ylabel("Number of Batches")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "expiry_risk_distribution.png",
        dpi=150
    )
    plt.close()

    print("\nEDA completed successfully!")
    print("Reports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()

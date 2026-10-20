from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "processed"


def _load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Dashboard dataset is unavailable: {filename}",
        )
    return pd.read_csv(path)


@router.get("/summary")
def get_dashboard_summary():
    inventory = _load_dataset("inventory_batches.csv")
    discounts = _load_dataset("discount_scenarios_simulated.csv")
    perishable = _load_dataset("perishable_inventory.csv")

    eligible_scenarios = discounts.loc[discounts["days_to_expiry"] >= 0]
    scenario_sales = eligible_scenarios.pivot_table(
        index="batch_id",
        columns="discount_pct",
        values="estimated_units_sold_simulated",
        aggfunc="sum",
    )
    simulated_sales_lift_at_20_pct = None
    if 0 in scenario_sales.columns and 20 in scenario_sales.columns:
        simulated_sales_lift_at_20_pct = int(
            round((scenario_sales[20] - scenario_sales[0]).sum())
        )

    high_risk_mask = inventory["expiry_risk_label"].astype(str).str.lower().isin(
        {"1", "high", "true"}
    )
    risk_counts = inventory["expiry_risk_label"].astype(str).str.lower().value_counts()
    risk_distribution = [
        {"label": "High risk" if label in {"1", "high", "true"} else "Low risk", "batches": int(count)}
        for label, count in risk_counts.items()
    ]

    category_data = (
        inventory.groupby("category", as_index=False)[
            [
                "quantity_remaining",
                "expected_demand_before_expiry",
                "waste_units_actual_simulated",
            ]
        ]
        .sum()
        .rename(
            columns={
                "quantity_remaining": "units",
                "expected_demand_before_expiry": "expected_demand",
                "waste_units_actual_simulated": "waste_units",
            }
        )
    )
    discount_data = (
        discounts.groupby("discount_pct", as_index=False)[
            ["estimated_units_sold_simulated", "estimated_gross_margin_simulated"]
        ]
        .mean()
        .rename(
            columns={
                "discount_pct": "discount",
                "estimated_units_sold_simulated": "estimated_units_sold",
                "estimated_gross_margin_simulated": "estimated_gross_margin",
            }
        )
        .sort_values("discount")
    )

    recent = inventory.sort_values("snapshot_date", ascending=False).head(5)
    recent_activity = [
        {
            "batch_id": str(row.batch_id),
            "category": str(row.category),
            "quantity_remaining": int(row.quantity_remaining),
            "days_to_expiry": int(row.days_to_expiry),
            "snapshot_date": str(row.snapshot_date),
        }
        for row in recent.itertuples(index=False)
    ]

    return {
        "data_source": "processed project datasets",
        "metrics": {
            "total_batches": int(len(inventory)),
            "high_risk_batches": int(high_risk_mask.sum()),
            "approaching_expiry_batches": int(
                inventory["days_to_expiry"].between(0, 7).sum()
            ),
            "simulated_waste_units": int(inventory["waste_units_actual_simulated"].sum()),
            "perishable_units_sold": int(perishable["units_sold"].sum()),
            "perishable_recorded_waste_units": int(perishable["waste_units"].sum()),
            "simulated_sales_lift_at_20_pct": simulated_sales_lift_at_20_pct,
            "discount_scenario_batch_count": int(len(scenario_sales)),
            "expected_demand_before_expiry": int(
                inventory["expected_demand_before_expiry"].sum()
            ),
        },
        "inventory_by_category": category_data.to_dict(orient="records"),
        "expiry_risk_distribution": risk_distribution,
        "discount_scenarios": discount_data.round(2).to_dict(orient="records"),
        "recent_activity": recent_activity,
    }
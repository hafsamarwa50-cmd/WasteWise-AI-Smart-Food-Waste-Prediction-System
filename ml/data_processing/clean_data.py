from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATE_COLUMNS = {
    "inventory_batches.csv": [
        "received_at",
        "snapshot_date",
        "expiry_date",
    ],
    "perishable_inventory.csv": ["date"],
}

def clean_csv(file_path: Path) -> None:
    print(f"\nProcessing: {file_path.name}")

    df = pd.read_csv(file_path)

    # Remove exact duplicate rows only
    before = len(df)
    df = df.drop_duplicates()
    print("Duplicate rows removed:", before - len(df))

    # Convert known date columns
    for column in DATE_COLUMNS.get(file_path.name, []):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # Report missing values; don't silently delete records
    print("Missing values after cleaning:")
    print(df.isnull().sum().to_string())

    # Save a separate processed copy
    output_path = PROCESSED_DIR / file_path.name

    # Keep dates in a consistent YYYY-MM-DD format
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print("Saved to:", output_path.relative_to(PROJECT_ROOT))
    print("Final shape:", df.shape)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(RAW_DIR.glob("*.csv")):
        clean_csv(file_path)

    print("\nData cleaning completed!")


if __name__ == "__main__":
    main()

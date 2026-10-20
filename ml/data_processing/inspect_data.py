from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def inspect_csv(file_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def main() -> None:
    for file_path in sorted(DATA_DIR.glob("*.csv")):
        inspect_csv(file_path)

    print("\nInspection completed!")


if __name__ == "__main__":
    main()

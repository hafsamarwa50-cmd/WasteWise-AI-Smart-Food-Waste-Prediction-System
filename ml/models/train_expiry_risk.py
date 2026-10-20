from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "expiry_training_data.csv"
MODEL_DIR = ROOT / "ml" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "proposed_expiry_risk"

# Exclude the target's direct inputs, IDs, dates, old labels,
# and simulated outcomes to reduce leakage.
excluded = [
    target,
    "batch_id",
    "product_id",
    "received_at",
    "snapshot_date",
    "expiry_date",
    "quantity_remaining",
    "expected_demand_before_expiry",
    "expiry_risk_label",
    "waste_units_actual_simulated",
]

X = df.drop(columns=excluded)
y = df[target]

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("=== EXPIRY-RISK MODEL EVALUATION ===")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X.columns.tolist())

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Lower risk", "High risk"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "\nPR-AUC (average precision):",
    round(average_precision_score(y_test, probabilities), 4),
)

model_path = MODEL_DIR / "expiry_risk_model.joblib"
joblib.dump(model, model_path)

print("\nSaved model to:", model_path)
print(
    "\nNote: These metrics measure agreement with the proposed "
    "rule-derived labels, not proven real-world waste prediction."
)

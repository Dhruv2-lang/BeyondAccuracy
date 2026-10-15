
"""Load the trained fraud model and make consistent predictions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"


def load_artifacts():
    """Load the model and its saved metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python src/train.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Metadata not found. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    return model, metadata


def predict_fraud(model, transaction_df, metadata):
    """Predict fraud probabilities and labels."""
    if not isinstance(transaction_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if transaction_df.empty:
        raise ValueError("Provide at least one transaction.")

    feature_names = metadata["feature_names"]
    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    missing = sorted(set(feature_names) - set(transaction_df.columns))
    extra = sorted(set(transaction_df.columns) - set(feature_names))

    if missing or extra:
        raise ValueError(
            f"Input columns mismatch. Missing: {missing}; "
            f"unexpected: {extra}"
        )

    features = transaction_df.loc[:, feature_names].copy()

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    ):
        raise ValueError("All features must be numeric.")

    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features contain missing or non-finite values.")

    probabilities = model.predict_proba(features)[:, 1]

    result = transaction_df.copy()
    result["fraud_probability"] = probabilities
    result["is_fraud"] = (
        probabilities >= threshold
    ).astype(int)

    return result


if __name__ == "__main__":
    model, metadata = load_artifacts()
    print("Model loaded successfully.")
    print("Saved threshold:", metadata["threshold"])
    print("Expected features:", len(metadata["feature_names"]))


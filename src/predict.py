"""Loading trained artifacts and scoring transactions.

This module is the single place where inference logic lives; the Streamlit
dashboard, the CLI below and the tests all use it.

Security note: ``model.pkl`` is a joblib/pickle file.  Unpickling can execute
arbitrary code, so only load model files you trained yourself or otherwise
trust.

CLI::

    python -m src.predict --input transactions.csv --output scored.csv
    python -m src.predict --input transactions.csv --threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from . import config

PROBABILITY_COLUMN = "fraud_probability"
FLAG_COLUMN = "flagged_as_fraud"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class FraudModelError(Exception):
    """Base class for all errors raised by this module."""


class ArtifactNotFoundError(FraudModelError, FileNotFoundError):
    """The model or metadata file is missing."""


class InvalidMetadataError(FraudModelError, ValueError):
    """The metadata file is unreadable or internally inconsistent."""


class InputValidationError(FraudModelError, ValueError):
    """The transactions passed for scoring are not valid."""


class InvalidThresholdError(InputValidationError):
    """The decision threshold is not a finite number in [0, 1]."""


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Artifacts:
    """A trained model together with the metadata needed to use it safely."""

    model: Any
    metadata: dict

    @property
    def feature_names(self) -> List[str]:
        return list(self.metadata["feature_names"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def validate_threshold(threshold: Any) -> float:
    """Return ``threshold`` as a float, or raise :class:`InvalidThresholdError`."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise InvalidThresholdError(f"Threshold must be a number, got {threshold!r}.")
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InvalidThresholdError(f"Threshold must be a finite number between 0 and 1, got {threshold!r}.")
    return value


def validate_metadata(metadata: Any) -> dict:
    """Check the structure of a metadata dict and return it."""
    if not isinstance(metadata, dict):
        raise InvalidMetadataError("Metadata must be a JSON object.")
    for key in ("threshold", "feature_names"):
        if key not in metadata:
            raise InvalidMetadataError(f"Metadata is missing required key '{key}'.")
    names = metadata["feature_names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise InvalidMetadataError("'feature_names' must be a non-empty list of unique strings.")
    try:
        validate_threshold(metadata["threshold"])
    except InvalidThresholdError as exc:
        raise InvalidMetadataError(f"Saved threshold is invalid: {exc}") from exc
    return metadata


def load_metadata(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else config.MODEL_DIR / config.METADATA_FILENAME
    if not path.exists():
        raise ArtifactNotFoundError(
            f"Model metadata not found: {path}. Train a model first with `python -m src.train`."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidMetadataError(f"Metadata file {path} is not valid JSON: {exc}") from exc
    return validate_metadata(metadata)


def load_artifacts(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Artifacts:
    """Load model + metadata and check that they agree with each other."""
    metadata = load_metadata(metadata_path)
    if model_path is None:
        model_path = config.MODEL_DIR / metadata.get("model_file", config.MODEL_FILENAME)
    model_path = Path(model_path)
    if not model_path.exists():
        raise ArtifactNotFoundError(
            f"Model file not found: {model_path}. Train a model first with `python -m src.train` "
            "(the model binary is not stored in Git)."
        )
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # corrupted file, incompatible library version, ...
        raise FraudModelError(
            f"Could not load model from {model_path} ({type(exc).__name__}: {exc}). "
            "If the file was created with a different library version, retrain with `python -m src.train`."
        ) from exc
    if not hasattr(model, "predict_proba"):
        raise FraudModelError(f"Loaded object of type {type(model).__name__} has no predict_proba().")
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != len(metadata["feature_names"]):
        raise InvalidMetadataError(
            f"Model expects {expected} features but metadata lists {len(metadata['feature_names'])}. "
            "The model and metadata files do not belong together."
        )
    return Artifacts(model=model, metadata=metadata)


# --------------------------------------------------------------------------- #
# Input validation and scoring
# --------------------------------------------------------------------------- #
def validate_transactions(
    transactions: pd.DataFrame,
    feature_names: List[str],
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Validate ``transactions`` and return the features in training order.

    Raises :class:`InputValidationError` for: non-DataFrame input, no rows,
    duplicate column names, missing features, unexpected extra columns
    (unless ``allow_extra``), non-numeric columns, and missing / infinite values.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise InputValidationError(f"Expected a pandas DataFrame, got {type(transactions).__name__}.")
    if len(transactions) == 0:
        raise InputValidationError("Input contains no transactions (0 rows).")
    if transactions.columns.duplicated().any():
        dupes = sorted(set(transactions.columns[transactions.columns.duplicated()].astype(str)))
        raise InputValidationError(f"Duplicate column names: {dupes}.")

    columns = list(transactions.columns)
    missing = [c for c in feature_names if c not in columns]
    if missing:
        raise InputValidationError(f"Missing required feature column(s): {missing}.")
    extra = [c for c in columns if c not in set(feature_names)]
    if extra and not allow_extra:
        raise InputValidationError(
            f"Unexpected column(s): {extra}. Remove them (the model uses only {len(feature_names)} features)."
        )

    features = transactions[feature_names]
    bad_type = [
        c for c in feature_names
        if pd.api.types.is_bool_dtype(features[c]) or not pd.api.types.is_numeric_dtype(features[c])
    ]
    if bad_type:
        raise InputValidationError(f"Non-numeric feature column(s): {bad_type}.")

    values = features.to_numpy(dtype=float)
    nan_mask = np.isnan(values)
    if nan_mask.any():
        cols = [c for c, bad in zip(feature_names, nan_mask.any(axis=0)) if bad]
        raise InputValidationError(
            f"Missing values (NaN) found in column(s) {cols} ({int(nan_mask.any(axis=1).sum())} row(s) affected)."
        )
    inf_mask = np.isinf(values)
    if inf_mask.any():
        cols = [c for c, bad in zip(feature_names, inf_mask.any(axis=0)) if bad]
        raise InputValidationError(f"Infinite values found in column(s) {cols}.")
    return features.astype(float)


def predict_fraud(
    artifacts: Artifacts,
    transactions: pd.DataFrame,
    threshold: Optional[float] = None,
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Score transactions.

    Returns a DataFrame with the features (in training order), the original
    index, ``fraud_probability`` in [0, 1] and ``flagged_as_fraud`` (1 when the
    probability is >= the threshold).  ``threshold`` defaults to the value
    saved with the model.  A flag is a model alert, not proof of fraud.
    """
    used = artifacts.threshold if threshold is None else validate_threshold(threshold)
    features = validate_transactions(transactions, artifacts.feature_names, allow_extra=allow_extra)
    probs = np.asarray(artifacts.model.predict_proba(features)[:, 1], dtype=float)
    result = features.copy()
    result[PROBABILITY_COLUMN] = probs
    result[FLAG_COLUMN] = (probs >= used).astype(int)
    return result


def apply_threshold(probabilities, threshold: float) -> np.ndarray:
    """Re-classify already-computed probabilities (no model call needed)."""
    return (np.asarray(probabilities, dtype=float) >= validate_threshold(threshold)).astype(int)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of transactions with the trained model.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with the model's feature columns.")
    parser.add_argument("--output", type=Path, help="Where to write the scored CSV (default: print summary only).")
    parser.add_argument("--threshold", type=float, help="Override the saved decision threshold.")
    args = parser.parse_args(argv)

    try:
        artifacts = load_artifacts()
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        data = pd.read_csv(args.input)
        if config.TARGET_COLUMN in data.columns:
            print(f"Note: ignoring '{config.TARGET_COLUMN}' column.", file=sys.stderr)
            data = data.drop(columns=[config.TARGET_COLUMN])
        result = predict_fraud(artifacts, data, threshold=args.threshold)
    except (FraudModelError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    used = artifacts.threshold if args.threshold is None else args.threshold
    print(f"Scored {len(result)} transaction(s); {int(result[FLAG_COLUMN].sum())} flagged at threshold {used:.4f}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

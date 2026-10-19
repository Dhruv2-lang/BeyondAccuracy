"""Dataset loading, validation and the shared train/validation/test split.

Every experiment, notebook and script uses :func:`split_data` so results are
comparable.  No statistic is ever computed on validation/test rows here;
scaling and resampling are fitted on the training split only (see
``src.models``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

Split = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]

DATASET_HELP = (
    "The dataset is not distributed with this repository. Download the "
    "'Credit Card Fraud Detection' dataset (ULB / Worldline, hosted on Kaggle) "
    "yourself and place the file at data/creditcard.csv, or point "
    "BEYONDACCURACY_DATA_PATH at it."
)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate the raw credit-card CSV."""
    path = Path(path) if path is not None else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\n{DATASET_HELP}")
    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` is not a usable labelled dataset."""
    target = config.TARGET_COLUMN
    if target not in df.columns:
        raise ValueError(f"Dataset must contain a '{target}' column.")
    if set(pd.unique(df[target])) - {0, 1}:
        raise ValueError(f"'{target}' must be binary (0 = legitimate, 1 = fraud).")
    if df[target].nunique() != 2:
        raise ValueError("Expected both classes (0 and 1) to be present.")
    features = df.drop(columns=[target])
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features contain missing or non-finite values.")


def split_data(
    df: pd.DataFrame,
    random_state: int = config.RANDOM_STATE,
    val_fraction: float = config.VAL_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
) -> Split:
    """Stratified train/validation/test split (default 70/15/15).

    Returns ``X_train, X_val, X_test, y_train, y_val, y_test``.  The feature
    column order of ``df`` (minus the target) is preserved.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to < 1.")
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN].astype(int)

    holdout = val_fraction + test_fraction
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold,
        test_size=test_fraction / holdout,
        stratify=y_hold,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

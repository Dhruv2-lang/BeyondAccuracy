"""Project-wide paths and constants.

All paths are resolved relative to the repository root, so scripts, the
dashboard and notebooks behave the same regardless of the current working
directory.  Each location can be overridden with an environment variable
(used by the tests and for deployments that keep artifacts elsewhere).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_PATH = _path_from_env("BEYONDACCURACY_DATA_PATH", ROOT / "data" / "creditcard.csv")
SAMPLE_PATH = _path_from_env(
    "BEYONDACCURACY_SAMPLE_PATH", ROOT / "data" / "sample_transactions.csv"
)
MODEL_DIR = _path_from_env("BEYONDACCURACY_MODEL_DIR", ROOT / "models")
RESULTS_DIR = _path_from_env("BEYONDACCURACY_RESULTS_DIR", ROOT / "results")
ASSETS_DIR = _path_from_env("BEYONDACCURACY_ASSETS_DIR", ROOT / "assets")

MODEL_FILENAME = "model.pkl"
METADATA_FILENAME = "metadata.json"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42

# 70% train / 15% validation / 15% test
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Threshold-selection rule: highest validation recall subject to this precision.
MIN_VALIDATION_PRECISION = 0.80

# ILLUSTRATIVE cost assumptions (currency units per event).  They are NOT
# derived from data; replace them with your own operating costs.
COST_FALSE_POSITIVE = 5.0    # analyst review / customer friction per false alarm
COST_FALSE_NEGATIVE = 100.0  # average loss per missed fraud

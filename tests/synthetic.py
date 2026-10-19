"""Synthetic stand-in for the credit-card dataset.

It reproduces only the *schema* (Time, V1-V28, Amount, Class) and a rare,
partly separable positive class so that code paths can be tested quickly
without the real data.  Numbers computed on it say nothing about real
fraud-detection performance.
"""

import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_synthetic_creditcard(n_rows: int = 6000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_rows * fraud_rate)), 10)
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    data = {"Time": np.sort(rng.uniform(0, 172_800, n_rows))}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n_rows)
    for col, shift in (("V14", -2.5), ("V10", -1.8), ("V12", -2.0), ("V17", -1.5), ("V4", 1.5)):
        data[col] = data[col] + y * shift
    data["Amount"] = rng.lognormal(3.0, 1.0, n_rows)
    df = pd.DataFrame(data)[FEATURES]
    df["Class"] = y
    return df

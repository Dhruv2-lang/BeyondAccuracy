"""Metrics, threshold selection and cost helpers for imbalanced fraud data."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def select_threshold(
    y_true,
    probs,
    min_precision: float = config.MIN_VALIDATION_PRECISION,
) -> Dict[str, object]:
    """Choose a decision threshold from *validation* labels/probabilities.

    Rule: the threshold with the highest recall among those whose precision is
    at least ``min_precision``.  If no threshold reaches that precision, fall
    back to the threshold with the best F1.  Returns the threshold and a
    human-readable description of the rule that produced it.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    if y_true.sum() == 0:
        raise ValueError("Cannot select a threshold: no positive samples in validation data.")

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall have one more element than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    eligible = np.flatnonzero(precision >= min_precision)
    if len(eligible):
        # argmax returns the first maximum: among equal-recall candidates this is the
        # lowest threshold (same behaviour as the original train.py).  A higher tied
        # threshold would sit right below the next validation fraud's score and
        # generalise worse.
        best_idx = eligible[np.argmax(recall[eligible])]
        rule = f"Highest validation recall with precision >= {min_precision:.2f}"
    else:
        f1_values = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1_values))
        rule = "Maximum validation F1; minimum precision target was unattainable"
    return {"threshold": float(thresholds[best_idx]), "rule": rule}


def compute_metrics(y_true, probs, threshold: float) -> Dict[str, object]:
    """All headline metrics at a given threshold, plus class counts."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rows": int(n),
        "fraud_cases": int(y_true.sum()),
        "legitimate_cases": int(n - y_true.sum()),
        "majority_class_accuracy": float(1 - y_true.mean()),
    }


def expected_cost(fp: int, fn: int, cost_fp: float, cost_fn: float) -> float:
    """Total cost of false alarms and missed fraud (illustrative unit costs)."""
    return float(fp * cost_fp + fn * cost_fn)


def threshold_table(y_true, probs, n_points: int = 1001, extra=()) -> "pd.DataFrame":
    """Precision / recall / error counts over a grid of thresholds.

    Computed in one pass with sorted scores, so it is cheap even for large
    validation sets.  ``extra`` thresholds (e.g. the chosen one) are included
    exactly.
    """
    import pandas as pd

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, n_points), np.asarray(list(extra), dtype=float)]))

    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_y = y_true[order]
    cum_pos = np.concatenate([[0], np.cumsum(sorted_y)])        # positives with score < t
    total_pos = int(y_true.sum())
    n = len(y_true)

    idx = np.searchsorted(sorted_probs, grid, side="left")      # first index with score >= t
    pos_below = cum_pos[idx]
    flagged = n - idx
    tp = total_pos - pos_below
    fp = flagged - tp
    fn = pos_below
    tn = (n - total_pos) - fp
    precision = np.divide(tp, flagged, out=np.ones_like(tp, dtype=float), where=flagged > 0)
    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    return pd.DataFrame(
        {"threshold": grid, "precision": precision, "recall": recall,
         "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": flagged}
    )

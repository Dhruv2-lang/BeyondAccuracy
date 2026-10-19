"""Figures generated from the *actual* experiment (never hand-labelled)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def precision_recall_plot(y_val, p_val, y_test, p_test, threshold, val_point, test_point, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for y, p, label, color in ((y_val, p_val, "Validation", "steelblue"), (y_test, p_test, "Test", "crimson")):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color, label=f"{label} PR curve")
    ax.scatter([val_point["recall"]], [val_point["precision"]], color="steelblue", zorder=5, marker="o",
               label=f"Validation @ threshold {threshold:.3f}")
    ax.scatter([test_point["recall"]], [test_point["precision"]], color="crimson", zorder=5, marker="s",
               label=f"Test @ threshold {threshold:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="Precision-Recall curve (selected model)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    return _save(fig, path)


def threshold_tradeoff_plot(table, threshold, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision", color="steelblue")
    ax.plot(table["threshold"], table["recall"], label="Recall", color="crimson")
    ax.axvline(threshold, linestyle="--", color="gray", label=f"Selected threshold ({threshold:.3f}, chosen on validation)")
    ax.set(xlabel="Decision threshold", ylabel="Score", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision & recall vs decision threshold (validation set)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center")
    return _save(fig, path)


def confusion_matrix_plot(cm: Dict[str, int], threshold: float, path: Path) -> Path:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not flagged", "Flagged"],
           yticklabels=["Legitimate", "Fraud"], xlabel="Model decision", ylabel="Actual",
           title=f"Test-set confusion matrix (threshold {threshold:.3f})")
    return _save(fig, path)


def model_comparison_plot(results: Dict[str, dict], selected: str, path: Path) -> Path:
    names = list(results)
    metrics = ["precision", "recall", "f1", "average_precision"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, m in enumerate(metrics):
        ax.bar(x + (k - 1.5) * width, [results[n]["test"][m] for n in names], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([n + (" *" if n == selected else "") for n in names], rotation=15)
    ax.set(ylabel="Score (test set)", ylim=(0, 1.05),
           title="Experiments compared on the test set (threshold tuned on validation; * = selected)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)

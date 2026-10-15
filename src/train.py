
"""Train and evaluate the BeyondAccuracy fraud detector."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
MIN_VALIDATION_PRECISION = 0.80


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if df["Class"].nunique() != 2:
        raise ValueError("Expected a binary fraud target.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain missing or non-finite values.")

    # 70% training, 15% validation, 15% final test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Weight positive fraud examples; do not resample validation/test data.
    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost...")
    print(f"Train: {len(X_train):,} rows")
    print(f"Validation: {len(X_val):,} rows")
    print(f"Test: {len(X_test):,} rows")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model.fit(X_train, y_train)

    # Select a threshold using validation data only.
    val_probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probs
    )

    # precision/recall have one extra endpoint relative to thresholds.
    eligible = np.flatnonzero(
        precision[:-1] >= MIN_VALIDATION_PRECISION
    )

    if len(eligible):
        best_idx = eligible[np.argmax(recall[:-1][eligible])]
        threshold = float(thresholds[best_idx])
        threshold_rule = (
            "Highest validation recall with precision >= "
            f"{MIN_VALIDATION_PRECISION:.2f}"
        )
    else:
        # Explicit fallback if the target precision is unattainable.
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1_values))
        threshold = float(thresholds[best_idx])
        threshold_rule = (
            "Maximum validation F1; minimum precision target "
            "was unattainable"
        )

    # Evaluate once on the untouched test set.
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "average_precision": float(
            average_precision_score(y_test, test_probs)
        ),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "precision": float(
            precision_score(y_test, test_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "test_rows": int(len(X_test)),
        "test_fraud_cases": int(y_test.sum()),
        "validation_precision_target": MIN_VALIDATION_PRECISION,
        "threshold_selection_rule": threshold_rule,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": float(scale_pos_weight),
    }

    # Save model and the threshold used by the application.
    joblib.dump(model, MODEL_DIR / "model.pkl")

    metadata = {
        "threshold": threshold,
        "feature_names": list(X.columns),
        "target_column": "Class",
        "model_type": "XGBClassifier",
        "threshold_selection_rule": threshold_rule,
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (RESULTS_DIR / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSelected threshold:", round(threshold, 6))
    print("Threshold rule:", threshold_rule)
    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model:", MODEL_DIR / "model.pkl")
    print("Saved metadata:", MODEL_DIR / "metadata.json")
    print("Saved test results:", RESULTS_DIR / "final_test_metrics.json")


if __name__ == "__main__":
    main()

"""Train, compare and save the BeyondAccuracy fraud detector.

Protocol (identical for every experiment)
-----------------------------------------
1. Stratified 70/15/15 train/validation/test split with a fixed seed.
2. Each experiment is fitted on the training split only (scaling / resampling
   are fitted there too).
3. The decision threshold of each experiment is chosen on the validation split.
4. The final model is the experiment with the best *validation* average
   precision (override with ``--final-model``).
5. The test split is used once, after all choices are made, to report metrics.

Usage::

    python -m src.train                       # all experiments
    python -m src.train --experiments xgb_weighted logreg_baseline
    python -m src.train --final-model xgb_weighted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from . import config, evaluate, models, plots, preprocess


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_existing(model_dir: Path) -> Optional[Path]:
    """Keep the previous model/metadata so a retrain never destroys them."""
    existing = [model_dir / config.MODEL_FILENAME, model_dir / config.METADATA_FILENAME]
    existing = [p for p in existing if p.exists()]
    if not existing:
        return None
    backup = model_dir / "previous"
    backup.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup / path.name)
    return backup


def run_training(
    data_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
    experiments: Optional[List[str]] = None,
    final_model: Optional[str] = None,
    min_precision: float = config.MIN_VALIDATION_PRECISION,
    cost_fp: float = config.COST_FALSE_POSITIVE,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    xgb_params: Optional[dict] = None,
    random_state: int = config.RANDOM_STATE,
    make_plots: bool = True,
    verbose: bool = True,
) -> dict:
    data_path = Path(data_path) if data_path else config.DATA_PATH
    model_dir = Path(model_dir) if model_dir else config.MODEL_DIR
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    assets_dir = Path(assets_dir) if assets_dir else config.ASSETS_DIR
    experiments = experiments or list(models.EXPERIMENTS)
    unknown = [e for e in experiments if e not in models.EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment(s) {unknown}. Choose from {sorted(models.EXPERIMENTS)}")
    if final_model is not None and final_model not in experiments:
        raise ValueError(f"--final-model '{final_model}' is not among the experiments being run: {experiments}")

    log = print if verbose else (lambda *a, **k: None)

    df = preprocess.load_dataset(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.split_data(df, random_state=random_state)
    feature_names = list(X_train.columns)
    log(f"Rows  train={len(X_train):,} (fraud {int(y_train.sum())})  "
        f"val={len(X_val):,} (fraud {int(y_val.sum())})  test={len(X_test):,} (fraud {int(y_test.sum())})")

    results: Dict[str, dict] = {}
    fitted = {}
    for name in experiments:
        log(f"\n[{name}] {models.DESCRIPTIONS[name]}")
        model = models.fit_experiment(name, X_train, y_train, params=xgb_params, random_state=random_state)
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        sel = evaluate.select_threshold(y_val, p_val, min_precision=min_precision)
        val_m = evaluate.compute_metrics(y_val, p_val, sel["threshold"])
        test_m = evaluate.compute_metrics(y_test, p_test, sel["threshold"])
        for split_m in (val_m, test_m):
            cm = split_m["confusion_matrix"]
            split_m["expected_cost"] = evaluate.expected_cost(cm["fp"], cm["fn"], cost_fp, cost_fn)
        results[name] = {
            "description": models.DESCRIPTIONS[name],
            "threshold": sel["threshold"],
            "threshold_selection_rule": sel["rule"],
            "validation": val_m,
            "test": test_m,
        }
        fitted[name] = (model, p_val, p_test)
        log(f"  threshold={sel['threshold']:.4f}  val AP={val_m['average_precision']:.4f}  "
            f"test AP={test_m['average_precision']:.4f}  test P={test_m['precision']:.3f} "
            f"R={test_m['recall']:.3f} F1={test_m['f1']:.3f}")

    if final_model is None:
        selected = max(results, key=lambda n: results[n]["validation"]["average_precision"])
        criterion = "highest validation average precision"
    else:
        selected = final_model
        criterion = "chosen explicitly with --final-model"
    log(f"\nSelected model: {selected} ({criterion})")

    model, p_val, p_test = fitted[selected]
    chosen = results[selected]
    threshold = chosen["threshold"]
    test_m = chosen["test"]

    # ---- persist artifacts -------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup_existing(model_dir)
    if backup:
        log(f"Existing model/metadata copied to {backup}")

    joblib.dump(model, model_dir / config.MODEL_FILENAME)
    metadata = {
        "schema_version": 2,
        "model_file": config.MODEL_FILENAME,
        "model_type": type(model).__name__,
        "selected_experiment": selected,
        "selection_criterion": criterion,
        "threshold": threshold,
        "threshold_selection_rule": chosen["threshold_selection_rule"],
        "feature_names": feature_names,
        "target_column": config.TARGET_COLUMN,
        "random_state": random_state,
        "split": {"train": 1 - config.VAL_FRACTION - config.TEST_FRACTION,
                  "validation": config.VAL_FRACTION, "test": config.TEST_FRACTION,
                  "stratified": True},
        "dataset_sha256": _sha256(data_path),
        "dataset_rows": int(len(df)),
        "library_versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (model_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cm = test_m["confusion_matrix"]
    final_metrics = {  # keys kept compatible with earlier versions of this file
        "selected_experiment": selected,
        "average_precision": test_m["average_precision"],
        "roc_auc": test_m["roc_auc"],
        "precision": test_m["precision"],
        "recall": test_m["recall"],
        "f1": test_m["f1"],
        "accuracy": test_m["accuracy"],
        "threshold": threshold,
        "confusion_matrix": cm,
        "test_rows": test_m["rows"],
        "test_fraud_cases": test_m["fraud_cases"],
        "validation_precision_target": min_precision,
        "threshold_selection_rule": chosen["threshold_selection_rule"],
        "random_state": random_state,
        "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
    }
    (results_dir / "final_test_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    report = {
        "protocol": {
            "split": "stratified 70/15/15 (train/validation/test)",
            "random_state": random_state,
            "model_selection": criterion,
            "threshold_selection": "validation set only",
            "test_set_usage": "final reporting only",
            "min_validation_precision": min_precision,
            "illustrative_costs": {"false_positive": cost_fp, "false_negative": cost_fn,
                                   "note": "Placeholder assumptions, not derived from data."},
            "xgboost_params": {**models.DEFAULT_XGB_PARAMS, **(xgb_params or {})},
            "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "fraud_counts": {"train": int(y_train.sum()), "validation": int(y_val.sum()), "test": int(y_test.sum())},
        },
        "selected_experiment": selected,
        "experiments": results,
    }
    (results_dir / "experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "experiment_summary.md").write_text(_summary_markdown(report), encoding="utf-8")

    table = evaluate.threshold_table(y_val, p_val, extra=[threshold])
    table.to_csv(results_dir / "threshold_curve_validation.csv", index=False, float_format="%.6f")

    if make_plots:
        plots.precision_recall_plot(y_val, p_val, y_test, p_test, threshold,
                                    chosen["validation"], test_m, assets_dir / "precision_recall_curve.png")
        plots.threshold_tradeoff_plot(table, threshold, assets_dir / "threshold_tradeoff.png")
        plots.confusion_matrix_plot(cm, threshold, assets_dir / "confusion_matrix.png")
        plots.model_comparison_plot(results, selected, assets_dir / "model_comparison.png")

    log(f"\nSaved model:    {model_dir / config.MODEL_FILENAME}")
    log(f"Saved metadata: {model_dir / config.METADATA_FILENAME}")
    log(f"Saved results:  {results_dir}")
    return report


def _summary_markdown(report: dict) -> str:
    p = report["protocol"]
    lines = [
        "# Experiment summary",
        "",
        f"Split: {p['split']}, seed {p['random_state']}. Thresholds chosen on validation data "
        f"(rule: highest recall with precision >= {p['min_validation_precision']:.2f}); "
        "metrics below are on the untouched test split. "
        f"Final model: **{report['selected_experiment']}** ({p['model_selection']}).",
        "",
        "| Experiment | Threshold | Accuracy | Precision | Recall | F1 | Avg. precision | ROC-AUC | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in report["experiments"].items():
        t = r["test"]
        c = t["confusion_matrix"]
        star = " *" if name == report["selected_experiment"] else ""
        lines.append(
            f"| {name}{star} | {r['threshold']:.4f} | {t['accuracy']:.4f} | {t['precision']:.3f} | "
            f"{t['recall']:.3f} | {t['f1']:.3f} | {t['average_precision']:.3f} | {t['roc_auc']:.3f} | "
            f"{c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |"
        )
    t = next(iter(report["experiments"].values()))["test"]
    lines += [
        "",
        f"Test split: {t['rows']:,} rows, {t['fraud_cases']} fraud. Always predicting 'legitimate' "
        f"would score accuracy {t['majority_class_accuracy']:.4f} while catching no fraud.",
        "",
        "Illustrative cost on the test split "
        f"(false alarm = {p['illustrative_costs']['false_positive']}, missed fraud = "
        f"{p['illustrative_costs']['false_negative']}; placeholder assumptions): "
        + ", ".join(f"{n}: {r['test']['expected_cost']:.0f}" for n, r in report["experiments"].items()),
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help=f"Path to creditcard.csv (default: {config.DATA_PATH})")
    parser.add_argument("--experiments", nargs="+", choices=sorted(models.EXPERIMENTS),
                        help="Subset of experiments to run (default: all).")
    parser.add_argument("--final-model", choices=sorted(models.EXPERIMENTS),
                        help="Force the saved model instead of selecting by validation average precision.")
    parser.add_argument("--min-precision", type=float, default=config.MIN_VALIDATION_PRECISION)
    parser.add_argument("--cost-fp", type=float, default=config.COST_FALSE_POSITIVE)
    parser.add_argument("--cost-fn", type=float, default=config.COST_FALSE_NEGATIVE)
    parser.add_argument("--n-estimators", type=int, help="Override XGBoost n_estimators (e.g. for quick runs).")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    try:
        run_training(
            data_path=args.data, experiments=args.experiments, final_model=args.final_model,
            min_precision=args.min_precision, cost_fp=args.cost_fp, cost_fn=args.cost_fn,
            xgb_params={"n_estimators": args.n_estimators} if args.n_estimators else None,
            random_state=args.seed, make_plots=not args.no_plots,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

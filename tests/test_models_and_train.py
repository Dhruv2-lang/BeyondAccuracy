import json

import numpy as np
import pandas as pd
import pytest

from src import config, models, preprocess, train
from src.predict import load_artifacts, predict_fraud

QUICK = {"n_estimators": 25}


def test_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown experiment"):
        models.fit_experiment("nope", pd.DataFrame({"a": [1.0]}), pd.Series([0]))


def test_smote_and_undersampling_only_touch_training_data(synthetic_df):
    """Resampling is fitted on the training split; validation/test are untouched."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = preprocess.split_data(synthetic_df)
    val_before, test_before = X_va.copy(), X_te.copy()
    for name in ("xgb_smote", "xgb_undersample"):
        models.fit_experiment(name, X_tr, y_tr, params=QUICK)
    pd.testing.assert_frame_equal(X_va, val_before)
    pd.testing.assert_frame_equal(X_te, test_before)


def test_smote_rebalances_only_training_rows(synthetic_df, monkeypatch):
    from imblearn.over_sampling import SMOTE
    seen = {}
    original = SMOTE.fit_resample

    def spy(self, X, y):
        seen["n_in"] = len(X)
        return original(self, X, y)

    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    X_tr, _, _, y_tr, _, _ = preprocess.split_data(synthetic_df)
    models.fit_experiment("xgb_smote", X_tr, y_tr, params=QUICK)
    assert seen["n_in"] == len(X_tr)


def test_weighted_model_uses_training_class_ratio(synthetic_df):
    X_tr, _, _, y_tr, _, _ = preprocess.split_data(synthetic_df)
    model = models.fit_experiment("xgb_weighted", X_tr, y_tr, params=QUICK)
    assert model.get_params()["scale_pos_weight"] == pytest.approx((y_tr == 0).sum() / (y_tr == 1).sum())


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_csv):
    out = tmp_path_factory.mktemp("run")
    report = train.run_training(
        data_path=synthetic_csv, model_dir=out / "models", results_dir=out / "results",
        assets_dir=out / "assets", xgb_params=QUICK, verbose=False,
    )
    return out, report


def test_training_writes_consistent_artifacts(trained, synthetic_df):
    out, report = trained
    meta = json.loads((out / "models" / "metadata.json").read_text())
    final = json.loads((out / "results" / "final_test_metrics.json").read_text())
    assert meta["threshold"] == final["threshold"] == report["experiments"][report["selected_experiment"]]["threshold"]
    assert meta["feature_names"] == [c for c in synthetic_df.columns if c != "Class"]
    assert meta["selected_experiment"] == report["selected_experiment"]
    assert len(meta["dataset_sha256"]) == 64
    for fname in ("experiment_results.json", "experiment_summary.md", "threshold_curve_validation.csv"):
        assert (out / "results" / fname).exists()
    for fname in ("precision_recall_curve.png", "threshold_tradeoff.png", "confusion_matrix.png", "model_comparison.png"):
        assert (out / "assets" / fname).stat().st_size > 1000


def test_all_experiments_use_identical_split_and_report_both_splits(trained):
    _, report = trained
    assert set(report["experiments"]) == set(models.EXPERIMENTS)
    sizes = {(e["validation"]["rows"], e["test"]["rows"], e["test"]["fraud_cases"]) for e in report["experiments"].values()}
    assert len(sizes) == 1                                  # same evaluation protocol everywhere


def test_model_selected_by_validation_not_test(trained):
    _, report = trained
    best_val = max(report["experiments"], key=lambda n: report["experiments"][n]["validation"]["average_precision"])
    assert report["selected_experiment"] == best_val


def test_saved_model_reproduces_reported_test_metrics(trained, synthetic_df):
    """The documented workflow's metrics can be re-derived from the saved artifacts."""
    out, report = trained
    arts = load_artifacts(out / "models" / "model.pkl", out / "models" / "metadata.json")
    _, _, X_te, _, _, y_te = preprocess.split_data(synthetic_df)
    scored = predict_fraud(arts, X_te)
    cm = report["experiments"][report["selected_experiment"]]["test"]["confusion_matrix"]
    flagged = scored["flagged_as_fraud"].to_numpy()
    assert int(((flagged == 1) & (y_te.to_numpy() == 1)).sum()) == cm["tp"]
    assert int(((flagged == 1) & (y_te.to_numpy() == 0)).sum()) == cm["fp"]


def test_training_is_deterministic(tmp_path, synthetic_csv, trained):
    _, first = trained
    second = train.run_training(
        data_path=synthetic_csv, model_dir=tmp_path / "m", results_dir=tmp_path / "r",
        assets_dir=tmp_path / "a", xgb_params=QUICK, make_plots=False, verbose=False,
    )
    for name in first["experiments"]:
        for split in ("validation", "test"):
            assert first["experiments"][name][split]["average_precision"] == pytest.approx(
                second["experiments"][name][split]["average_precision"])
        assert first["experiments"][name]["threshold"] == pytest.approx(second["experiments"][name]["threshold"])


def test_retraining_backs_up_existing_model(tmp_path, synthetic_csv):
    kw = dict(data_path=synthetic_csv, model_dir=tmp_path / "m", results_dir=tmp_path / "r",
              assets_dir=tmp_path / "a", experiments=["xgb_weighted"], xgb_params=QUICK,
              make_plots=False, verbose=False)
    train.run_training(**kw)
    assert not (tmp_path / "m" / "previous").exists()
    train.run_training(**kw)
    assert (tmp_path / "m" / "previous" / "model.pkl").exists()
    assert (tmp_path / "m" / "previous" / "metadata.json").exists()


def test_final_model_override(tmp_path, synthetic_csv):
    report = train.run_training(
        data_path=synthetic_csv, model_dir=tmp_path / "m", results_dir=tmp_path / "r",
        assets_dir=tmp_path / "a", experiments=["logreg_baseline", "xgb_weighted"],
        final_model="logreg_baseline", xgb_params=QUICK, make_plots=False, verbose=False)
    assert report["selected_experiment"] == "logreg_baseline"
    arts = load_artifacts(tmp_path / "m" / "model.pkl", tmp_path / "m" / "metadata.json")
    assert arts.metadata["model_type"] == "Pipeline"


def test_train_cli_missing_dataset(tmp_path, capsys):
    assert train.main(["--data", str(tmp_path / "missing.csv")]) == 1
    assert "data/creditcard.csv" in capsys.readouterr().err


def test_training_does_not_write_into_repo_by_default_paths(trained):
    out, _ = trained
    assert config.ROOT not in out.parents


def test_xgb_weighted_reproduces_the_original_train_script(synthetic_df):
    """Split, hyper-parameters and threshold rule match the previously committed src/train.py.

    This is why re-running ``python -m src.train`` on the real data is expected to regenerate the
    metrics recorded in results/final_test_metrics.json (given the same library versions).
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y = synthetic_df.drop(columns=["Class"]), synthetic_df["Class"].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    ref = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=25, max_depth=4,
                        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                        random_state=42, n_jobs=-1, tree_method="hist").fit(X_tr, y_tr)
    prec, rec, thr = precision_recall_curve(y_va, ref.predict_proba(X_va)[:, 1])
    ok = np.flatnonzero(prec[:-1] >= 0.80)
    ref_threshold = float(thr[ok[np.argmax(rec[:-1][ok])]])

    ours = preprocess.split_data(synthetic_df)
    assert ours[2].index.equals(X_te.index) and ours[1].index.equals(X_va.index)
    model = models.fit_experiment("xgb_weighted", ours[0], ours[3], params=QUICK)
    np.testing.assert_allclose(model.predict_proba(X_te)[:, 1], ref.predict_proba(X_te)[:, 1])
    from src import evaluate
    assert evaluate.select_threshold(ours[4], model.predict_proba(ours[1])[:, 1])["threshold"] == ref_threshold

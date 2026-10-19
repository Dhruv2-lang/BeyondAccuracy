"""Dashboard tests using Streamlit's headless AppTest harness."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config, evaluate, preprocess
from src.predict import load_artifacts

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


@pytest.fixture()
def env(monkeypatch, tmp_path, artifact_dir, synthetic_df, synthetic_csv, artifacts):
    results = tmp_path / "results"
    results.mkdir()
    _, X_val, _, _, y_val, _ = preprocess.split_data(synthetic_df)
    probs = artifacts.model.predict_proba(X_val)[:, 1]
    evaluate.threshold_table(y_val, probs, extra=[artifacts.threshold]).to_csv(
        results / "threshold_curve_validation.csv", index=False)
    monkeypatch.setattr(config, "MODEL_DIR", artifact_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results)
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "DATA_PATH", synthetic_csv)
    monkeypatch.setattr(config, "SAMPLE_PATH", tmp_path / "no_sample.csv")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def run_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_app_starts_with_saved_threshold_and_warning(env, artifacts):
    at = run_app()
    assert not at.exception
    assert at.slider[0].value == pytest.approx(artifacts.threshold)
    assert any("not verdicts" in w.value for w in at.warning)


def test_sample_mode_works_without_sample_csv(env):
    at = click(run_app(), "Try with sample data")
    assert not at.exception and not at.error
    assert metric(at, "Transactions") == "500"
    assert any("local dataset" in c.value for c in at.caption)
    assert len(at.dataframe) >= 1


def test_threshold_change_reclassifies_without_losing_results(env):
    at = click(run_app(), "Try with sample data")
    base = int(metric(at, "Flagged").replace(",", ""))
    at.slider[0].set_value(0.0).run()                       # slider rerun, no button click
    assert not at.exception
    assert metric(at, "Transactions") == "500"             # results persisted (regression test)
    assert metric(at, "Flagged") == "500"                   # threshold 0 flags everything
    assert metric(at, "Not flagged") == "0"
    at.slider[0].set_value(1.0).run()
    assert int(metric(at, "Flagged").replace(",", "")) <= base
    assert int(metric(at, "Flagged").replace(",", "")) + int(metric(at, "Not flagged").replace(",", "")) == 500


def test_reset_button_restores_saved_threshold(env, artifacts):
    at = click(run_app(), "Try with sample data")
    at.slider[0].set_value(0.9).run()
    at = click(at, "Reset to saved threshold")
    assert at.slider[0].value == pytest.approx(artifacts.threshold)


def test_threshold_explorer_shows_validation_numbers(env):
    at = run_app()
    labels = {m.label for m in at.metric}
    assert {"Precision", "Recall", "False alarms", "Missed frauds"} <= labels


def test_labels_in_sample_trigger_performance_expander(env):
    at = click(run_app(), "Try with sample data")
    assert any("uploaded labels" in c.value for c in at.caption)


def test_missing_model_shows_friendly_error_not_traceback(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "empty")
    st.cache_resource.clear()
    at = run_app()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)
    assert any("python -m src.train" in i.value for i in at.info)


def test_sample_unavailable_shows_error(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_PATH", tmp_path / "absent.csv")
    st.cache_data.clear()
    at = click(run_app(), "Try with sample data")
    assert not at.exception
    assert any("No sample available" in e.value for e in at.error)


def test_missing_curve_file_is_handled(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "no_results")
    at = run_app()
    assert not at.exception
    assert any("threshold_curve_validation.csv" in i.value for i in at.info)

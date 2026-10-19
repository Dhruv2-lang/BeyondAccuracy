import json

import numpy as np
import pandas as pd
import pytest

from src import config, predict
from src.predict import (
    ArtifactNotFoundError, FraudModelError, InputValidationError, InvalidMetadataError,
    InvalidThresholdError, apply_threshold, load_artifacts, load_metadata, predict_fraud,
    validate_threshold,
)
from tests.synthetic import FEATURES


# ----------------------------- normal behaviour ----------------------------- #
def test_predict_returns_probabilities_and_flags(artifacts, transactions):
    out = predict_fraud(artifacts, transactions)
    assert len(out) == len(transactions)
    assert out["fraud_probability"].between(0, 1).all()
    assert set(out["flagged_as_fraud"].unique()) <= {0, 1}
    assert list(out.columns[:30]) == FEATURES


def test_default_threshold_is_the_saved_one(artifacts, transactions):
    out = predict_fraud(artifacts, transactions)
    expected = (out["fraud_probability"] >= artifacts.threshold).astype(int)
    assert (out["flagged_as_fraud"] == expected).all()


def test_threshold_override_and_monotonicity(artifacts, transactions):
    low = predict_fraud(artifacts, transactions, threshold=0.0)["flagged_as_fraud"].sum()
    mid = predict_fraud(artifacts, transactions, threshold=0.5)["flagged_as_fraud"].sum()
    high = predict_fraud(artifacts, transactions, threshold=1.0)["flagged_as_fraud"].sum()
    assert low == len(transactions)           # threshold 0 flags everything
    assert low >= mid >= high


def test_probabilities_do_not_depend_on_threshold(artifacts, transactions):
    a = predict_fraud(artifacts, transactions, threshold=0.1)["fraud_probability"]
    b = predict_fraud(artifacts, transactions, threshold=0.9)["fraud_probability"]
    pd.testing.assert_series_equal(a, b)


def test_feature_order_is_restored(artifacts, transactions):
    shuffled = transactions[list(reversed(transactions.columns))]
    a = predict_fraud(artifacts, transactions)["fraud_probability"].to_numpy()
    b = predict_fraud(artifacts, shuffled)["fraud_probability"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_input_is_not_mutated(artifacts, transactions):
    before = transactions.copy()
    predict_fraud(artifacts, transactions)
    pd.testing.assert_frame_equal(before, transactions)


def test_single_row(artifacts, transactions):
    assert len(predict_fraud(artifacts, transactions.head(1))) == 1


def test_integer_columns_are_accepted(artifacts, transactions):
    transactions["Time"] = transactions["Time"].astype(int)
    predict_fraud(artifacts, transactions)


def test_apply_threshold_matches_predict(artifacts, transactions):
    out = predict_fraud(artifacts, transactions)
    np.testing.assert_array_equal(
        apply_threshold(out["fraud_probability"], artifacts.threshold), out["flagged_as_fraud"])


# ------------------------------ invalid inputs ------------------------------ #
def test_empty_input(artifacts, transactions):
    with pytest.raises(InputValidationError, match="no transactions"):
        predict_fraud(artifacts, transactions.iloc[0:0])


def test_not_a_dataframe(artifacts):
    with pytest.raises(InputValidationError, match="DataFrame"):
        predict_fraud(artifacts, [[1, 2, 3]])


def test_missing_feature(artifacts, transactions):
    with pytest.raises(InputValidationError, match=r"Missing required.*V14"):
        predict_fraud(artifacts, transactions.drop(columns=["V14"]))


def test_extra_column_rejected_by_default(artifacts, transactions):
    transactions["Class"] = 0
    with pytest.raises(InputValidationError, match=r"Unexpected.*Class"):
        predict_fraud(artifacts, transactions)


def test_extra_column_allowed_when_requested(artifacts, transactions):
    transactions["Class"] = 0
    out = predict_fraud(artifacts, transactions, allow_extra=True)
    assert "Class" not in out.columns


def test_duplicate_columns(artifacts, transactions):
    dup = pd.concat([transactions, transactions[["V1"]]], axis=1)
    with pytest.raises(InputValidationError, match="Duplicate"):
        predict_fraud(artifacts, dup)


def test_non_numeric_column(artifacts, transactions):
    transactions["Amount"] = ["abc"] * len(transactions)
    with pytest.raises(InputValidationError, match=r"Non-numeric.*Amount"):
        predict_fraud(artifacts, transactions)


def test_boolean_column_rejected(artifacts, transactions):
    transactions["V1"] = True
    with pytest.raises(InputValidationError, match="Non-numeric"):
        predict_fraud(artifacts, transactions)


def test_missing_values(artifacts, transactions):
    transactions.loc[transactions.index[3], "V5"] = np.nan
    with pytest.raises(InputValidationError, match=r"Missing values.*V5"):
        predict_fraud(artifacts, transactions)


def test_infinite_values(artifacts, transactions):
    transactions.loc[transactions.index[0], "V2"] = np.inf
    with pytest.raises(InputValidationError, match="Infinite"):
        predict_fraud(artifacts, transactions)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf"), "0.5", True, [0.5]])
def test_invalid_threshold_values(artifacts, transactions, bad):
    with pytest.raises(InvalidThresholdError):
        predict_fraud(artifacts, transactions, threshold=bad)


@pytest.mark.parametrize("good", [0, 1, 0.0, 0.5, np.float32(0.25)])
def test_valid_threshold_values(good):
    assert 0.0 <= validate_threshold(good) <= 1.0


# ------------------------------ artifact errors ----------------------------- #
def test_missing_model_file(artifact_dir, tmp_path):
    with pytest.raises(ArtifactNotFoundError, match="python -m src.train"):
        load_artifacts(tmp_path / "nope.pkl", artifact_dir / config.METADATA_FILENAME)


def test_missing_metadata_file(artifact_dir, tmp_path):
    with pytest.raises(ArtifactNotFoundError, match="metadata"):
        load_artifacts(artifact_dir / config.MODEL_FILENAME, tmp_path / "nope.json")


def test_missing_artifacts_are_also_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path / "m.pkl", tmp_path / "m.json")


def test_corrupt_model_file(artifact_dir, tmp_path):
    bad = tmp_path / "model.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(FraudModelError, match="Could not load model"):
        load_artifacts(bad, artifact_dir / config.METADATA_FILENAME)


def test_corrupt_metadata_json(artifact_dir, tmp_path):
    bad = tmp_path / "metadata.json"
    bad.write_text("{not json")
    with pytest.raises(InvalidMetadataError, match="not valid JSON"):
        load_metadata(bad)


@pytest.mark.parametrize("patch, match", [
    ({"threshold": 1.7}, "threshold"),
    ({"threshold": "high"}, "threshold"),
    ({"feature_names": []}, "feature_names"),
    ({"feature_names": ["a", "a"]}, "feature_names"),
    ({"feature_names": "V1"}, "feature_names"),
])
def test_invalid_metadata(tmp_path, patch, match):
    meta = {"threshold": 0.5, "feature_names": ["a", "b"], **patch}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(InvalidMetadataError, match=match):
        load_metadata(path)


def test_metadata_missing_key(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"threshold": 0.5}))
    with pytest.raises(InvalidMetadataError, match="feature_names"):
        load_metadata(path)


def test_model_and_metadata_feature_count_mismatch(artifact_dir, tmp_path):
    meta = json.loads((artifact_dir / config.METADATA_FILENAME).read_text())
    meta["feature_names"] = meta["feature_names"][:10]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(InvalidMetadataError, match="do not belong together"):
        load_artifacts(artifact_dir / config.MODEL_FILENAME, path)


def test_default_paths_follow_environment(artifact_dir, monkeypatch, transactions):
    monkeypatch.setattr(config, "MODEL_DIR", artifact_dir)
    arts = load_artifacts()
    assert len(predict_fraud(arts, transactions)) == len(transactions)


# ----------------------------------- CLI ------------------------------------ #
def test_cli_roundtrip(artifact_dir, monkeypatch, tmp_path, transactions, capsys):
    monkeypatch.setattr(config, "MODEL_DIR", artifact_dir)
    src_csv, out_csv = tmp_path / "in.csv", tmp_path / "out.csv"
    transactions.assign(Class=0).to_csv(src_csv, index=False)   # Class is dropped with a note
    assert predict.main(["--input", str(src_csv), "--output", str(out_csv), "--threshold", "0.3"]) == 0
    scored = pd.read_csv(out_csv)
    assert {"fraud_probability", "flagged_as_fraud"} <= set(scored.columns)
    assert "Scored 50" in capsys.readouterr().out


def test_cli_reports_errors_without_traceback(artifact_dir, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "MODEL_DIR", artifact_dir)
    assert predict.main(["--input", str(tmp_path / "missing.csv")]) == 1
    assert "error:" in capsys.readouterr().err

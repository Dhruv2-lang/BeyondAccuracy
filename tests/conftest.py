import json

import joblib
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src import config, evaluate, preprocess
from src.predict import load_artifacts
from tests.synthetic import FEATURES, make_synthetic_creditcard


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_creditcard()


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, synthetic_df):
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, synthetic_df):
    """A tiny but real model + metadata, built once per test session."""
    out = tmp_path_factory.mktemp("models")
    X_train, X_val, _, y_train, y_val, _ = preprocess.split_data(synthetic_df)
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0, n_jobs=1,
                          scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))
    model.fit(X_train, y_train)
    sel = evaluate.select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    joblib.dump(model, out / config.MODEL_FILENAME)
    (out / config.METADATA_FILENAME).write_text(
        json.dumps({"threshold": sel["threshold"], "feature_names": FEATURES,
                    "threshold_selection_rule": sel["rule"], "random_state": 0}),
        encoding="utf-8",
    )
    return out


@pytest.fixture(scope="session")
def artifacts(artifact_dir):
    return load_artifacts(artifact_dir / config.MODEL_FILENAME, artifact_dir / config.METADATA_FILENAME)


@pytest.fixture()
def transactions(synthetic_df) -> pd.DataFrame:
    return synthetic_df.drop(columns=["Class"]).head(50).copy()

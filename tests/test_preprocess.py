import numpy as np
import pandas as pd
import pytest

from src import preprocess
from tests.synthetic import FEATURES


def test_split_sizes_and_stratification(synthetic_df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = preprocess.split_data(synthetic_df)
    n = len(synthetic_df)
    assert len(X_tr) + len(X_va) + len(X_te) == n
    assert abs(len(X_tr) / n - 0.70) < 0.01
    assert abs(len(X_va) / n - 0.15) < 0.01
    assert abs(len(X_te) / n - 0.15) < 0.01
    overall = synthetic_df["Class"].mean()
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - overall) < 0.005          # stratified


def test_splits_are_disjoint(synthetic_df):
    X_tr, X_va, X_te, *_ = preprocess.split_data(synthetic_df)
    ids = [set(X.index) for X in (X_tr, X_va, X_te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert set.union(*ids) == set(synthetic_df.index)


def test_split_is_reproducible_and_seed_sensitive(synthetic_df):
    a = preprocess.split_data(synthetic_df, random_state=42)
    b = preprocess.split_data(synthetic_df, random_state=42)
    c = preprocess.split_data(synthetic_df, random_state=7)
    assert a[0].index.equals(b[0].index) and a[2].index.equals(b[2].index)
    assert not a[2].index.equals(c[2].index)


def test_target_is_not_in_features_and_order_is_kept(synthetic_df):
    X_tr, *_ = preprocess.split_data(synthetic_df)
    assert "Class" not in X_tr.columns
    assert list(X_tr.columns) == FEATURES


def test_invalid_fractions(synthetic_df):
    with pytest.raises(ValueError):
        preprocess.split_data(synthetic_df, val_fraction=0.6, test_fraction=0.6)


def test_missing_dataset_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/creditcard.csv"):
        preprocess.load_dataset(tmp_path / "creditcard.csv")


def test_load_dataset_roundtrip(synthetic_csv, synthetic_df):
    assert preprocess.load_dataset(synthetic_csv).shape == synthetic_df.shape


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.drop(columns=["Class"]), "Class"),
    (lambda d: d.assign(Class=2), "binary"),
    (lambda d: d.assign(Class=0), "both classes"),
    (lambda d: d.assign(V1=np.nan), "non-finite"),
    (lambda d: d.assign(V1="x"), "Non-numeric"),
])
def test_dataset_validation(synthetic_df, mutate, match):
    with pytest.raises(ValueError, match=match):
        preprocess.validate_dataset(mutate(synthetic_df.copy()))

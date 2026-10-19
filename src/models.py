"""Model/experiment definitions.

Every experiment is trained on the *training split only*.  Scaling (logistic
regression) and resampling (SMOTE, undersampling) are fitted on training rows
and never see validation or test data.  Each ``fit_experiment`` call returns a
plain fitted estimator exposing ``predict_proba`` so it can be saved and used
for inference without any imbalance-handling machinery.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config

# Hyper-parameters shared by every XGBoost experiment so that differences come
# from imbalance handling, not from tuning.  (No hyper-parameter search is done.)
DEFAULT_XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    n_jobs=-1,
)


def _xgb(params: Optional[dict], random_state: int, **overrides) -> XGBClassifier:
    merged = {**DEFAULT_XGB_PARAMS, **(params or {}), **overrides}
    return XGBClassifier(random_state=random_state, **merged)


def _fit_logreg(X, y, params, random_state):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    return model.fit(X, y)


def _fit_xgb_unweighted(X, y, params, random_state):
    return _xgb(params, random_state).fit(X, y)


def _fit_xgb_weighted(X, y, params, random_state):
    spw = float((y == 0).sum() / (y == 1).sum())
    return _xgb(params, random_state, scale_pos_weight=spw).fit(X, y)


def _fit_xgb_smote(X, y, params, random_state):
    from imblearn.over_sampling import SMOTE

    # SMOTE interpolates between nearest neighbours, so distances must not be
    # dominated by Time/Amount.  Scale on the training split, oversample in
    # scaled space, then map back to raw units (trees are scale-invariant).
    scaler = StandardScaler().fit(X)
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(scaler.transform(X), y)
    X_res = pd.DataFrame(scaler.inverse_transform(X_res), columns=X.columns)
    return _xgb(params, random_state).fit(X_res, y_res)


def _fit_xgb_undersample(X, y, params, random_state):
    from imblearn.under_sampling import RandomUnderSampler

    X_res, y_res = RandomUnderSampler(random_state=random_state).fit_resample(X, y)
    return _xgb(params, random_state).fit(X_res, y_res)


EXPERIMENTS: Dict[str, Callable] = {
    "logreg_baseline": _fit_logreg,
    "xgb_unweighted": _fit_xgb_unweighted,
    "xgb_weighted": _fit_xgb_weighted,
    "xgb_smote": _fit_xgb_smote,
    "xgb_undersample": _fit_xgb_undersample,
}

DESCRIPTIONS = {
    "logreg_baseline": "StandardScaler + LogisticRegression, no imbalance handling",
    "xgb_unweighted": "XGBoost, no imbalance handling",
    "xgb_weighted": "XGBoost with scale_pos_weight = n_legit / n_fraud (train split)",
    "xgb_smote": "XGBoost trained on SMOTE-oversampled training data",
    "xgb_undersample": "XGBoost trained on randomly undersampled training data",
}


def fit_experiment(name: str, X_train, y_train, params: Optional[dict] = None,
                   random_state: int = config.RANDOM_STATE):
    """Fit the named experiment on training data and return the estimator."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Choose from: {sorted(EXPERIMENTS)}")
    y_train = np.asarray(y_train).astype(int) if not hasattr(y_train, "iloc") else y_train
    return EXPERIMENTS[name](X_train, y_train, params, random_state)

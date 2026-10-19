import numpy as np
import pytest

from src import evaluate


def _toy(seed=0, n=2000, pos=60):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:pos] = 1
    p = np.clip(rng.normal(0.15, 0.12, n) + y * 0.55, 0, 1)
    return y, p


def test_selected_threshold_meets_precision_target_and_maximises_recall():
    y, p = _toy()
    sel = evaluate.select_threshold(y, p, min_precision=0.8)
    m = evaluate.compute_metrics(y, p, sel["threshold"])
    assert m["precision"] >= 0.8
    # No other threshold with precision >= 0.8 has strictly higher recall.
    for t in np.unique(p):
        other = evaluate.compute_metrics(y, p, t)
        if other["precision"] >= 0.8:
            assert other["recall"] <= m["recall"] + 1e-12


def test_fallback_to_f1_when_precision_target_unreachable():
    y, p = _toy()
    sel = evaluate.select_threshold(y, p, min_precision=1.01)
    assert "unattainable" in sel["rule"]
    assert 0 <= sel["threshold"] <= 1


def test_threshold_requires_positives():
    with pytest.raises(ValueError):
        evaluate.select_threshold(np.zeros(10), np.linspace(0, 1, 10))


def test_threshold_selection_ignores_data_it_was_not_given():
    """Selecting on validation labels must not depend on any other split."""
    y_val, p_val = _toy(seed=1)
    y_test, p_test = _toy(seed=2)
    t1 = evaluate.select_threshold(y_val, p_val)["threshold"]
    _ = evaluate.compute_metrics(y_test, p_test, t1)    # test only evaluates
    assert evaluate.select_threshold(y_val, p_val)["threshold"] == t1


def test_compute_metrics_counts_and_accuracy_trap():
    y = np.array([0] * 98 + [1] * 2)
    p = np.zeros(100)                                    # never flags anything
    m = evaluate.compute_metrics(y, p, 0.5)
    assert m["accuracy"] == 0.98 and m["recall"] == 0.0
    assert m["majority_class_accuracy"] == 0.98
    assert m["confusion_matrix"] == {"tn": 98, "fp": 0, "fn": 2, "tp": 0}
    assert m["fraud_cases"] == 2 and m["legitimate_cases"] == 98


def test_threshold_table_matches_brute_force():
    y, p = _toy(seed=3)
    table = evaluate.threshold_table(y, p, n_points=51, extra=[0.4179530143737793])
    assert 0.4179530143737793 in set(table["threshold"])
    for _, row in table.iloc[::7].iterrows():
        m = evaluate.compute_metrics(y, p, row["threshold"])
        c = m["confusion_matrix"]
        assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (c["tp"], c["fp"], c["fn"], c["tn"])
        assert row["recall"] == pytest.approx(m["recall"])
        assert row["precision"] == pytest.approx(m["precision"])


def test_expected_cost():
    assert evaluate.expected_cost(fp=10, fn=2, cost_fp=5, cost_fn=100) == 250.0


def _original_train_py_rule(y_val, val_probs, min_precision=0.80):
    """Verbatim logic of the threshold rule in the previously committed src/train.py."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    if len(eligible):
        return float(thresholds[eligible[np.argmax(recall[:-1][eligible])]])
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("min_precision", [0.5, 0.8, 0.95, 1.01])
def test_threshold_rule_matches_previously_committed_behaviour(seed, min_precision):
    y, p = _toy(seed=seed)
    assert evaluate.select_threshold(y, p, min_precision)["threshold"] == _original_train_py_rule(y, p, min_precision)

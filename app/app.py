"""Streamlit dashboard for the BeyondAccuracy fraud detector.

Run from anywhere with:  streamlit run app/app.py
All inference goes through ``src.predict`` (no duplicated model logic here).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import expected_cost  # noqa: E402
from src.predict import (  # noqa: E402
    FLAG_COLUMN, PROBABILITY_COLUMN, FraudModelError, InputValidationError,
    apply_threshold, load_artifacts, predict_fraud,
)

MAX_ROWS = 200_000
SAMPLE_FRAUD_ROWS, SAMPLE_LEGIT_ROWS = 20, 480
CURVE_FILE = config.RESULTS_DIR / "threshold_curve_validation.csv"

st.set_page_config(page_title="Beyond Accuracy — Fraud Detection", page_icon="💳", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    """Return ``(artifacts, error_message)``; never raises."""
    try:
        return load_artifacts(), None
    except FraudModelError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_threshold_curve(path: str, mtime: float):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Preparing sample transactions…")
def build_sample(sample_path: str, data_path: str):
    """Sample CSV if present, otherwise draw a sample from the local dataset."""
    if Path(sample_path).exists():
        return pd.read_csv(sample_path), f"Using {Path(sample_path).name}."
    if Path(data_path).exists():
        full = pd.read_csv(data_path)
        fraud = full[full[config.TARGET_COLUMN] == 1]
        legit = full[full[config.TARGET_COLUMN] == 0]
        sample = pd.concat([
            fraud.sample(min(SAMPLE_FRAUD_ROWS, len(fraud)), random_state=config.RANDOM_STATE),
            legit.sample(min(SAMPLE_LEGIT_ROWS, len(legit)), random_state=config.RANDOM_STATE),
        ]).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        return sample, (
            f"Drew {len(sample)} random transactions from the local dataset "
            f"({min(SAMPLE_FRAUD_ROWS, len(fraud))} known frauds + {min(SAMPLE_LEGIT_ROWS, len(legit))} legitimate). "
            "Fraud is deliberately over-represented here compared with real life (~0.17%)."
        )
    raise FileNotFoundError(
        "No sample available: neither data/sample_transactions.csv nor data/creditcard.csv was found. "
        "Upload your own CSV instead."
    )


# --------------------------------------------------------------------------- #
# Page header and model loading
# --------------------------------------------------------------------------- #
st.title("💳 Beyond Accuracy: Credit Card Fraud Detection")
st.markdown(
    "A fraud detector built for severe class imbalance — because on this dataset, "
    "**99.8% accuracy is achievable while catching no fraud at all**."
)
st.warning(
    "**Model scores are not verdicts.** A high fraud probability means a transaction resembles past "
    "fraud; it does not establish that fraud occurred. Flagged transactions should be reviewed by a "
    "person, and legitimate customers will sometimes be flagged (false alarms).",
    icon="⚠️",
)

artifacts, load_error = get_artifacts()
if artifacts is None:
    st.error(f"The model could not be loaded.\n\n{load_error}")
    st.info("Train it locally with `python -m src.train` (needs `data/creditcard.csv`), then reload this page.")
    st.stop()

saved_threshold = artifacts.threshold
meta = artifacts.metadata

with st.sidebar:
    st.header("Model")
    st.write(f"**Type:** {meta.get('model_type', type(artifacts.model).__name__)}")
    if meta.get("selected_experiment"):
        st.write(f"**Experiment:** `{meta['selected_experiment']}`")
    st.write(f"**Saved threshold:** `{saved_threshold:.4f}`")
    if meta.get("threshold_selection_rule"):
        st.caption(f"Chosen on validation data: {meta['threshold_selection_rule']}.")
    st.write(f"**Features:** {len(artifacts.feature_names)}")
    with st.expander("Feature order"):
        st.code(", ".join(artifacts.feature_names))


def reset_threshold():
    st.session_state["threshold"] = saved_threshold


if "threshold" not in st.session_state:
    st.session_state["threshold"] = saved_threshold

tab_upload, tab_explore = st.tabs(["📁 Score Transactions", "🎚️ Threshold Explorer"])

# --------------------------------------------------------------------------- #
# Tab 1: scoring
# --------------------------------------------------------------------------- #
with tab_upload:
    st.subheader("Upload a CSV of transactions")
    st.caption(
        "Required columns: " + ", ".join(artifacts.feature_names[:3]) + ", …, " + artifacts.feature_names[-1]
        + f" ({len(artifacts.feature_names)} numeric columns). A 'Class' column is optional and, if present, "
        "is used only to show how the current threshold would have performed."
    )

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("Try with sample data")

    def score(data: pd.DataFrame, source: str) -> None:
        """Validate and score once; the threshold is applied separately below."""
        labels = None
        if config.TARGET_COLUMN in data.columns:
            labels = data[config.TARGET_COLUMN].copy()
            data = data.drop(columns=[config.TARGET_COLUMN])
        if len(data) > MAX_ROWS:
            raise InputValidationError(f"Too many rows ({len(data):,}); the limit is {MAX_ROWS:,}.")
        scored = predict_fraud(artifacts, data)
        st.session_state["scored"] = scored.drop(columns=[FLAG_COLUMN])
        st.session_state["labels"] = labels
        st.session_state["source"] = source

    try:
        if use_sample:
            sample, note = build_sample(str(config.SAMPLE_PATH), str(config.DATA_PATH))
            score(sample, "sample")
            st.session_state["source_note"] = note
        elif uploaded is not None and st.session_state.get("uploaded_id") != (uploaded.name, uploaded.size):
            try:
                frame = pd.read_csv(uploaded)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                raise InputValidationError(f"Could not read the file as CSV: {exc}") from exc
            score(frame, f"upload: {uploaded.name}")
            st.session_state["uploaded_id"] = (uploaded.name, uploaded.size)
            st.session_state["source_note"] = f"Scored {uploaded.name}."
    except (InputValidationError, FileNotFoundError) as exc:
        for key in ("scored", "labels", "source", "source_note"):
            st.session_state.pop(key, None)
        st.error(f"Could not score this input: {exc}")
    except FraudModelError as exc:
        st.error(f"Model error: {exc}")

    st.slider(
        "Fraud probability threshold", 0.0, 1.0, step=0.001, format="%.3f", key="threshold",
        help="Transactions with probability at or above this value are flagged. "
             "Changing it re-classifies instantly; the model is not retrained or re-run.",
    )
    st.button("Reset to saved threshold", on_click=reset_threshold,
              disabled=abs(st.session_state["threshold"] - saved_threshold) < 1e-9)
    threshold = st.session_state["threshold"]

    scored = st.session_state.get("scored")
    if scored is None:
        st.info("Upload a CSV or click **Try with sample data** to see predictions here.")
    else:
        st.caption(st.session_state.get("source_note", ""))
        flags = apply_threshold(scored[PROBABILITY_COLUMN], threshold)
        results = scored.assign(**{FLAG_COLUMN: flags})
        front = [PROBABILITY_COLUMN, FLAG_COLUMN, "Time", "Amount"]
        results = results[front + [c for c in results.columns if c not in front]]
        results = results.sort_values(PROBABILITY_COLUMN, ascending=False)
        results.index.name = "row"

        total, flagged = len(results), int(flags.sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions", f"{total:,}")
        m2.metric("Flagged", f"{flagged:,}")
        m3.metric("Not flagged", f"{total - flagged:,}")
        m4.metric("Flag rate", f"{flagged / total:.2%}")

        labels = st.session_state.get("labels")
        if labels is not None and set(pd.unique(labels.dropna())) <= {0, 1}:
            actual = labels.loc[results.index].astype(int).to_numpy()
            pred = results[FLAG_COLUMN].to_numpy()
            tp = int(((pred == 1) & (actual == 1)).sum()); fp = int(((pred == 1) & (actual == 0)).sum())
            fn = int(((pred == 0) & (actual == 1)).sum()); tn = int(((pred == 0) & (actual == 0)).sum())
            with st.expander("How this threshold performed on the uploaded labels", expanded=True):
                st.dataframe(pd.DataFrame(
                    [[tn, fp], [fn, tp]], index=["Actually legitimate", "Actually fraud"],
                    columns=["Not flagged", "Flagged"]))
                prec = tp / (tp + fp) if tp + fp else float("nan")
                rec = tp / (tp + fn) if tp + fn else float("nan")
                st.write(f"Precision **{prec:.1%}** · Recall **{rec:.1%}** · "
                         f"{fp} false alarm(s) · {fn} missed fraud(s).")
                st.caption("Computed on your uploaded labels only; this is not the model's held-out test performance.")

        st.dataframe(results)
        st.download_button(
            "Download results as CSV", results.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="fraud_results.csv", mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# Tab 2: threshold exploration
# --------------------------------------------------------------------------- #
with tab_explore:
    st.subheader("Understand the precision-recall trade-off")
    st.markdown(
        "Moving the threshold trades off **catching more fraud (recall)** against **fewer false alarms "
        "(precision)**:\n\n"
        "- **Lower threshold** → more transactions flagged → more fraud caught, but more legitimate "
        "customers disturbed and more review work.\n"
        "- **Higher threshold** → fewer flags → fewer false alarms, but more fraud slips through.\n\n"
        "The numbers below are measured on the **validation set** the threshold was tuned on, "
        "using the slider value from the first tab."
    )
    threshold = st.session_state["threshold"]
    st.write(f"Current threshold: **{threshold:.3f}** (saved model threshold: {saved_threshold:.3f})")

    if CURVE_FILE.exists():
        curve = load_threshold_curve(str(CURVE_FILE), CURVE_FILE.stat().st_mtime)
        row = curve.iloc[(curve["threshold"] - threshold).abs().argmin()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{row['precision']:.1%}")
        c2.metric("Recall", f"{row['recall']:.1%}")
        c3.metric("False alarms", f"{int(row['fp']):,}")
        c4.metric("Missed frauds", f"{int(row['fn']):,}")
        n_val = int(row["tp"] + row["fp"] + row["fn"] + row["tn"])
        st.caption(
            f"Validation set: {n_val:,} transactions, {int(row['tp'] + row['fn'])} fraudulent. "
            f"At this threshold, {int(row['flagged']):,} were flagged, of which "
            f"{int(row['fp']):,} were false alarms."
        )

        st.markdown("**Weigh the costs** (placeholder values — use your own):")
        k1, k2 = st.columns(2)
        cost_fp = k1.number_input("Cost of one false alarm", min_value=0.0, value=config.COST_FALSE_POSITIVE, step=1.0)
        cost_fn = k2.number_input("Cost of one missed fraud", min_value=0.0, value=config.COST_FALSE_NEGATIVE, step=10.0)
        costs = curve["fp"] * cost_fp + curve["fn"] * cost_fn
        best = curve.iloc[int(costs.argmin())]
        st.write(
            f"Expected cost at the current threshold: **{expected_cost(int(row['fp']), int(row['fn']), cost_fp, cost_fn):,.0f}**. "
            f"With these costs the cheapest threshold on the validation set is **{best['threshold']:.3f}** "
            f"(cost {costs.min():,.0f})."
        )
        st.line_chart(curve.set_index("threshold")[["precision", "recall"]])
    else:
        st.info("`results/threshold_curve_validation.csv` was not found. It is created by `python -m src.train`.")

    figures = [("threshold_tradeoff.png", "Precision & recall vs threshold (validation)"),
               ("precision_recall_curve.png", "Precision-recall curve"),
               ("confusion_matrix.png", "Final test-set confusion matrix at the saved threshold")]
    shown = False
    for name, caption in figures:
        path = config.ASSETS_DIR / name
        if path.exists():
            st.image(str(path), caption=caption)
            shown = True
    if not shown:
        st.caption("Figures appear here after `python -m src.train` has generated them in `assets/`.")

st.markdown("---")
st.caption(
    "Methodology and reproduction steps: see the "
    "[GitHub repository](https://github.com/Dhruv2-lang/BeyondAccuracy). "
    "This is an educational project, not a production fraud system."
)

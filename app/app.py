
"""Streamlit dashboard for BeyondAccuracy fraud detection."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_fraud


st.set_page_config(
    page_title="BeyondAccuracy | Fraud Detection",
    page_icon="💳",
    layout="wide",
)

FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def load_sample_data():
    """Load a local sample file or sample rows from the local dataset."""
    sample_path = ROOT / "data" / "sample_transactions.csv"
    dataset_path = ROOT / "data" / "creditcard.csv"

    if sample_path.exists():
        return pd.read_csv(sample_path)

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        return df.sample(
            n=min(50, len(df)),
            random_state=42,
        ).reset_index(drop=True)

    return None


try:
    model, metadata = get_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("From the project root, run: python src/train.py")
    st.stop()


default_threshold = float(metadata["threshold"])
feature_names = metadata["feature_names"]

st.title("💳 BeyondAccuracy")
st.subheader("Credit Card Fraud Detection")
st.markdown(
    "An imbalanced-classification project focused on identifying "
    "potentially fraudulent transactions without relying on accuracy alone."
)

st.caption(
    f"Model: XGBoost | Decision threshold: {default_threshold:.4f} "
    f"| Expected features: {len(feature_names)}"
)

upload_tab, explorer_tab = st.tabs(
    ["📁 Transaction Predictions", "🎚️ Threshold Explorer"]
)

with upload_tab:
    st.subheader("Upload transactions")
    st.write(
        "Upload a CSV containing the 30 model input features: "
        "Time, V1–V28, and Amount. A Class column, if present, is removed."
    )

    uploaded_file = st.file_uploader(
        "Choose a transaction CSV",
        type=["csv"],
        key="transaction_upload",
    )

    sample_col, threshold_col = st.columns([1, 2])

    with sample_col:
        use_sample = st.button(
            "Try sample transactions",
            use_container_width=True,
        )

    with threshold_col:
        threshold = st.slider(
            "Fraud probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold,
            step=0.01,
            help=(
                "Lower thresholds flag more transactions and may increase "
                "false alarms. Higher thresholds flag fewer transactions."
            ),
        )

    data = None

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

    elif use_sample:
        try:
            data = load_sample_data()
            if data is None:
                st.warning(
                    "No local sample data was found. Upload a CSV to continue."
                )
            else:
                st.info(
                    "Using local sample rows. These demonstrate the app "
                    "and are not new or live transactions."
                )
        except Exception as exc:
            st.error(f"Could not load sample data: {exc}")

    if data is not None:
        if data.empty:
            st.error("The CSV contains no transactions.")
        else:
            if "Class" in data.columns:
                data = data.drop(columns=["Class"])
                st.warning(
                    "Removed the Class column. This dashboard produces "
                    "predictions; it does not evaluate ground-truth labels."
                )

            # Use the shared validation and prediction function.
            try:
                results = predict_fraud(model, data, metadata)

                # Recalculate labels with the current dashboard slider.
                results["is_fraud"] = (
                    results["fraud_probability"] >= threshold
                ).astype(int)

                flagged = int(results["is_fraud"].sum())
                total = len(results)
                unflagged = total - flagged

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Transactions", f"{total:,}")
                metric2.metric("Flagged for review", f"{flagged:,}")
                metric3.metric("Not flagged", f"{unflagged:,}")

                st.caption(
                    "A flagged transaction is a model prediction, not proof "
                    "of fraud. Review flagged transactions before taking action."
                )

                st.subheader("Prediction results")
                st.dataframe(
                    results.sort_values(
                        "fraud_probability",
                        ascending=False,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

            except (ValueError, TypeError) as exc:
                st.error(f"Input validation failed: {exc}")
            except Exception as exc:
                st.error(f"Could not generate predictions: {exc}")

    else:
        st.info(
            "Upload a CSV or click 'Try sample transactions' to see predictions."
        )


with explorer_tab:
    st.subheader("Understanding the precision–recall trade-off")
    st.write(
        "The decision threshold changes how many transactions are flagged. "
        "Lower thresholds may catch more fraud but can also increase false alarms."
    )

    st.metric(
        "Training-selected threshold",
        f"{default_threshold:.4f}",
    )

    st.markdown("**Precision–recall curve**")
    pr_path = ROOT / "assets" / "precision_recall_curve.png"
    if pr_path.exists():
        st.image(
            str(pr_path),
            caption="Precision–recall curve from the saved project analysis",
            use_container_width=True,
        )
    else:
        st.info("Precision–recall plot not found in assets/.")

    st.markdown("**Threshold trade-off**")
    threshold_path = ROOT / "assets" / "threshold_tradeoff.png"
    if threshold_path.exists():
        st.image(
            str(threshold_path),
            caption="Precision and recall at different thresholds",
            use_container_width=True,
        )
    else:
        st.info("Threshold trade-off plot not found in assets.")

st.divider()
st.caption(
    "BeyondAccuracy | Educational fraud-detection project. "
    "Model predictions should not be treated as definitive fraud decisions."
)

st.markdown(
    "[View the project on GitHub]"
    "(https://github.com/Dhruv2-lang/BeyondAccuracy)"
)


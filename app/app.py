import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Beyond Accuracy — Fraud Detection", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

@st.cache_data
def load_sample_data():
    return pd.read_csv("data/sample_transactions.csv")

model = load_model()

st.title("💳 Beyond Accuracy: Credit Card Fraud Detection")
st.markdown(
    "A fraud detection model built to handle severe class imbalance properly — "
    "because on this dataset, **99.9% accuracy still misses real fraud**."
)

tab1, tab2 = st.tabs(["📁 Upload Transactions", "🎚️ Threshold Explorer"])

with tab1:
    st.subheader("Upload a CSV of transactions")
    st.caption("File must have the same columns as the training data (Time, V1-V28, Amount) — no 'Class' column needed.")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    with col2:
        st.write("")
        st.write("")
        use_sample = st.button("Try with sample data")

    threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01, key="upload_threshold")

    data = None
    if use_sample:
        data = load_sample_data()
        st.info("Using built-in sample data (50 random transactions).")
    elif uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'Class' in data.columns:
            data = data.drop('Class', axis=1)
            st.warning("Detected and removed a 'Class' column from the upload.")

    if data is not None:
        probs = model.predict_proba(data)[:, 1]
        data['fraud_probability'] = probs
        data['flagged_as_fraud'] = (probs >= threshold).astype(int)

        n_flagged = data['flagged_as_fraud'].sum()
        st.success(f"Found {n_flagged} transaction(s) flagged as fraud out of {len(data)} total.")

        st.dataframe(
            data.sort_values('fraud_probability', ascending=False),
            use_container_width=True
        )

        csv_download = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", csv_download, "fraud_results.csv", "text/csv")
    else:
        st.info("Upload a CSV or click 'Try with sample data' to see predictions here.")

with tab2:
    st.subheader("Understand the precision-recall tradeoff")
    st.markdown(
        "Fraud detection isn't just about accuracy. Moving the threshold trades off "
        "**catching more fraud (recall)** against **fewer false alarms (precision)**."
    )

    st.image("assets/threshold_tradeoff.png", caption="Precision & Recall vs Decision Threshold", use_container_width=True)
    st.image("assets/precision_recall_curve.png", caption="Precision-Recall Curve", use_container_width=True)

st.markdown("---")
st.caption("Built as part of a fraud detection project focused on proper handling of severe class imbalance. See the [GitHub repo](https://github.com/Dhruv2-lang/BeyondAccuracy) for full methodology.")

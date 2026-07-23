import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def load_model(path=MODEL_PATH):
    """Load the trained fraud detection model."""
    return joblib.load(path)

def predict_fraud(model, transaction_df, threshold=0.5):
    """
    Predict fraud probability and label for one or more transactions.

    Args:
        model: trained XGBoost model
        transaction_df: DataFrame with the same columns used in training
        threshold: probability cutoff for flagging fraud

    Returns:
        DataFrame with added 'fraud_probability' and 'is_fraud' columns
    """
    probs = model.predict_proba(transaction_df)[:, 1]
    result = transaction_df.copy()
    result['fraud_probability'] = probs
    result['is_fraud'] = (probs >= threshold).astype(int)
    return result

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")

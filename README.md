# Beyond Accuracy: Credit Card Fraud Detection

A fraud detection project built around one core idea: on this dataset, a model can score **99.9% accuracy while still missing most fraud**. This project shows why that happens, and how to fix it with proper imbalance handling and evaluation.

## The Problem

Credit card fraud is rare — only 0.17% of transactions in this dataset are fraudulent (492 out of 284,807). With that level of imbalance, standard evaluation approaches are actively misleading. A model that predicts "not fraud" for every single transaction would still be 99.83% accuracy, while catching zero fraud.

This project walks through the full process of recognizing that trap, comparing techniques to fix it, and picking a model based on the tradeoff that actually matters — not the headline accuracy number.

## Key Findings

| Model | Accuracy | Fraud Recall | Fraud Precision |
|---|---|---|---|
| Baseline (no handling) | 99.95% | 80.0% | 90.0% |
| Class Weighting | 99.95% | 82.7% | 87.1% |
| SMOTE | 99.93% | 85.7% | 77.1% |
| Undersampling | 95.50% | 91.8% | **3.4%** |

**Undersampling looks best on recall alone, but it's a trap** — 96.6% of its fraud alerts are false alarms, which would be unusable in a real system. **Class weighting** offered the strongest balance of precision and recall while barely touching overall accuracy, and was the cheapest to compute since it required no synthetic data generation or downsampling. It was selected as the final model.

![Class Distribution](assets/class_distribution.png)

## Methodology

1. **Exploratory Data Analysis** — examined class imbalance, transaction amount and time patterns, and feature correlations with fraud
2. **Baseline models** — trained Logistic Regression and XGBoost with no imbalance handling to demonstrate how accuracy alone hides poor fraud detection
3. **Imbalance handling comparison** — tested class weighting, SMOTE (oversampling), and random undersampling against the baseline
4. **Evaluation** — used precision-recall curves rather than accuracy or ROC-AUC, since PR curves are more informative on severely imbalanced data, and selected a decision threshold based on an explicit recall target rather than the default 0.5 cutoff
5. **Explainability** — used SHAP to identify which features drive individual fraud predictions (V14, V10, V12, and V17 emerged as the strongest predictors, consistent with the correlation analysis from EDA)
6. **Interactive dashboard** — built a Streamlit app to upload transactions, view fraud probabilities, and explore the precision-recall threshold tradeoff live

## Why Precision-Recall Over ROC-AUC

On imbalanced datasets, ROC-AUC can look excellent even when a model is performing poorly on the minority class, because the false positive *rate* stays low simply due to the huge number of true negatives. Precision-recall curves give a more honest picture of how the model performs specifically on the class that matters — fraud.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · Streamlit


## Limitations & Future Work

Most features (V1–V28) are PCA-anonymized for privacy, which limits interpretability of what each feature actually represents in a business sense

The dataset covers a fixed two-day window of 2013 European card transactions — a production system would need continuous retraining as fraud patterns evolve over time

The classification threshold used here was chosen to illustrate a recall-driven tradeoff; a real deployment would tune this against the actual business cost of false positives versus missed fraud

No hyperparameter tuning was performed on the final model — results reflect default XGBoost parameters with class weighting applied


## Running Locally

```bash
git clone [https://github.com/Dhruv2-lang/BeyondAccuracy.git](https://github.com/Dhruv2-lang/BeyondAccuracy.git)
cd BeyondAccuracy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py 



## Project Structure
```text
├── data/                       # Dataset (not committed — see below)
│   └── sample_transactions.csv
├── notebooks/
│   ├── 01_eda.ipynb            # Class imbalance, amount/time patterns, correlations
│   ├── 02_baseline.ipynb       # Naive models, the accuracy trap
│   ├── 03_imbalance_handling.ipynb # Class weighting vs SMOTE vs undersampling
│   ├── 04_evaluation.ipynb     # Precision-recall curves, threshold selection
│   └── 05_finalize_model.ipynb # Final model training + SHAP explainability
├── src/
│   └── predict.py              # Reusable model loading and prediction function
├── models/
│   └── model.pkl               # Final trained XGBoost model (class-weighted)
├── app/
│   └── app.py                  # Streamlit dashboard
├── assets/                     # Saved plots used in this README
└── requirements.txt




# Data directory

The dataset is **not** stored in this repository (it is ~150 MB and is distributed by its owners).

Required file: `data/creditcard.csv`

- **Dataset:** "Credit Card Fraud Detection" — European cardholder transactions from September 2013
  collected by Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles),
  published on Kaggle. Obtain it from the dataset owner's page and review its licence/terms yourself.
- **Expected shape:** 284,807 rows × 31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`
  (`Class` = 1 for fraud; 492 positives).
- Alternatively keep the file anywhere and set `BEYONDACCURACY_DATA_PATH=/path/to/creditcard.csv`.

Optional: `data/sample_transactions.csv` (same columns, with or without `Class`) powers the dashboard's
sample button. If it is absent, the dashboard samples from `creditcard.csv` instead.

CSV files in this directory are git-ignored.

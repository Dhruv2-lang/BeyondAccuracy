# Experiment summary

Split: stratified 70/15/15 (train/validation/test), seed 42. Thresholds chosen on validation data (rule: highest recall with precision >= 0.80); metrics below are on the untouched test split. Final model: **xgb_unweighted** (highest validation average precision).

| Experiment | Threshold | Accuracy | Precision | Recall | F1 | Avg. precision | ROC-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg_baseline | 0.4080 | 0.9992 | 0.860 | 0.662 | 0.748 | 0.750 | 0.956 | 49 | 8 | 25 | 42640 |
| xgb_unweighted * | 0.0862 | 0.9994 | 0.843 | 0.797 | 0.819 | 0.837 | 0.969 | 59 | 11 | 15 | 42637 |
| xgb_weighted | 0.4180 | 0.9994 | 0.847 | 0.824 | 0.836 | 0.839 | 0.966 | 61 | 11 | 13 | 42637 |
| xgb_smote | 0.8421 | 0.9992 | 0.747 | 0.797 | 0.771 | 0.820 | 0.974 | 59 | 20 | 15 | 42628 |
| xgb_undersample | 0.9994 | 0.9991 | 0.909 | 0.541 | 0.678 | 0.706 | 0.971 | 40 | 4 | 34 | 42644 |

Test split: 42,722 rows, 74 fraud. Always predicting 'legitimate' would score accuracy 0.9983 while catching no fraud.

Illustrative cost on the test split (false alarm = 5.0, missed fraud = 100.0; placeholder assumptions): logreg_baseline: 2540, xgb_unweighted: 1555, xgb_weighted: 1355, xgb_smote: 1600, xgb_undersample: 3420

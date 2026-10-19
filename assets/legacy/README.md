# Legacy figures (do not cite as current results)

These plots were produced by the earlier exploratory notebooks (`03`, `04`, `05`)
under a different protocol than the current pipeline:

- 80/20 train/test split with **no validation set**;
- default XGBoost hyper-parameters (not the tuned configuration in `src/train.py`);
- in `threshold_tradeoff.png`, the threshold (~0.003, "90 % recall") was selected **on the
  test set**, which is test-set tuning and is why it disagrees with the validation-selected
  threshold stored in `models/metadata.json` (~0.418);
- SHAP plots describe that earlier model, not the model saved by `python -m src.train`.

They are kept for provenance. Running `python -m src.train` regenerates current versions of the
threshold, precision-recall, confusion-matrix and model-comparison figures in `assets/`, and
notebooks `03` and `05` regenerate the comparison and SHAP figures.

# Customer Churn Prediction and Retention Analysis

## Project objective
Build an end-to-end classifier that ranks subscription customers by churn risk and translates scores into a limited-capacity retention strategy.

## Files
- `01_data_science_notebook.ipynb` — executable audit, EDA, feature engineering, modelling, tuning, evaluation, explainability, calibration and retention simulation.
- `02_cleaned_dataset.csv` — data with engineered predictors; source identifier and target retained for traceability.
- `03_model_artifact.joblib` — fitted preprocessing/model pipeline, decision threshold, feature list and random seed.
- `04_business_report.pdf` — business-focused report.
- `model_comparison.csv` — test metrics and best parameters for each model.
- `feature_importance.csv` — permutation importance estimates.
- `figures/` — charts used in the report.
- `requirements.txt` — package dependencies.

## Setup
Python 3.10+ is recommended. From this folder:
```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
jupyter notebook 01_data_science_notebook.ipynb
```
Keep `customer_churn_advanced_dataset.csv` in this folder before running the notebook. The source dataset is supplied separately with the assignment.

## Reproducibility and assumptions
- Random seed: 42.
- Stratified 80/20 train/test split.
- 3-fold stratified cross-validation.
- Hyperparameter selection uses average precision (PR-AUC).
- `customer_id` is an identifier, not a predictor.
- Missing numeric values are median-imputed; missing categorical values are mode-imputed inside the pipeline.
- Class weights are used; resampling is not applied.
- The final threshold is selected on training out-of-fold predictions by maximizing F2, giving recall more weight.
- Contact cost ($5), retained customer value ($100), and retention success rate (25%) are hypothetical simulation assumptions only.
- Model associations are not causal explanations. Validate on representative current production data before deployment.

## Run order
1. Place the source CSV beside the notebook.
2. Install dependencies.
3. Run all notebook cells from top to bottom.
4. Review the data audit and plots, then inspect model metrics and the assumptions in the report.
5. Do not compare or tune against the held-out test set during model development.

## Current execution environment
The supplied run used Python 3.13.5, pandas 2.2.3, and scikit-learn 1.8.0. Package versions may differ across environments.

# Machine failure prediction

This project predicts whether a machine will fail (`0` = no failure, `1` = failure) using the AI4I 2020 predictive-maintenance dataset.

## Project structure

- `data/raw/` — original dataset
- `data/processed/` — cleaned data and train/test artifacts
- `python/` — reusable data-cleaning and preparation scripts
- `notebooks/` — exploratory analysis, preparation, and baseline modelling
- `results/` — exported baseline metrics and error analyses

## Requirements

- Python 3.9 or later
- Dependencies listed in `requirements.txt`

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

The AI4I 2020 preventive-maintenance dataset is included in `data/raw/`. If you need the original source, it's available at:
https://archive.ics.uci.edu/ml/datasets/ai4i+2020+preventive+maintenance

## Run the data pipeline

From the project root, activate the virtual environment and run the pipeline:

```powershell
.venv\Scripts\Activate.ps1
python python/cleaning.py
python python/data_preparation.py
```

The pipeline creates six CSV files in `data/processed/`:
- Raw and scaled features
- Target labels
- Original `record_index` for tracing predictions back to source records

### Verify the setup

After running the pipeline, check that these files exist in `data/processed/`:
- `X_train_raw.csv` and `X_test_raw.csv`
- `X_train_scaled.csv` and `X_test_scaled.csv`
- `y_train.csv` and `y_test.csv`

## Run the notebooks (exploratory analysis)

For detailed exploration, visualization, and model evaluation, run the notebooks in this order:

1. `01_eda.ipynb` — explore the raw data.
2. `02_data_cleaning.ipynb` — inspect and document cleaning steps.
3. `03_data_preparation.ipynb` — create and validate the train/test split.
4. `04_modeling.ipynb` — train and evaluate baseline models.

**Notes:**
- The split uses `test_size=0.2`, `random_state=42`, and stratification to balance classes.
- The `record_index` is preserved for tracing predictions back to source records and is never used as a model feature.
- Notebooks `02_` and `03_` document the steps performed by the `cleaning.py` and `data_preparation.py` scripts.

## Baseline models

The baseline compares four models:

- Dummy Classifier (baseline)
- Logistic Regression
- Random Forest
- Gradient Boosting

**Evaluation metrics:**
Because failures are rare (imbalanced classes), the evaluation focuses on:
- Precision, recall, F1-score
- PR-AUC and ROC-AUC
- Confusion matrices, false negatives, and false positives

We do NOT rely on accuracy alone.

**Decision threshold:**
The current baseline uses `probability >= 0.5` for predicting class `1`. This can be adjusted based on the cost of false positives vs. false negatives in your maintenance workflow.

**Results:**
Baseline metrics and error analyses are exported to `results/`.

## Key results

Baseline models on the 2,000-record development set (decision threshold 0.5):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| Dummy Classifier | 0.000 | 0.000 | 0.000 | 0.500 | 0.033 | 0 | 66 |
| Logistic Regression | 0.160 | 0.848 | 0.269 | 0.920 | 0.419 | 295 | 10 |
| Random Forest | 0.846 | 0.833 | 0.840 | 0.991 | 0.893 | 10 | 11 |
| Gradient Boosting | 0.881 | 0.788 | 0.832 | 0.994 | 0.915 | 7 | 14 |

The best current variant (notebook `05_feature_experiment.ipynb`) adds the
engineered `OSF criterion = Tool wear [min] * Torque [Nm]` feature:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| Gradient Boosting + OSF criterion | 0.983 | 0.848 | 0.911 | 0.995 | 0.936 | 1 | 10 |

These numbers come from the development set, which was already used during
exploration and feature design — treat them as a development check, not as
untouched final validation (see the limitations below).

## Important limitations

The current 2,000-record test split has already been used for model comparison and error analysis, so it should be treated as a development set rather than an untouched final test set. Further experiments should use cross-validation and, eventually, a separate final test set.

The data contains a project-specific correction of nine records whose failure target conflicted with all available failure flags. This assumption should be tested later with a sensitivity analysis.

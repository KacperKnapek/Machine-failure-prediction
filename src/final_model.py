"""Final model selection: reduced feature set, cost-based threshold, artifact.

Decision closed by this script:

- model: GradientBoostingClassifier(random_state=42) — confirmed by the
  hyperparameter comparison in notebook 05,
- features: baseline + OSF criterion minus raw Torque and Process
  temperature (7 features) — confirmed by the reduced-set experiment,
- threshold: chosen from out-of-fold probabilities on the training data
  using explicit FN:FP cost ratios; X_test is used only as a final
  development check, never for tuning.

Outputs:

- ``results/final_threshold_costs.csv`` — OOF cost table per threshold,
- ``results/final_model_summary.csv`` — X_test development check,
- ``models/final_model.joblib`` — fitted model, scaler and metadata.

Run from the project root: ``python src/final_model.py``
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluation import (
    CATEGORICAL_FEATURES,
    cross_validate,
    evaluate_thresholds,
    get_metrics,
)
from features import add_osf_criterion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

REDUNDANT_FEATURES = ["Torque [Nm]", "Process temperature [K]"]
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]
COST_RATIOS = [1, 5, 10, 20, 30, 50]


def load_reduced_data():
    X_train = pd.read_csv(PROCESSED / "X_train_raw.csv", index_col="record_index")
    X_test = pd.read_csv(PROCESSED / "X_test_raw.csv", index_col="record_index")
    y_train = pd.read_csv(PROCESSED / "y_train.csv", index_col="record_index").squeeze("columns")
    y_test = pd.read_csv(PROCESSED / "y_test.csv", index_col="record_index").squeeze("columns")

    X_train = add_osf_criterion(X_train).drop(columns=REDUNDANT_FEATURES)
    X_test = add_osf_criterion(X_test).drop(columns=REDUNDANT_FEATURES)
    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = load_reduced_data()
    model = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Threshold analysis on out-of-fold probabilities only.
    _, oof_proba = cross_validate(model, X_train, y_train, cv)
    threshold_table = evaluate_thresholds(y_train, oof_proba, THRESHOLDS)
    for ratio in COST_RATIOS:
        threshold_table[f"cost_FN_{ratio}x_FP"] = (
            threshold_table["false_positives"]
            + ratio * threshold_table["false_negatives"]
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    threshold_table.round(4).to_csv(RESULTS / "final_threshold_costs.csv", index=False)

    print("OOF threshold optima by cost ratio (cost = FP + ratio * FN):")
    for ratio in COST_RATIOS:
        column = f"cost_FN_{ratio}x_FP"
        best = threshold_table.loc[threshold_table[column].idxmin()]
        print(
            f"  FN = {ratio:>2}x FP: threshold {best['threshold']:.2f}, "
            f"FP {int(best['false_positives'])}, FN {int(best['false_negatives'])}, "
            f"precision {best['precision']:.4f}, recall {best['recall']:.4f}, "
            f"cost {int(best[column])}"
        )

    # Train the final model on the full training set.
    numeric_features = [
        column for column in X_train.columns
        if column not in CATEGORICAL_FEATURES
    ]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = StandardScaler()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    model.fit(X_train_scaled, y_train)
    y_proba_test = pd.Series(
        model.predict_proba(X_test_scaled)[:, 1], index=X_test.index
    )

    # Development check on X_test at candidate thresholds.
    summary_rows = []
    for threshold in sorted({0.5, *(
        threshold_table.loc[threshold_table[f"cost_FN_{ratio}x_FP"].idxmin(), "threshold"]
        for ratio in COST_RATIOS
    )}):
        y_pred = (y_proba_test >= threshold).astype(int)
        metrics = get_metrics(y_test, y_proba_test, threshold)
        metrics["threshold"] = threshold
        metrics["false_positives"] = int(((y_test == 0) & (y_pred == 1)).sum())
        metrics["false_negatives"] = int(((y_test == 1) & (y_pred == 0)).sum())
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows).set_index("threshold")
    summary.round(4).to_csv(RESULTS / "final_model_summary.csv")
    print("\nX_test development check (final model, reduced features):")
    print(summary.round(4).to_string())

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": X_train.columns.tolist(),
            "numeric_features": numeric_features,
            "decision_rule": "probability >= threshold -> class 1",
            "thresholds_by_cost_ratio": {
                ratio: float(
                    threshold_table.loc[
                        threshold_table[f"cost_FN_{ratio}x_FP"].idxmin(), "threshold"
                    ]
                )
                for ratio in COST_RATIOS
            },
        },
        MODELS / "final_model.joblib",
    )
    print(f"\nSaved model artifact to {MODELS / 'final_model.joblib'}")


if __name__ == "__main__":
    main()

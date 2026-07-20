"""Plot the predicted probability distribution of the final GB model.

Loads the saved artifact (``models/final_model.joblib``), scores X_test and
draws a histogram of predicted probabilities split by true class, with the
0.30 (5-10x) and 0.5 candidate thresholds marked for reference.

Run from the project root: ``python python/plot_final_probabilities.py``
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import add_osf_criterion
from visualization import plot_probability_distribution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"


def main() -> None:
    artifact = joblib.load(MODELS / "final_model.joblib")
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]
    numeric_features = artifact["numeric_features"]

    X_test = pd.read_csv(PROCESSED / "X_test_raw.csv", index_col="record_index")
    y_test = pd.read_csv(PROCESSED / "y_test.csv", index_col="record_index").squeeze("columns")

    X_test = add_osf_criterion(X_test)[features]
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    y_proba = pd.Series(model.predict_proba(X_test_scaled)[:, 1], index=X_test.index)

    thresholds = {
        "0.30 (5-10x)": artifact["thresholds_by_cost_ratio"][10],
        "0.50": 0.5,
    }
    fig = plot_probability_distribution(y_test, y_proba, thresholds=thresholds)

    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / "final_model_probability_distribution.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved probability distribution plot to {output_path}")


if __name__ == "__main__":
    main()

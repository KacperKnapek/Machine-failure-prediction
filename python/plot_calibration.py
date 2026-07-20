"""Calibration check for the final GB model (reliability diagram + Brier score).

Two probability sources are compared:

- OOF (train): out-of-fold probabilities from the same 5-fold CV used in
  ``final_model.py`` to pick the threshold. This is the more trustworthy
  view, since it never touches X_test.
- X_test: probabilities from the fitted artifact model on the held-out
  development check set (small, and already used during exploration -
  a secondary view, not the final word).

Run from the project root: ``python python/plot_calibration.py``
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from evaluation import cross_validate, get_calibration_data
from features import add_osf_criterion
from visualization import plot_calibration_curve


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

    X_train = pd.read_csv(PROCESSED / "X_train_raw.csv", index_col="record_index")
    X_test = pd.read_csv(PROCESSED / "X_test_raw.csv", index_col="record_index")
    y_train = pd.read_csv(PROCESSED / "y_train.csv", index_col="record_index").squeeze("columns")
    y_test = pd.read_csv(PROCESSED / "y_test.csv", index_col="record_index").squeeze("columns")

    X_train = add_osf_criterion(X_train)[features]
    X_test = add_osf_criterion(X_test)[features]

    # Same CV setup as final_model.py, so the OOF probabilities used here
    # are the ones the threshold decision was actually based on.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _, oof_proba = cross_validate(
        GradientBoostingClassifier(random_state=42), X_train, y_train, cv
    )

    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    test_proba = pd.Series(model.predict_proba(X_test_scaled)[:, 1], index=X_test.index)

    calibration_tables = {
        "OOF (train)": get_calibration_data(y_train, oof_proba),
        "X_test": get_calibration_data(y_test, test_proba),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    for label, (table, brier) in calibration_tables.items():
        print(f"{label}: Brier score {brier:.4f}")
        print(table.round(4).to_string(index=False))
        print()

    combined = pd.concat(
        {label: table for label, (table, _) in calibration_tables.items()},
        names=["source", "bin"],
    ).reset_index(level="source")
    combined.round(4).to_csv(RESULTS / "final_model_calibration.csv", index=False)

    fig = plot_calibration_curve(calibration_tables)
    output_path = RESULTS / "final_model_calibration.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved calibration plot to {output_path}")


if __name__ == "__main__":
    main()

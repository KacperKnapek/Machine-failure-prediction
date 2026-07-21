"""Drift-monitoring sanity check: X_train vs X_test.

There is no production data stream yet, so this script cannot check for
real drift. It exists to validate the drift_monitoring.py metric itself:
X_train and X_test come from the same stratified split of the same data,
so they should show no meaningful drift (low PSI, no significant KS
p-values). If that holds, the tool is trustworthy to point at a real new
data batch once one exists.

To check real drift later: replace the ``current`` load below with the new
production batch (same columns as X_train_raw.csv), re-run, and read
results/drift_check_train_vs_test.csv (rename the output as needed).

Run from the project root: ``python src/check_drift.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drift_monitoring import compute_drift_report
from features import add_osf_criterion
from visualization import plot_drift_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    reference = pd.read_csv(PROCESSED / "X_train_raw.csv", index_col="record_index")
    current = pd.read_csv(PROCESSED / "X_test_raw.csv", index_col="record_index")

    reference = add_osf_criterion(reference)
    current = add_osf_criterion(current)
    features = reference.columns.tolist()

    report = compute_drift_report(reference, current, features)

    RESULTS.mkdir(parents=True, exist_ok=True)
    report.round(4).to_csv(RESULTS / "drift_check_train_vs_test.csv")
    print("Sanity check (X_train vs X_test, same distribution expected):")
    print(report.round(4).to_string())

    fig = plot_drift_report(report)
    output_path = RESULTS / "drift_check_train_vs_test.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved drift report plot to {output_path}")


if __name__ == "__main__":
    main()

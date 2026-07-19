"""Sensitivity analysis for the target-label correction (330 vs 339 failures).

The project sets ``Machine failure`` to 0 for nine records whose failure
flags are all zero. This is a project assumption, not certain truth. The
script compares 5-fold cross-validation results of the current best
configuration (baseline features + OSF criterion) under both label variants:

- corrected: 330 positives (the current pipeline),
- uncorrected: 339 positives (original labels kept).

Each variant gets its own stratified 80/20 split with the standard project
parameters; the held-out 20% is not touched. Run from the project root:

    python python/sensitivity_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from cleaning import prepare_dataset
from evaluation import cross_validate, summarize_fold_metrics
from features import add_osf_criterion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "produkcja.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "sensitivity_target_correction.csv"

REPORTED_METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def main() -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for variant, correct_labels in [
        ("corrected (330 failures)", True),
        ("uncorrected (339 failures)", False),
    ]:
        df = prepare_dataset(
            RAW_DATA, correct_inconsistent_labels=correct_labels
        )
        X = df.drop(columns=["Machine failure"])
        y = df["Machine failure"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = add_osf_criterion(X_train)

        print(f"\n{variant}: {int(y.sum())} positives total, "
              f"{int(y_train.sum())} in train")

        for model_name, model in models.items():
            fold_metrics, _ = cross_validate(model, X_train, y_train, cv)
            summary = summarize_fold_metrics(fold_metrics)
            rows.append({
                "label_variant": variant,
                "positives_total": int(y.sum()),
                "model": model_name,
                **summary,
            })
            report = ", ".join(
                f"{metric} {summary[f'{metric}_mean']:.4f}"
                f"±{summary[f'{metric}_std']:.4f}"
                for metric in REPORTED_METRICS
            )
            print(f"  {model_name}: {report}")

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.round(6).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

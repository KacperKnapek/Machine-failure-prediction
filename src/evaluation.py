"""Cross-validation and out-of-fold evaluation helpers.

Consolidates the fold loops from ``notebooks/05_feature_experiment.ipynb``:
one training pass per fold produces both per-fold metrics and out-of-fold
probabilities, from which predictions at any threshold can be derived.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DECISION_THRESHOLD = 0.5
CATEGORICAL_FEATURES = ["Type_L", "Type_M"]


def prepare_fold_data(
    X_train_fold: pd.DataFrame, X_validation_fold: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale numeric features using only the training part of one fold."""
    X_train_fold = X_train_fold.copy()
    X_validation_fold = X_validation_fold.copy()

    numeric_features = [
        column
        for column in X_train_fold.columns
        if column not in CATEGORICAL_FEATURES
    ]

    scaler = StandardScaler()
    X_train_fold[numeric_features] = scaler.fit_transform(
        X_train_fold[numeric_features]
    )
    X_validation_fold[numeric_features] = scaler.transform(
        X_validation_fold[numeric_features]
    )

    return X_train_fold, X_validation_fold


def get_metrics(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Compute the project metrics for one set of probabilities.

    The decision rule is ``probability >= threshold -> class 1``; a record
    with probability exactly equal to the threshold is classified as failure.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": (y_true == y_pred).mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    threshold: float = DECISION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit ``model`` on each fold with fold-local scaling.

    Returns per-fold metrics (indexed by fold number) and out-of-fold
    probabilities aligned with the index of ``X``. Each model is trained
    once per fold; predictions at other thresholds can be derived from
    the probabilities without retraining.
    """
    fold_metrics = []
    y_proba_oof = pd.Series(index=X.index, dtype=float, name="y_proba")

    for train_indices, validation_indices in cv.split(X, y):
        X_train_fold = X.iloc[train_indices]
        X_validation_fold = X.iloc[validation_indices]
        y_train_fold = y.iloc[train_indices]
        y_validation_fold = y.iloc[validation_indices]

        X_train_fold, X_validation_fold = prepare_fold_data(
            X_train_fold, X_validation_fold
        )

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        y_proba = fold_model.predict_proba(X_validation_fold)[:, 1]

        y_proba_oof.iloc[validation_indices] = y_proba

        metrics = get_metrics(y_validation_fold, y_proba, threshold)
        metrics["fold"] = len(fold_metrics) + 1
        fold_metrics.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    return fold_metrics_df, y_proba_oof


def summarize_fold_metrics(fold_metrics_df: pd.DataFrame) -> dict:
    """Return the mean and standard deviation of each per-fold metric."""
    summary = {}
    for metric in fold_metrics_df.columns:
        summary[f"{metric}_mean"] = fold_metrics_df[metric].mean()
        summary[f"{metric}_std"] = fold_metrics_df[metric].std()
    return summary


def evaluate_thresholds(
    y_true: pd.Series, y_proba: pd.Series, thresholds
) -> pd.DataFrame:
    """Evaluate the decision rule ``proba >= threshold`` for each threshold.

    The ``ties`` column explicitly reports records whose probability equals
    the threshold; under the project convention they are classified as
    failures.
    """
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_positives": ((y_true == 0) & (y_pred == 1)).sum(),
            "false_negatives": ((y_true == 1) & (y_pred == 0)).sum(),
            "ties": (y_proba == threshold).sum(),
        })
    return pd.DataFrame(rows)


def get_calibration_data(
    y_true: pd.Series, y_proba: pd.Series, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Return a reliability-diagram table and the Brier score.

    Uses equal-width probability bins (``strategy="uniform"``); the
    predicted probabilities in this project are strongly bimodal, so
    quantile bins would collapse near 0 and 1 and hide the calibration
    gap in the middle range where the decision thresholds sit.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = pd.cut(y_proba, bins=n_bins, retbins=True)[1]
    counts = pd.cut(y_proba, bins=bin_edges, include_lowest=True).value_counts(
        sort=False
    )
    # calibration_curve silently drops empty bins, so align counts to the
    # non-empty ones by recomputing which bins actually produced a point.
    non_empty_counts = counts[counts > 0].to_numpy()

    table = pd.DataFrame({
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "count": non_empty_counts,
    })
    return table, brier_score_loss(y_true, y_proba)


def get_threshold_ties(
    y_proba: pd.Series, threshold: float = DECISION_THRESHOLD
) -> pd.Series:
    """Return records whose probability is exactly equal to the threshold.

    These records land on the decision boundary; the project convention
    ``probability >= threshold -> class 1`` classifies them as failures
    (e.g. record 5653 for the Random Forest baseline). Report them
    explicitly before any threshold tuning.
    """
    return y_proba[y_proba == threshold]

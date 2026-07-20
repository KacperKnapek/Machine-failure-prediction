"""Plotting helpers for the modelling notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

NO_FAILURE_COLOR = "#4C72B0"
FAILURE_COLOR = "#DD8452"
THRESHOLD_COLOR = "#55A868"
DRIFT_GOOD_COLOR = "#55A868"
DRIFT_WARNING_COLOR = "#E5B94E"
DRIFT_CRITICAL_COLOR = "#C44E52"


def plot_confusion_matrix_grid(matrices: dict) -> Figure:
    """Plot a grid of confusion matrices.

    ``matrices`` maps a row label (e.g. feature variant) to an inner dict
    mapping a column label (e.g. model name) to a confusion matrix.
    """
    row_labels = list(matrices)
    column_labels = list(matrices[row_labels[0]])

    fig, axes = plt.subplots(
        len(row_labels),
        len(column_labels),
        figsize=(5 * len(column_labels), 4 * len(row_labels)),
        squeeze=False,
    )

    for row, row_label in enumerate(row_labels):
        for column, column_label in enumerate(column_labels):
            ConfusionMatrixDisplay(
                matrices[row_label][column_label], display_labels=[0, 1]
            ).plot(ax=axes[row, column], cmap="Blues", colorbar=False)
            axes[row, column].set_title(f"{row_label}\n{column_label}")

    plt.tight_layout()
    return fig


def plot_probability_distribution(
    y_true: pd.Series,
    y_proba: pd.Series,
    thresholds: dict[str, float] | None = None,
    title: str = "Predicted failure probability by true class",
) -> Figure:
    """Plot a histogram of predicted probabilities split by true class.

    ``thresholds`` optionally maps a label (e.g. a cost ratio) to a decision
    threshold, drawn as a vertical line for reference.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [step / 40 for step in range(41)]

    ax.hist(
        y_proba[y_true == 0], bins=bins, alpha=0.7,
        color=NO_FAILURE_COLOR, label="No failure (0)",
    )
    ax.hist(
        y_proba[y_true == 1], bins=bins, alpha=0.7,
        color=FAILURE_COLOR, label="Failure (1)",
    )

    if thresholds:
        for label, threshold in thresholds.items():
            ax.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.5)
            ax.text(
                threshold, ax.get_ylim()[1] * 0.97, f" {label}",
                color=THRESHOLD_COLOR, rotation=90, va="top", ha="left", fontsize=9,
            )

    ax.set(
        title=title, xlabel="Predicted probability of failure",
        ylabel="Number of records", xlim=(0, 1),
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_calibration_curve(
    calibration_tables: dict[str, tuple[pd.DataFrame, float]],
    title: str = "Calibration (reliability diagram)",
) -> Figure:
    """Plot reliability curves for one or more probability sources.

    ``calibration_tables`` maps a label (e.g. "OOF (train)", "X_test") to
    ``(table, brier_score)`` as returned by ``evaluation.get_calibration_data``.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    colors = [NO_FAILURE_COLOR, FAILURE_COLOR, THRESHOLD_COLOR]
    for (label, (table, brier)), color in zip(calibration_tables.items(), colors):
        ax.plot(
            table["mean_predicted"], table["fraction_positive"],
            color=color, linewidth=1, zorder=1,
        )
        # Marker area scales with bin count (normalized within each series,
        # sqrt so area rather than radius tracks count), so sparse bins (a
        # handful of records) visibly carry less weight than the dense
        # extremes without the largest bin swallowing the plot.
        relative_size = (table["count"] / table["count"].max()) ** 0.5
        ax.scatter(
            table["mean_predicted"], table["fraction_positive"],
            s=30 + 250 * relative_size, color=color, zorder=2,
            label=f"{label} (Brier {brier:.4f})",
        )

    ax.set(
        title=title, xlabel="Mean predicted probability",
        ylabel="Observed fraction of failures", xlim=(0, 1), ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_drift_report(drift_report: pd.DataFrame) -> Figure:
    """Bar chart of per-feature PSI (or proportion diff), status-colored.

    ``drift_report`` is the output of ``drift_monitoring.compute_drift_report``.
    Numeric features (metric "psi") are colored by the standard PSI bands
    (< 0.1 good, 0.1-0.25 warning, > 0.25 critical); categorical features
    (metric "proportion_diff") are plotted on the same axis but are not
    colored by those bands, since it is a different unit.
    """
    from drift_monitoring import PSI_MODERATE, PSI_SIGNIFICANT

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(drift_report) + 2))

    colors = []
    for _, row in drift_report.iterrows():
        if row["metric"] != "psi":
            colors.append("#8C8C8C")
        elif row["value"] < PSI_MODERATE:
            colors.append(DRIFT_GOOD_COLOR)
        elif row["value"] < PSI_SIGNIFICANT:
            colors.append(DRIFT_WARNING_COLOR)
        else:
            colors.append(DRIFT_CRITICAL_COLOR)

    ax.barh(drift_report.index, drift_report["value"], color=colors)
    ax.axvline(PSI_MODERATE, color=DRIFT_WARNING_COLOR, linestyle="--", linewidth=1)
    ax.axvline(PSI_SIGNIFICANT, color=DRIFT_CRITICAL_COLOR, linestyle="--", linewidth=1)

    ax.set(
        title="Feature drift: current vs reference (PSI; gray bars = proportion diff)",
        xlabel="PSI (numeric features) / proportion difference (Type_L, Type_M)",
    )
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_threshold_analysis(
    threshold_results,
    feature_variant: str,
    model_names,
) -> Figure:
    """Plot metrics and error counts against the decision threshold.

    ``threshold_results`` is the long-format table produced with
    ``evaluation.evaluate_thresholds`` plus ``feature_variant`` and
    ``model`` columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for model_name in model_names:
        selection = (
            (threshold_results["feature_variant"] == feature_variant)
            & (threshold_results["model"] == model_name)
        )
        model_thresholds = threshold_results[selection]

        axes[0].plot(
            model_thresholds["threshold"], model_thresholds["precision"],
            marker="o", label=f"{model_name} precision",
        )
        axes[0].plot(
            model_thresholds["threshold"], model_thresholds["recall"],
            marker="o", label=f"{model_name} recall",
        )
        axes[0].plot(
            model_thresholds["threshold"], model_thresholds["f1"],
            marker="o", linestyle="--", label=f"{model_name} F1",
        )
        axes[1].plot(
            model_thresholds["threshold"], model_thresholds["false_positives"],
            marker="o", label=f"{model_name} FP",
        )
        axes[1].plot(
            model_thresholds["threshold"], model_thresholds["false_negatives"],
            marker="o", label=f"{model_name} FN",
        )

    axes[0].set(
        title="Metrics by threshold", xlabel="Decision threshold",
        ylabel="Score", ylim=(0, 1),
    )
    axes[1].set(
        title="Errors by threshold", xlabel="Decision threshold",
        ylabel="Number of records",
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig

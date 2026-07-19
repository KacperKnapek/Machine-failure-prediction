"""Plotting helpers for the modelling notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay


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

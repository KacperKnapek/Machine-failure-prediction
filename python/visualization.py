"""Plotting helpers for the modelling notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

NO_FAILURE_SCATTER = "lightgray"
FAILURE_SCATTER = "red"

DEFAULT_CORRELATION_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "Power [W]",
    "Temperature difference",
]
OSF_THRESHOLD = 11003.2

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
            ax.axvline(threshold, color=THRESHOLD_COLOR,
                       linestyle="--", linewidth=1.5)
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
    ax.plot([0, 1], [0, 1], linestyle="--",
            color="gray", label="Perfectly calibrated")

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
    ax.axvline(PSI_MODERATE, color=DRIFT_WARNING_COLOR,
               linestyle="--", linewidth=1)
    ax.axvline(PSI_SIGNIFICANT, color=DRIFT_CRITICAL_COLOR,
               linestyle="--", linewidth=1)

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


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Correlation matrix of process parameters",
) -> Figure:
    """Heatmap of the correlation between the process parameters (EDA)."""
    if columns is None:
        columns = DEFAULT_CORRELATION_COLUMNS
    columns = [column for column in columns if column in df.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[columns].corr(), annot=True, cmap="coolwarm", fmt=".2f",
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_failure_map(
    df: pd.DataFrame,
    types: tuple[str, ...] = ("L", "M", "H"),
    title: str = "When failures occur (red), by machine type",
) -> Figure:
    """Scatter of Rotational speed vs Torque per machine type, failures in red."""
    fig, axes = plt.subplots(
        1, len(types), figsize=(5 * len(types), 5), sharex=True, sharey=True,
        squeeze=False,
    )
    for ax, machine_type in zip(axes[0], types):
        subset = df[df["Type"] == machine_type]
        healthy = subset[subset["Machine failure"] == 0]
        failed = subset[subset["Machine failure"] == 1]
        ax.scatter(
            healthy["Rotational speed [rpm]"], healthy["Torque [Nm]"],
            s=12, color=NO_FAILURE_SCATTER, alpha=0.6, label="No failure",
        )
        ax.scatter(
            failed["Rotational speed [rpm]"], failed["Torque [Nm]"],
            s=20, color=FAILURE_SCATTER, alpha=0.8, label="Failure",
        )
        ax.set(title=f"Machine type: {machine_type}",
               xlabel="Rotational speed [rpm]")
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel("Torque [Nm]")
    axes[0, -1].legend()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_osf_criterion(
    df: pd.DataFrame,
    threshold: float = OSF_THRESHOLD,
    title: str = "OSF failure analysis (Tool wear × Torque)",
) -> Figure:
    """Tool wear vs Torque scatter with the empirical OSF boundary drawn in."""
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df, x="Tool wear [min]", y="Torque [Nm]", hue="OSF",
        size="Power [W]", sizes=(20, 300), alpha=0.7,
        palette={0: "#a1c9f4", 1: "#ffb3bd"}, ax=ax,
    )
    x_line = np.linspace(150, 250, 100)
    ax.plot(
        x_line, threshold / x_line, color="red", linestyle="--", linewidth=2,
        label=f"OSF boundary (~{threshold:.0f})",
    )
    ax.set(title=title, xlabel="Tool wear [min]", ylabel="Torque [Nm]")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_flag_analysis(
    df: pd.DataFrame,
    flag_col: str,
    value_col: str,
    kind: str = "box",
    palette: str = "Set1",
    title: str | None = None,
) -> Figure:
    """Distribution of ``value_col`` split by a failure-mechanism flag (EDA).

    ``kind`` is "box" or "violin" for the left panel; the right panel is always
    a KDE histogram. Used for TWF (Tool wear), HDF (Temperature difference) and
    PWF (Power).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if kind == "violin":
        sns.violinplot(
            x=flag_col, y=value_col, data=df, hue=flag_col,
            palette=palette, ax=axes[0], legend=False,
        )
    else:
        sns.boxplot(
            x=flag_col, y=value_col, hue=flag_col, data=df,
            palette=palette, ax=axes[0], legend=False,
        )
    axes[0].set_title(f"{value_col} by {flag_col}")

    sns.histplot(
        data=df, x=value_col, hue=flag_col, palette=palette, ax=axes[1], kde=True,
    )
    axes[1].set_title(f"Histogram of {value_col}")

    fig.suptitle(title or f"{flag_col}: {value_col}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances,
    title: str = "Feature importance (final GB model, impurity-based)",
    xlabel: str = "Importance",
) -> Figure:
    """Horizontal bar chart of feature importances (Series or {feature: value})."""
    if not isinstance(importances, pd.Series):
        importances = pd.Series(importances)
    importances = importances.sort_values()

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(importances) + 2))
    ax.barh(importances.index, importances.to_numpy(), color=NO_FAILURE_COLOR)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    return fig

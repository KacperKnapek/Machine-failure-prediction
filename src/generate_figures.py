"""Regenerate the key EDA and model figures into ``results/``.

Consolidates the most important plots from the notebooks so they exist as
saved artifacts and as reusable functions in ``visualization.py`` (single
source of truth). Two groups are produced:

- EDA figures from the raw dataset (``data/raw/produkcja.csv``): correlation
  matrix, failure map, the failure-mechanism distributions (TWF/HDF/PWF) and
  the OSF-criterion boundary.
- Final-model feature importance from the saved artifact
  (``models/final_model.joblib``), impurity-based, plus a CSV.

The final-evaluation figures (probability distribution, calibration, drift)
have their own scripts: ``plot_final_probabilities.py``, ``plot_calibration.py``
and ``check_drift.py``.

Run from the project root: ``python src/generate_figures.py``
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from visualization import (
    plot_correlation_matrix,
    plot_failure_map,
    plot_feature_importance,
    plot_flag_analysis,
    plot_osf_criterion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "produkcja.csv"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"


def load_raw_with_engineered_features() -> pd.DataFrame:
    """Raw dataset plus the two engineered columns used across the EDA."""
    df = pd.read_csv(RAW)
    df["Temperature difference"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(1)
    df["Power [W]"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)
    ).round(2)
    return df


def save(fig, name: str) -> None:
    output_path = RESULTS / name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")


def generate_eda_figures(df: pd.DataFrame) -> None:
    save(plot_correlation_matrix(df), "eda_correlation_matrix.png")
    save(plot_failure_map(df), "eda_failure_map.png")
    save(plot_osf_criterion(df), "eda_osf_criterion.png")
    save(
        plot_flag_analysis(df, "TWF", "Tool wear [min]", kind="box",
                           title="TWF vs Tool wear"),
        "eda_twf_tool_wear.png",
    )
    save(
        plot_flag_analysis(df, "HDF", "Temperature difference", kind="box",
                           title="HDF vs Temperature difference"),
        "eda_hdf_temperature_difference.png",
    )
    save(
        plot_flag_analysis(df, "PWF", "Power [W]", kind="violin",
                           title="PWF vs Power"),
        "eda_pwf_power.png",
    )


def generate_feature_importance() -> None:
    artifact = joblib.load(MODELS / "final_model.joblib")
    importances = pd.Series(
        artifact["model"].feature_importances_, index=artifact["features"]
    )

    importances.sort_values(ascending=False).round(4).to_csv(
        RESULTS / "final_model_feature_importance.csv",
        header=["importance"], index_label="feature",
    )
    print("Saved results/final_model_feature_importance.csv")
    save(plot_feature_importance(importances), "final_model_feature_importance.png")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    generate_eda_figures(load_raw_with_engineered_features())
    generate_feature_importance()


if __name__ == "__main__":
    main()

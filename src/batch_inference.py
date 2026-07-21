"""Validated batch inference for the final machine-failure model.

Run from the project root::

    python src/batch_inference.py data/test/valid_input.csv \
        results/batch_predictions.csv --threshold 0.30

The input contains raw measurements.  This module recreates the engineered
features used during training, validates the batch before prediction, and
writes traceable decisions together with probabilities and model metadata.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_THRESHOLD = 0.30

RAW_REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_INPUT_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ALLOWED_MACHINE_TYPES = {"L", "M", "H"}
REQUIRED_ARTIFACT_KEYS = {
    "model",
    "scaler",
    "features",
    "numeric_features",
    "decision_rule",
}


class InputValidationError(ValueError):
    """Raised when a batch cannot safely be passed to the model."""


def validate_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate raw input and return a numeric-normalized copy.

    Extra columns are allowed because a production extract can carry metadata.
    Only the explicitly required measurement columns become model features.
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise InputValidationError("Input must be a pandas DataFrame.")

    duplicate_columns = df_raw.columns[df_raw.columns.duplicated()].tolist()
    if duplicate_columns:
        raise InputValidationError(
            f"Input contains duplicate columns: {duplicate_columns}."
        )

    missing_columns = [
        column for column in RAW_REQUIRED_COLUMNS if column not in df_raw.columns
    ]
    if missing_columns:
        raise InputValidationError(
            f"Input is missing required columns: {missing_columns}."
        )

    if df_raw.empty:
        raise InputValidationError("Input contains no data rows.")

    df = df_raw.copy()
    required_col_with_missing = [
        column for column in RAW_REQUIRED_COLUMNS if df[column].isna().any()
    ]
    if required_col_with_missing:
        raise InputValidationError(
            f"Required columns contain missing values: {required_col_with_missing}."
        )

    invalid_type_values = set(df["Type"]) - ALLOWED_MACHINE_TYPES
    if invalid_type_values:
        invalid_types = sorted(repr(value) for value in invalid_type_values)
        raise InputValidationError(
            "Column 'Type' contains unsupported values: "
            f"{invalid_types}. Allowed values are {sorted(ALLOWED_MACHINE_TYPES)}."
        )

    for column in NUMERIC_INPUT_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any():
            invalid_rows = df.index[converted.isna()].tolist()
            raise InputValidationError(
                f"Column '{column}' contains non-numeric values at rows "
                f"{invalid_rows}."
            )
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise InputValidationError(
                f"Column '{column}' contains an infinite value."
            )
        df[column] = converted

    if "record_index" in df.columns:
        if df["record_index"].isna().any():
            raise InputValidationError(
                "Column 'record_index' contains missing values.")
        if df["record_index"].duplicated().any():
            duplicates = df.loc[
                df["record_index"].duplicated(keep=False), "record_index"
            ].tolist()
            raise InputValidationError(
                f"Column 'record_index' contains duplicates: {duplicates}."
            )

    return df


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create the training-time features using a batch-independent schema."""
    df = validate_input(df_raw)

    # Explicit binary columns are stable even when a batch contains only one
    # machine type. pd.get_dummies(drop_first=True) cannot guarantee that.
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)

    df["Temperature difference"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(1)
    df["Power [W]"] = (
        df["Rotational speed [rpm]"]
        * df["Torque [Nm]"]
        * (2 * np.pi / 60)
    ).round(2)
    df["OSF criterion"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the model artifact and verify its inference contract."""
    model_path = Path(model_path)
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")

    missing_keys = sorted(REQUIRED_ARTIFACT_KEYS - set(artifact))
    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {missing_keys}.")
    return artifact


def get_model_version(model_path: str | Path) -> str:
    """Return a content-based model version that changes with the artifact."""
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def predict_dataframe(
    df_raw: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model_version: str = "unknown",
) -> pd.DataFrame:
    """Predict a validated DataFrame and return a traceable decision report."""
    if isinstance(threshold, bool):
        raise ValueError("Threshold must be a number between 0 and 1.")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Threshold must be a number between 0 and 1.") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a number between 0 and 1.")

    df = engineer_features(df_raw)
    features = list(artifact["features"])
    numeric_features = list(artifact["numeric_features"])

    missing_features = [
        column for column in features if column not in df.columns]
    if missing_features:
        raise ValueError(
            f"Feature engineering did not create model features: {missing_features}."
        )

    X = df.loc[:, features].copy()
    scaled_values = artifact["scaler"].transform(X.loc[:, numeric_features])
    X[numeric_features] = pd.DataFrame(
        scaled_values,
        index=X.index,
        columns=numeric_features,
    )
    probabilities = artifact["model"].predict_proba(X)[:, 1]

    if "record_index" in df_raw.columns:
        record_ids = df_raw["record_index"].to_numpy()
    else:
        record_ids = df_raw.index.to_numpy()

    return pd.DataFrame(
        {
            "record_index": record_ids,
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
            "threshold": threshold,
            "model_version": model_version,
        }
    )


def run_batch(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Read a CSV batch, predict it, and optionally save the decision report."""
    input_path = Path(input_path)
    model_path = Path(model_path)

    try:
        df_raw = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise InputValidationError(
            "Input CSV is empty and has no header.") from exc

    artifact = load_model_artifact(model_path)
    result = predict_dataframe(
        df_raw,
        artifact,
        threshold=threshold,
        model_version=get_model_version(model_path),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validated batch inference with the final model."
    )
    parser.add_argument("input_path", type=Path, help="Raw input CSV path.")
    parser.add_argument("output_path", type=Path, help="Prediction CSV path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model artifact path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Decision threshold (default: {DEFAULT_THRESHOLD:.2f}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_batch(
            args.input_path,
            args.output_path,
            args.model_path,
            threshold=args.threshold,
        )
    except (FileNotFoundError, InputValidationError, ValueError) as exc:
        raise SystemExit(f"Batch inference failed: {exc}") from exc

    print(result.to_string(index=False))
    print(f"\nSaved {len(result)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()

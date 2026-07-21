"""Prepare the machine-failure dataset for the modelling step."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
}

FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLUMNS = ["UDI", "Product ID", "Air temperature [K]", *FAILURE_FLAGS]


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    correct_inconsistent_labels: bool = True,
) -> pd.DataFrame:
    """Create the modelling table from the raw CSV and optionally save it.

    ``correct_inconsistent_labels`` controls the project assumption that the
    nine records with ``Machine failure = 1`` but all failure flags equal to
    zero are labelling errors; pass ``False`` to keep the original labels
    (sensitivity analysis).
    """
    input_path = Path(input_path)
    df = pd.read_csv(input_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Temperature difference"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(1)
    df["Power [W]"] = (
        df["Rotational speed [rpm]"]
        * df["Torque [Nm]"]
        * (2 * np.pi / 60)
    ).round(2)

    if correct_inconsistent_labels:
        unknown_failure_type = (
            df[FAILURE_FLAGS].sum(axis=1).eq(0)
            & df["Machine failure"].eq(1)
        )
        df.loc[unknown_failure_type, "Machine failure"] = 0

    df = df.drop(columns=DROP_COLUMNS)
    df = pd.get_dummies(df, columns=["Type"], drop_first=True, dtype=int)

    expected_features = [
        "Type_L", "Type_M", "Process temperature [K]",
        "Temperature difference", "Rotational speed [rpm]", "Torque [Nm]",
        "Power [W]", "Tool wear [min]", "Machine failure",
    ]
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Unexpected machine types or missing columns: {sorted(missing_features)}"
        )

    df = df[expected_features]
    if df.isna().any().any():
        raise ValueError("Prepared dataset contains missing values")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/raw/produkcja.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/produkcja_clean.csv")
    )
    args = parser.parse_args()
    prepared = prepare_dataset(args.input, args.output)
    print(f"Saved {len(prepared)} rows to {args.output}")


if __name__ == "__main__":
    main()

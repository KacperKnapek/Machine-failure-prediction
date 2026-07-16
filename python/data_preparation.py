from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {
    "Type_L",
    "Type_M",
    "Process temperature [K]",
    "Temperature difference",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Power [W]",
    "Tool wear [min]",
    "Machine failure",
}

COLS_TO_SCALE = [
    "Process temperature [K]",
    "Temperature difference",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Power [W]",
    "Tool wear [min]",
]


def prepare_data(input_path: str | Path, output_dir: str | Path) -> None:
    """Prepare the cleaned dataset for modeling."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}")

    if df.isna().any().any():
        raise ValueError("Data contains missing values.")

    if df.duplicated().any():
        raise ValueError("Data contains duplicate rows.")

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    if "Machine failure" in X.columns:
        raise ValueError("'Machine failure' column should not be in features.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "'Machine failure' column should only contain 0 and 1.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    scaler = StandardScaler()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train_raw[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test_raw[COLS_TO_SCALE])

    if not X_train.index.equals(y_train.index):
        raise ValueError("Indices of X_train and y_train do not match.")

    if not X_test.index.equals(y_test.index):
        raise ValueError("Indices of X_test and y_test do not match.")

    if "Machine failure" in X_train.columns:
        raise ValueError("'Machine failure' column should not be in X_train.")

    if "record_index" in X_train.columns:
        raise ValueError("'record_index' column should not be in X_train.")

    train_means = X_train[COLS_TO_SCALE].mean()
    if not (train_means.abs() < 1e-10).all():
        raise ValueError("Train features were not correctly scaled.")

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_raw.to_csv(
        output_dir / "X_train_raw.csv",
        index=True,
        index_label="record_index",
    )
    X_test_raw.to_csv(
        output_dir / "X_test_raw.csv",
        index=True,
        index_label="record_index",
    )
    X_train.to_csv(
        output_dir / "X_train_scaled.csv",
        index=True,
        index_label="record_index",
    )
    X_test.to_csv(
        output_dir / "X_test_scaled.csv",
        index=True,
        index_label="record_index",
    )
    y_train.to_frame().to_csv(
        output_dir / "y_train.csv",
        index=True,
        index_label="record_index",
    )
    y_test.to_frame().to_csv(
        output_dir / "y_test.csv",
        index=True,
        index_label="record_index",
    )

    print("[1/4] Loaded data")
    print(f"      Shape: {df.shape}")
    print(f"      Target distribution: {y.value_counts().to_dict()}")
    print("[2/4] Split data")
    print(
        f"      Train: {X_train_raw.shape}, target: {y_train.value_counts().to_dict()}")
    print(
        f"      Test:  {X_test_raw.shape}, target: {y_test.value_counts().to_dict()}")
    print("[3/4] Scaled numeric features")
    print(f"      Max absolute train mean: {train_means.abs().max():.2e}")
    print("[4/4] Saved artifacts")
    print(f"      Output directory: {output_dir}")


if __name__ == "__main__":
    prepare_data(
        "data/processed/produkcja_clean.csv",
        "data/processed",
    )

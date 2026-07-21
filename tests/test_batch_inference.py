"""Tests for the production-facing batch inference boundary."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_inference import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    InputValidationError,
    engineer_features,
    predict_dataframe,
    run_batch,
)
from src.features import add_osf_criterion  # noqa: E402


TEST_DATA = PROJECT_ROOT / "data" / "test"
PROCESSED = PROJECT_ROOT / "data" / "processed"


class BatchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = joblib.load(DEFAULT_MODEL_PATH)

    def test_valid_csv_returns_complete_decision_report(self) -> None:
        result = run_batch(TEST_DATA / "valid_input.csv", threshold=0.30)

        self.assertEqual(len(result), 3)
        self.assertEqual(
            result.columns.tolist(),
            [
                "record_index",
                "probability",
                "prediction",
                "threshold",
                "model_version",
            ],
        )
        self.assertTrue(result["probability"].between(0, 1).all())
        self.assertTrue(result["prediction"].isin([0, 1]).all())
        self.assertTrue((result["threshold"] == 0.30).all())
        self.assertTrue(result["model_version"].str.match(r"sha256:[0-9a-f]{64}").all())

    def test_missing_column_is_rejected_before_prediction(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "Torque"):
            run_batch(TEST_DATA / "missing_column.csv")

    def test_header_only_csv_is_rejected(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "no data rows"):
            run_batch(TEST_DATA / "empty.csv")

    def test_each_machine_type_has_the_same_binary_schema(self) -> None:
        template = pd.read_csv(TEST_DATA / "valid_input.csv").iloc[[0]]

        expected = {
            "L": (1, 0),
            "M": (0, 1),
            "H": (0, 0),
        }
        for machine_type, binary_values in expected.items():
            with self.subTest(machine_type=machine_type):
                batch = template.copy()
                batch["Type"] = machine_type
                engineered = engineer_features(batch)
                actual = (
                    int(engineered.iloc[0]["Type_L"]),
                    int(engineered.iloc[0]["Type_M"]),
                )
                self.assertEqual(actual, binary_values)

    def test_non_numeric_measurement_is_rejected(self) -> None:
        batch = pd.read_csv(TEST_DATA / "valid_input.csv")
        batch["Torque [Nm]"] = batch["Torque [Nm]"].astype(object)
        batch.loc[1, "Torque [Nm]"] = "not-a-number"

        with self.assertRaisesRegex(InputValidationError, "non-numeric"):
            engineer_features(batch)

    def test_missing_measurement_is_rejected(self) -> None:
        batch = pd.read_csv(TEST_DATA / "valid_input.csv")
        batch.loc[1, "Tool wear [min]"] = np.nan

        with self.assertRaisesRegex(InputValidationError, "missing values"):
            engineer_features(batch)

    def test_unsupported_machine_type_is_rejected(self) -> None:
        batch = pd.read_csv(TEST_DATA / "valid_input.csv")
        batch.loc[1, "Type"] = "X"

        with self.assertRaisesRegex(InputValidationError, "unsupported values"):
            engineer_features(batch)

    def test_existing_record_index_is_preserved(self) -> None:
        batch = pd.read_csv(TEST_DATA / "valid_input.csv")
        batch.insert(0, "record_index", [101, 205, 999])

        result = predict_dataframe(batch, self.artifact, threshold=0.50)

        self.assertEqual(result["record_index"].tolist(), [101, 205, 999])

    def test_raw_batch_probabilities_match_direct_model_input(self) -> None:
        """Point 7: raw feature engineering must preserve model probabilities."""
        processed = pd.read_csv(
            PROCESSED / "X_test_raw.csv", index_col="record_index"
        ).head(100)
        raw_batch = pd.DataFrame(
            {
                "record_index": processed.index,
                "Type": np.select(
                    [processed["Type_L"].eq(1), processed["Type_M"].eq(1)],
                    ["L", "M"],
                    default="H",
                ),
                "Air temperature [K]": (
                    processed["Process temperature [K]"]
                    - processed["Temperature difference"]
                ),
                "Process temperature [K]": processed["Process temperature [K]"],
                "Rotational speed [rpm]": processed["Rotational speed [rpm]"],
                "Torque [Nm]": processed["Torque [Nm]"],
                "Tool wear [min]": processed["Tool wear [min]"],
            }
        )

        batch_result = predict_dataframe(raw_batch, self.artifact, threshold=0.30)

        direct = add_osf_criterion(processed)
        direct = direct.loc[:, self.artifact["features"]].copy()
        numeric = self.artifact["numeric_features"]
        scaled_values = self.artifact["scaler"].transform(direct.loc[:, numeric])
        direct[numeric] = pd.DataFrame(
            scaled_values,
            index=direct.index,
            columns=numeric,
        )
        expected_probabilities = self.artifact["model"].predict_proba(direct)[:, 1]

        np.testing.assert_allclose(
            batch_result["probability"].to_numpy(),
            expected_probabilities,
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

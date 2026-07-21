"""Experimental feature engineering shared by the modelling notebooks.

Base features (Power, Temperature difference) are created in ``cleaning.py``;
this module holds features designed later, during error analysis.
"""

from __future__ import annotations

import pandas as pd


OSF_FEATURE = "OSF criterion"


def add_osf_criterion(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``X`` with the OSF criterion feature added.

    The feature is ``Tool wear [min] * Torque [Nm]``. It uses only parameters
    available before a failure, but its design was inspired by the OSF label
    analysis, which must be stated in any report.
    """
    X = X.copy()
    X[OSF_FEATURE] = X["Tool wear [min]"] * X["Torque [Nm]"]
    return X

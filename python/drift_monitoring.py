"""Feature-drift detection between a reference dataset and a new one.

Ready to run once production data starts arriving after deployment: pass
the training data as ``reference`` and a batch of new production records as
``current``. Until then, ``check_drift.py`` uses X_train vs X_test as a
sanity check (they come from the same distribution by construction, so a
low-drift result there confirms the metric itself is working).

Two metrics per feature:

- **PSI** (Population Stability Index) for numeric features, binned on the
  reference distribution's deciles. Common reading: < 0.1 no meaningful
  shift, 0.1-0.25 moderate, > 0.25 significant.
- **Proportion difference** for the binary categorical features
  (``Type_L``, ``Type_M``), where decile binning does not apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from evaluation import CATEGORICAL_FEATURES

PSI_EPSILON = 1e-4
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    reference: pd.Series, current: pd.Series, n_bins: int = 10
) -> float:
    """PSI of ``current`` against ``reference``, binned on reference deciles."""
    bin_edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    )
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    reference_pct = pd.cut(reference, bin_edges).value_counts(
        sort=False, normalize=True
    )
    current_pct = pd.cut(current, bin_edges).value_counts(
        sort=False, normalize=True
    )

    # Empty bins would make the log ratio undefined; a small floor keeps
    # PSI finite without materially changing it for populated bins.
    reference_pct = reference_pct.clip(lower=PSI_EPSILON)
    current_pct = current_pct.clip(lower=PSI_EPSILON)

    return float(
        ((current_pct - reference_pct) * np.log(current_pct / reference_pct)).sum()
    )


def compute_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Per-feature drift report: PSI (or proportion diff) and a KS test."""
    rows = []
    for feature in features:
        ks_stat, ks_pvalue = ks_2samp(reference[feature], current[feature])

        if feature in CATEGORICAL_FEATURES:
            metric_name = "proportion_diff"
            metric_value = float(current[feature].mean() - reference[feature].mean())
        else:
            metric_name = "psi"
            metric_value = population_stability_index(reference[feature], current[feature])

        rows.append({
            "feature": feature,
            "metric": metric_name,
            "value": metric_value,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
        })

    return pd.DataFrame(rows).set_index("feature")

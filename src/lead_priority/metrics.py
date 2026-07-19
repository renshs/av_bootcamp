"""Competition metrics and rank-normalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def daily_average_precision(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
) -> tuple[float, pd.Series]:
    """Return macro-average AP across assignment dates and per-day values."""
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "score": np.asarray(scores, dtype=float),
            "date": pd.to_datetime(np.asarray(dates)).date,
        }
    )
    if frame[["target", "score", "date"]].isna().any().any():
        raise ValueError("Metric inputs must not contain missing values")

    per_day = frame.groupby("date", sort=True).apply(
        lambda group: average_precision_score(group["target"], group["score"]),
        include_groups=False,
    )
    return float(per_day.mean()), per_day


def within_day_percentile_rank(
    scores: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
) -> np.ndarray:
    """Convert scores to continuous percentile ranks separately within each day."""
    frame = pd.DataFrame(
        {
            "score": np.asarray(scores, dtype=float),
            "date": pd.to_datetime(np.asarray(dates)).date,
            "row_order": np.arange(len(scores)),
        }
    )
    ranked = frame.groupby("date", sort=False)["score"].rank(method="average", pct=True)
    return ranked.to_numpy(dtype=float)

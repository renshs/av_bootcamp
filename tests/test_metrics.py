import numpy as np

from lead_priority.metrics import daily_average_precision, within_day_percentile_rank


def test_daily_average_precision_is_macro_average() -> None:
    target = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.6])
    dates = np.array(["2026-01-01"] * 2 + ["2026-01-02"] * 3)

    value, per_day = daily_average_precision(target, scores, dates)

    assert np.isclose(per_day.iloc[0], 1.0)
    assert np.isclose(per_day.iloc[1], 0.5)
    assert np.isclose(value, 0.75)


def test_within_day_rank_preserves_order() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    dates = np.array(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"])

    ranked = within_day_percentile_rank(scores, dates)

    assert np.allclose(ranked, [0.5, 1.0, 0.5, 1.0])

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from lead_priority.metrics import within_day_percentile_rank


SEEDS = (42, 2026, 17)
N_TREES = 1000


def make_model(seed: int, iterations: int = N_TREES) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )


def predict_ensemble(
    models: Sequence[CatBoostClassifier],
    features: pd.DataFrame,
    dates: pd.Series,
) -> np.ndarray:
    ranked = []
    for model in models:
        scores = model.predict_proba(features)[:, 1]
        ranked.append(within_day_percentile_rank(scores, dates))
    return np.mean(ranked, axis=0)

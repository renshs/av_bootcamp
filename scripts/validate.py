from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score

from lead_priority.data import load_data
from lead_priority.features import categorical_feature_names, prepare_features
from lead_priority.metrics import daily_average_precision
from lead_priority.model import SEEDS, make_model, predict_ensemble


def folds(dates: pd.Series):
    days = sorted(pd.to_datetime(dates).dt.normalize().unique())
    boundaries = ((6, 9), (9, 12), (12, len(days)))
    for start, end in boundaries:
        valid_from = pd.Timestamp(days[start])
        valid_to = (
            pd.Timestamp(days[end])
            if end < len(days)
            else pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
        )
        yield valid_from, valid_to


def main() -> None:
    train, test, events = load_data()
    features, _ = prepare_features(train, test, events)
    cat_features = categorical_feature_names(features)
    dates = pd.to_datetime(train["assignment_date"])

    fold_scores = []
    for fold_number, (valid_from, valid_to) in enumerate(folds(dates), start=1):
        train_mask = dates < valid_from
        valid_mask = (dates >= valid_from) & (dates < valid_to)

        models = []
        for seed in SEEDS:
            model = make_model(seed, iterations=900)
            model.fit(
                features.loc[train_mask],
                train.loc[train_mask, "target"],
                cat_features=cat_features,
                eval_set=(features.loc[valid_mask], train.loc[valid_mask, "target"]),
                early_stopping_rounds=120,
                use_best_model=True,
            )
            models.append(model)

        scores = predict_ensemble(
            models, features.loc[valid_mask], dates.loc[valid_mask]
        )
        daily_ap, _ = daily_average_precision(
            train.loc[valid_mask, "target"],
            scores,
            dates.loc[valid_mask],
        )
        ap = average_precision_score(train.loc[valid_mask, "target"], scores)
        fold_scores.append(daily_ap)
        print(f"fold {fold_number}: daily_ap={daily_ap:.5f}, ap={ap:.5f}")

    print(f"mean daily_ap={sum(fold_scores) / len(fold_scores):.5f}")


if __name__ == "__main__":
    main()

# Lead prioritization solution v2

## Approach

The solution uses an ensemble of three CatBoostClassifier models with fixed
seeds 42, 2026, and 17. The final score is the mean percentile rank of the
three models calculated separately within each assignment_date. This matches
the primary Daily Average Precision ranking objective.

lead_id, user_id, raw assignment_ts, and assignment_date are excluded from
model features. Assignment hour, weekday, and weekend context are retained.

## Leakage prevention

All event features are computed only after applying the strict filter:

~~~text
event_ts < assignment_ts
~~~

This removes 16,124 unavailable train events. A unit test verifies that events
at the assignment timestamp and after assignment cannot enter either the base
or advanced event feature pipeline.

## Features

The v2 model uses 734 features, including 25 categorical features:

- all provided tabular features;
- window ratios, increments, and recent-versus-history intensity;
- smoothed funnel ratios such as favorite/view and positive/answered;
- event counts and recency by type over 1, 3, 7, 14, and 30 days;
- granular event windows from 1 hour to 21 days;
- exponentially decayed activity with half-lives from 6 hours to 30 days;
- event type, context, and source diversity and entropy;
- event_type x ctx_seq counts and recency;
- the five most recent event types, contexts, sources, prices, and recencies;
- event-type transitions and change counts;
- inter-event gaps, active span, active days, and event-price statistics.

## Validation

Three expanding-window folds are used:

| Fold | Train dates | Validation dates | Daily AP, 3 seeds |
|---|---|---|---:|
| 1 | Apr 07-12 | Apr 13-15 | 0.68674 |
| 2 | Apr 07-15 | Apr 16-18 | 0.71213 |
| 3 | Apr 07-18 | Apr 19-22 | 0.72469 |

The v2 OOF Daily AP over ten validation days is **0.70953**. The previous v1
ensemble scored **0.70216** locally and obtained **0.72241** Daily AP on the
competition test set.

No test labels, manual labeling, or lead-specific rules were used.

## Reproduction

~~~bash
uv sync
uv run pytest -q
uv run python scripts/validate.py
uv run python scripts/train.py
uv run python scripts/predict.py
~~~

`scripts/train.py` creates:

- `submission.csv`;
- three CatBoost `.cbm` files in `artifacts/models/`;
- `artifacts/models/metadata.json` with the exact feature schema and ensemble settings.

`scripts/predict.py` loads the saved models and recreates `submission.csv`
without retraining. The persisted-model output was verified to be byte-identical
to the training output.

## Open-source dependencies

- CatBoost;
- pandas;
- NumPy;
- scikit-learn;
- pytest for tests.

Exact versions are pinned in uv.lock.

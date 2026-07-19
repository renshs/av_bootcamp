# Lead prioritization

Solution for the lead ranking task. The main metric is Average Precision
calculated separately for each assignment date.

## Repository layout

- data/ - train, test and event logs
- src/lead_priority/ - feature engineering, metrics and model settings
- scripts/ - training, validation and inference entry points
- artifacts/models/ - fitted CatBoost models and feature metadata
- submission.csv - current best submission
- SOLUTION.md - validation details and feature notes

## Run

~~~bash
uv sync
uv run pytest -q
uv run python scripts/validate.py
uv run python scripts/train.py
uv run python scripts/predict.py
~~~

The event pipeline always applies event_ts < assignment_ts before aggregation.
The prediction script loads the saved models; it does not retrain them.

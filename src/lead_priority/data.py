from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"


def load_data(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    events = pd.read_csv(data_dir / "events.csv")
    return train, test, events


def check_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if submission.columns.tolist() != ["lead_id", "score"]:
        raise ValueError("Submission must contain lead_id and score")
    if len(submission) != len(test):
        raise ValueError("Submission length does not match test")
    if not submission["lead_id"].equals(test["lead_id"].astype(str)):
        raise ValueError("lead_id order does not match test")
    if not submission["lead_id"].is_unique:
        raise ValueError("lead_id is not unique")
    if submission["score"].isna().any():
        raise ValueError("score contains missing values")
    if not submission["score"].between(0, 1).all():
        raise ValueError("score must be between 0 and 1")

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from lead_priority.data import PROJECT_DIR, check_submission, load_data
from lead_priority.features import prepare_features
from lead_priority.model import predict_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_DIR / "artifacts/models"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        (args.model_dir / "metadata.json").read_text(encoding="utf-8")
    )

    train, test, events = load_data()
    _, test_features = prepare_features(train, test, events)
    if test_features.columns.tolist() != metadata["feature_names"]:
        raise ValueError("Feature schema differs from the saved model")

    models = []
    for filename in metadata["model_files"]:
        model = CatBoostClassifier()
        model.load_model(args.model_dir / filename)
        models.append(model)

    scores = predict_ensemble(models, test_features, test["assignment_date"])
    submission = pd.DataFrame({"lead_id": test["lead_id"].astype(str), "score": scores})
    check_submission(submission, test)
    submission.to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

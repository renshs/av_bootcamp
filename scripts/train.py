from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lead_priority.data import PROJECT_DIR, check_submission, load_data
from lead_priority.features import categorical_feature_names, prepare_features
from lead_priority.model import N_TREES, SEEDS, make_model, predict_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=N_TREES)
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_DIR / "artifacts/models"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test, events = load_data()
    train_features, test_features = prepare_features(train, test, events)
    cat_features = categorical_feature_names(train_features)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    models = []
    model_files = []

    for seed in SEEDS:
        model = make_model(seed, args.iterations)
        model.fit(train_features, train["target"], cat_features=cat_features)

        filename = f"catboost_events_v2_seed_{seed}.cbm"
        model.save_model(args.model_dir / filename)
        models.append(model)
        model_files.append(filename)
        print(f"trained seed {seed}")

    scores = predict_ensemble(models, test_features, test["assignment_date"])
    submission = pd.DataFrame({"lead_id": test["lead_id"].astype(str), "score": scores})
    check_submission(submission, test)
    submission.to_csv(args.output, index=False)

    metadata = {
        "feature_names": train_features.columns.tolist(),
        "categorical_features": cat_features,
        "seeds": list(SEEDS),
        "iterations": args.iterations,
        "model_files": model_files,
        "score_transform": "mean within-day percentile rank",
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

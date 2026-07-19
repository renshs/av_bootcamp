"""Leak-free feature engineering for tabular data and historical events."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


ID_COLUMNS = {"lead_id", "user_id"}
TIME_COLUMNS = {"assignment_ts", "assignment_date"}
NON_FEATURE_COLUMNS = ID_COLUMNS | TIME_COLUMNS | {"target"}
WINDOWS = (1, 3, 7, 14, 30, 90)
EVENT_WINDOWS = (1, 3, 7, 14, 30)
EVENT_HOUR_WINDOWS = (1, 6, 12, 48, 120, 240, 504)
EVENT_HALF_LIVES = (6, 24, 72, 168, 336, 720)
CATEGORICAL_COLUMNS = (
    "lead_source",
    "call_center",
    "region",
    "car_segment",
    "lead_channel",
    "user_tenure_bucket",
    "price_bucket",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Smoothed ratio that stays finite for sparse counters."""
    return (numerator.astype(float) + 0.5) / (denominator.astype(float) + 1.0)


def _window_groups(columns: Iterable[str]) -> dict[str, dict[int, str]]:
    groups: dict[str, dict[int, str]] = {}
    for column in columns:
        match = re.fullmatch(r"(.+)_([0-9]+)d", column)
        if match:
            groups.setdefault(match.group(1), {})[int(match.group(2))] = column
    return groups


def build_tabular_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model features available at assignment time."""
    source = frame.copy()
    assignment_ts = pd.to_datetime(source["assignment_ts"], errors="raise")
    feature_columns = [
        column for column in source.columns if column not in NON_FEATURE_COLUMNS
    ]
    result = source[feature_columns].copy()
    additions: dict[str, pd.Series | np.ndarray] = {}

    minute_of_day = assignment_ts.dt.hour * 60 + assignment_ts.dt.minute
    additions["assignment_minute_of_day"] = minute_of_day
    additions["assignment_hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    additions["assignment_hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    numeric_source = source.select_dtypes(include=[np.number])
    model_numeric = [
        column for column in numeric_source.columns if column not in NON_FEATURE_COLUMNS
    ]
    additions["numeric_missing_count"] = source[model_numeric].isna().sum(axis=1)

    groups = _window_groups(source.columns)
    for base, mapping in groups.items():
        if not all(window in mapping for window in WINDOWS):
            continue
        for short, long in ((1, 7), (3, 14), (7, 30), (30, 90)):
            short_values = source[mapping[short]]
            long_values = source[mapping[long]]
            additions[f"{base}_share_{short}_{long}"] = _safe_ratio(
                short_values, long_values
            )
            additions[f"{base}_rate_delta_{short}_{long}"] = (
                short_values / short - long_values / long
            )
        for previous, current in zip(WINDOWS[:-1], WINDOWS[1:]):
            additions[f"{base}_increment_{previous}_{current}"] = (
                source[mapping[current]] - source[mapping[previous]]
            )

    for window in WINDOWS:

        def col(name: str) -> pd.Series:
            return source[f"{name}_{window}d"]

        additions[f"favorite_per_view_{window}d"] = _safe_ratio(
            col("item_favorites"), col("item_views")
        )
        additions[f"detail_per_view_{window}d"] = _safe_ratio(
            col("detail_expands"), col("item_views")
        )
        additions[f"contact_per_view_{window}d"] = _safe_ratio(
            col("user_contacts"), col("item_views")
        )
        additions[f"chat_per_contact_{window}d"] = _safe_ratio(
            col("chat_opens"), col("user_contacts")
        )
        additions[f"call_per_contact_{window}d"] = _safe_ratio(
            col("call_clicks"), col("user_contacts")
        )
        additions[f"answered_per_assigned_{window}d"] = _safe_ratio(
            col("leadgen_prev_answered"), col("leadgen_prev_assigned")
        )
        additions[f"positive_per_answered_{window}d"] = _safe_ratio(
            col("leadgen_prev_positive"), col("leadgen_prev_answered")
        )
        additions[f"positive_per_assigned_{window}d"] = _safe_ratio(
            col("leadgen_prev_positive"), col("leadgen_prev_assigned")
        )
        engagement_columns = [
            f"{name}_{window}d"
            for name in (
                "item_views",
                "item_favorites",
                "detail_expands",
                "photo_swipes",
                "seller_page_views",
                "search_views",
                "query_refinements",
                "similar_item_clicks",
                "saved_search_matches",
                "user_contacts",
                "chat_opens",
                "call_clicks",
            )
        ]
        additions[f"engagement_total_{window}d"] = source[engagement_columns].sum(
            axis=1, min_count=1
        )

    result = pd.concat([result, pd.DataFrame(additions, index=source.index)], axis=1)
    for column in CATEGORICAL_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string").fillna("__MISSING__")
    return result


def _entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True, dropna=False).to_numpy()
    return float(-(probabilities * np.log(probabilities)).sum())


def build_event_features(
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    advanced: bool = False,
) -> pd.DataFrame:
    """Aggregate only events strictly earlier than each lead assignment."""
    assignment_map = assignments[["lead_id", "assignment_ts"]].copy()
    if assignment_map["lead_id"].duplicated().any():
        raise ValueError("lead_id must be unique in assignments")
    assignment_map["assignment_ts"] = pd.to_datetime(
        assignment_map["assignment_ts"], errors="raise"
    )

    history = events.copy()
    history["event_ts"] = pd.to_datetime(history["event_ts"], errors="raise")
    history = history.merge(
        assignment_map, on="lead_id", how="inner", validate="many_to_one"
    )
    history["hours_before_assignment"] = (
        history["assignment_ts"] - history["event_ts"]
    ).dt.total_seconds() / 3600
    history = history.loc[history["hours_before_assignment"] > 0].copy()
    history.sort_values(["lead_id", "event_ts"], inplace=True)

    grouped = history.groupby("lead_id", sort=False)
    features = grouped.agg(
        event_count=("event_type", "size"),
        event_type_nunique=("event_type", "nunique"),
        event_ctx_nunique=("ctx_seq", "nunique"),
        event_src_nunique=("src_slot", "nunique"),
        event_recency_hours=("hours_before_assignment", "min"),
        event_oldest_hours=("hours_before_assignment", "max"),
        event_active_days=("event_ts", lambda x: x.dt.date.nunique()),
        event_price_mean=("item_price_log", "mean"),
        event_price_std=("item_price_log", "std"),
        event_price_min=("item_price_log", "min"),
        event_price_max=("item_price_log", "max"),
        event_src_mean=("src_slot", "mean"),
        event_src_std=("src_slot", "std"),
        event_src_min=("src_slot", "min"),
        event_src_max=("src_slot", "max"),
    )
    features["event_active_span_hours"] = (
        features["event_oldest_hours"] - features["event_recency_hours"]
    )
    features["event_type_entropy"] = grouped["event_type"].apply(_entropy)
    features["event_ctx_entropy"] = grouped["ctx_seq"].apply(_entropy)
    features["event_src_entropy"] = grouped["src_slot"].apply(_entropy)

    last_events = grouped.tail(1).set_index("lead_id")
    features["last_event_type"] = last_events["event_type"].astype("string")
    features["last_event_ctx"] = last_events["ctx_seq"].astype("string")
    features["last_event_src"] = last_events["src_slot"].astype("string")
    features["last_event_price"] = last_events["item_price_log"]

    event_types = sorted(history["event_type"].dropna().unique())
    contexts = sorted(history["ctx_seq"].dropna().unique())
    total_count = features["event_count"]

    additions: dict[str, pd.Series] = {}
    for event_type in event_types:
        subset = history.loc[history["event_type"] == event_type]
        by_lead = subset.groupby("lead_id", sort=False)
        count = by_lead.size()
        additions[f"event_{event_type}_count"] = count
        additions[f"event_{event_type}_share"] = count / total_count
        additions[f"event_{event_type}_recency_hours"] = by_lead[
            "hours_before_assignment"
        ].min()
        additions[f"event_{event_type}_price_mean"] = by_lead["item_price_log"].mean()
        for window in EVENT_WINDOWS:
            window_count = (
                subset.loc[subset["hours_before_assignment"] <= 24 * window]
                .groupby("lead_id", sort=False)
                .size()
            )
            additions[f"event_{event_type}_{window}d"] = window_count

    for context in contexts:
        subset = history.loc[history["ctx_seq"] == context]
        count = subset.groupby("lead_id", sort=False).size()
        additions[f"event_ctx_{context}_count"] = count
        additions[f"event_ctx_{context}_share"] = count / total_count
        additions[f"event_ctx_{context}_recency_hours"] = subset.groupby(
            "lead_id", sort=False
        )["hours_before_assignment"].min()

    gaps = grouped["event_ts"].diff().dt.total_seconds().div(3600)
    gap_frame = pd.DataFrame({"lead_id": history["lead_id"], "gap": gaps})
    gap_grouped = gap_frame.groupby("lead_id", sort=False)["gap"]
    additions["event_gap_mean_hours"] = gap_grouped.mean()
    additions["event_gap_std_hours"] = gap_grouped.std()
    additions["event_gap_min_hours"] = gap_grouped.min()

    if advanced:
        for hours in EVENT_HOUR_WINDOWS:
            recent = history.loc[history["hours_before_assignment"] <= hours]
            additions[f"event_count_{hours}h"] = recent.groupby(
                "lead_id", sort=False
            ).size()
            for event_type in event_types:
                type_recent = recent.loc[recent["event_type"] == event_type]
                additions[f"event_{event_type}_{hours}h"] = type_recent.groupby(
                    "lead_id", sort=False
                ).size()

        for half_life in EVENT_HALF_LIVES:
            weights = np.exp(
                -np.log(2) * history["hours_before_assignment"] / half_life
            )
            additions[f"event_decay_{half_life}h"] = weights.groupby(
                history["lead_id"], sort=False
            ).sum()
            for event_type in event_types:
                mask = history["event_type"] == event_type
                additions[f"event_{event_type}_decay_{half_life}h"] = (
                    weights.loc[mask]
                    .groupby(history.loc[mask, "lead_id"], sort=False)
                    .sum()
                )
            for context in contexts:
                mask = history["ctx_seq"] == context
                additions[f"event_ctx_{context}_decay_{half_life}h"] = (
                    weights.loc[mask]
                    .groupby(history.loc[mask, "lead_id"], sort=False)
                    .sum()
                )

        cross = (
            history["event_type"].astype(str) + "__" + history["ctx_seq"].astype(str)
        )
        for value in sorted(cross.unique()):
            mask = cross == value
            cross_grouped = history.loc[mask].groupby("lead_id", sort=False)
            safe_value = value.replace("__", "_ctx_")
            additions[f"event_cross_{safe_value}_count"] = cross_grouped.size()
            additions[f"event_cross_{safe_value}_recency_hours"] = cross_grouped[
                "hours_before_assignment"
            ].min()

        reverse_position = history.groupby("lead_id", sort=False).cumcount(
            ascending=False
        )
        for lag in range(1, 6):
            lagged = history.loc[reverse_position == lag - 1].set_index("lead_id")
            features[f"event_lag{lag}_type"] = lagged["event_type"].astype("string")
            features[f"event_lag{lag}_ctx"] = lagged["ctx_seq"].astype("string")
            features[f"event_lag{lag}_src"] = lagged["src_slot"].astype("string")
            additions[f"event_lag{lag}_recency_hours"] = lagged[
                "hours_before_assignment"
            ]
            additions[f"event_lag{lag}_price"] = lagged["item_price_log"]

        previous_type = grouped["event_type"].shift()
        transitions = (
            previous_type.astype("string")
            + "__"
            + history["event_type"].astype("string")
        )
        for value in sorted(transitions.dropna().unique()):
            mask = transitions == value
            additions[f"event_transition_{value}_count"] = (
                history.loc[mask].groupby("lead_id", sort=False).size()
            )

        additions["event_type_change_count"] = (
            (previous_type.notna() & previous_type.ne(history["event_type"]))
            .groupby(history["lead_id"], sort=False)
            .sum()
        )
    features = features.join(pd.DataFrame(additions), how="left")
    return features


def _is_categorical(dtype: object) -> bool:
    return isinstance(
        dtype,
        (pd.StringDtype, pd.CategoricalDtype),
    ) or pd.api.types.is_object_dtype(dtype)


def combine_features(
    frame: pd.DataFrame,
    event_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine assignment features with optional lead-indexed event features."""
    result = build_tabular_features(frame)
    if event_features is None:
        return result

    aligned_events = event_features.reindex(frame["lead_id"].astype(str))
    aligned_events.index = result.index
    result = pd.concat([result, aligned_events], axis=1)
    result["event_price_minus_item"] = (
        result["event_price_mean"] - frame["item_price_log"].to_numpy()
    )
    result["last_event_price_minus_item"] = (
        result["last_event_price"] - frame["item_price_log"].to_numpy()
    )
    for column in result.columns:
        if _is_categorical(result[column].dtype):
            result[column] = (
                result[column].astype("string").fillna("__NO_EVENT__").astype(object)
            )
    return result


def categorical_feature_names(frame: pd.DataFrame) -> list[str]:
    """Return categorical columns in a prepared feature frame."""
    return [column for column in frame.columns if _is_categorical(frame[column].dtype)]


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.concat(
        [
            train[["lead_id", "assignment_ts"]],
            test[["lead_id", "assignment_ts"]],
        ],
        ignore_index=True,
    )
    event_features = build_event_features(events, assignments, advanced=True)
    train_features = combine_features(train, event_features)
    test_features = combine_features(test, event_features)

    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train and test feature schemas differ")

    return train_features, test_features

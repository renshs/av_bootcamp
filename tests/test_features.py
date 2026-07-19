import pandas as pd

from lead_priority.features import build_event_features


def test_event_features_exclude_events_at_or_after_assignment() -> None:
    assignments = pd.DataFrame(
        {
            "lead_id": ["lead_1"],
            "assignment_ts": ["2026-04-10 12:00:00"],
        }
    )
    events = pd.DataFrame(
        {
            "lead_id": ["lead_1", "lead_1", "lead_1"],
            "user_id": ["user_1", "user_1", "user_1"],
            "event_ts": [
                "2026-04-10 11:00:00",
                "2026-04-10 12:00:00",
                "2026-04-10 13:00:00",
            ],
            "event_type": ["item_view", "favorite", "call_click"],
            "item_price_log": [10.0, 10.0, 10.0],
            "src_slot": [1.0, 2.0, 3.0],
            "ctx_seq": ["c01", "c02", "c03"],
        }
    )

    features = build_event_features(events, assignments, advanced=True)

    assert features.loc["lead_1", "event_count"] == 1
    assert features.loc["lead_1", "event_item_view_count"] == 1
    assert "event_favorite_count" not in features
    assert "event_call_click_count" not in features
    assert features.loc["lead_1", "event_recency_hours"] == 1.0

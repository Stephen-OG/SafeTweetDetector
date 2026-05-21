from safetweet.policy import ThresholdConfig
from safetweet.storage import ModerationStore


def test_logs_prediction_and_queues_review_item(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()

    prediction_id = store.log_prediction(
        text="Potentially harmful text",
        predicted_label=0,
        predicted_label_name="Severe Harm",
        confidence=0.95,
        probabilities={"0": 0.95, "1": 0.03, "2": 0.01, "3": 0.01},
        action="block",
        reason="severe_harm_above_block_threshold",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)

    items = store.list_queue()

    assert queue_id == 1
    assert items[0]["id"] == 1
    assert items[0]["predicted_label"] == 0
    assert items[0]["status"] == "pending"


def test_review_updates_queue_item(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    prediction_id = store.log_prediction(
        text="Friendly text",
        predicted_label=3,
        predicted_label_name="Safe",
        confidence=0.92,
        probabilities={"0": 0.01, "1": 0.02, "2": 0.05, "3": 0.92},
        action="allow",
        reason="safe_above_allow_threshold",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)

    updated = store.review(queue_id, status="reviewed", reviewer_label=3, notes="Confirmed")

    assert updated["status"] == "reviewed"
    assert updated["reviewer_label"] == 3
    assert updated["notes"] == "Confirmed"


def test_list_queue_returns_pending_items_first(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    first_prediction_id = store.log_prediction(
        text="First review item",
        predicted_label=1,
        predicted_label_name="Non-violent Harm",
        confidence=0.72,
        probabilities={"0": 0.05, "1": 0.72, "2": 0.1, "3": 0.13},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    second_prediction_id = store.log_prediction(
        text="Second review item",
        predicted_label=2,
        predicted_label_name="Social / Contextual Harm",
        confidence=0.68,
        probabilities={"0": 0.04, "1": 0.12, "2": 0.68, "3": 0.16},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    reviewed_queue_id = store.enqueue(first_prediction_id)
    pending_queue_id = store.enqueue(second_prediction_id)
    store.review(reviewed_queue_id, status="reviewed", reviewer_label=1, notes="Handled")

    items = store.list_queue()

    assert items[0]["id"] == pending_queue_id
    assert items[0]["status"] == "pending"
    assert items[1]["id"] == reviewed_queue_id


def test_analytics_counts_predictions_and_queue(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    prediction_id = store.log_prediction(
        text="Potentially harmful text",
        predicted_label=1,
        predicted_label_name="Non-violent Harm",
        confidence=0.77,
        probabilities={"0": 0.05, "1": 0.77, "2": 0.1, "3": 0.08},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)
    store.review(queue_id, status="reviewed", reviewer_label=1, notes="Confirmed")

    analytics = store.analytics()

    assert analytics["total_predictions"] == 1
    assert analytics["queued_items"] == 1
    assert analytics["by_label"]["1"] == 1
    assert analytics["by_model_version"]["mock-v1"] == 1
    assert analytics["review_outcomes"]["reviewed"] == 1
    assert analytics["confidence_bands"]["0.60-0.80"] == 1


def test_threshold_settings_are_versioned(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()

    settings_id = store.save_thresholds(
        ThresholdConfig(severe_block_threshold=0.92),
        model_version="mock-v1",
    )
    latest = store.get_latest_thresholds()

    assert settings_id == 1
    assert latest["model_version"] == "mock-v1"
    assert latest["severe_block_threshold"] == 0.92

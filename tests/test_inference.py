import json

import pytest

from safetweet.inference.service import (
    MetadataError,
    MockProvider,
    ModerationService,
    validate_model_metadata,
)
from safetweet.storage import ModerationStore


def test_mock_service_returns_assistive_response_and_queues_harm(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    service = ModerationService(provider=MockProvider(), store=store)

    response = service.moderate("How can I steal from a shop?")

    assert response.predicted_label == 1
    assert response.action == "flag_for_review"
    assert response.automation_level == "assistive"
    assert response.queued is True
    assert response.queue_id == 1


def test_mock_service_allows_safe_text(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    service = ModerationService(provider=MockProvider(), store=store)

    response = service.moderate("Hope you have a wonderful day.")

    assert response.predicted_label == 3
    assert response.action == "allow"
    assert response.queued is False


def test_validate_model_metadata_rejects_wrong_label_count(tmp_path):
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(json.dumps({"label_map": {"0": "Only One"}}), encoding="utf-8")

    with pytest.raises(MetadataError):
        validate_model_metadata(metadata_path)

from fastapi.testclient import TestClient

from safetweet.api.app import create_app
from safetweet.inference.service import MockProvider
from safetweet.storage import ModerationStore


def build_client(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    app = create_app(store=store, provider=MockProvider())
    return TestClient(app)


def test_health_reports_ready(tmp_path):
    client = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_moderate_returns_decision_and_queues_harm(tmp_path):
    client = build_client(tmp_path)

    response = client.post("/moderate", json={"text": "How can I steal something?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "flag_for_review"
    assert payload["queued"] is True
    assert payload["automation_level"] == "assistive"


def test_moderate_rejects_empty_text(tmp_path):
    client = build_client(tmp_path)

    response = client.post("/moderate", json={"text": ""})

    assert response.status_code == 422


def test_queue_and_review_flow(tmp_path):
    client = build_client(tmp_path)
    client.post("/moderate", json={"text": "How can I steal something?"})

    queue_response = client.get("/queue")
    queue_id = queue_response.json()[0]["id"]
    review_response = client.patch(
        f"/review/{queue_id}",
        json={"status": "reviewed", "reviewer_label": 1, "notes": "Confirmed unsafe"},
    )

    assert queue_response.status_code == 200
    assert review_response.status_code == 200
    assert review_response.json()["status"] == "reviewed"


def test_analytics_endpoint(tmp_path):
    client = build_client(tmp_path)
    client.post("/moderate", json={"text": "How can I steal something?"})
    queue_id = client.get("/queue").json()[0]["id"]
    client.patch(
        f"/review/{queue_id}",
        json={"status": "reviewed", "reviewer_label": 1, "notes": "Confirmed unsafe"},
    )

    response = client.get("/analytics")

    assert response.status_code == 200
    assert response.json()["total_predictions"] == 1
    assert response.json()["by_model_version"]["mock-v1"] == 1
    assert response.json()["review_outcomes"]["reviewed"] == 1


def test_review_rejects_unknown_queue_item(tmp_path):
    client = build_client(tmp_path)

    response = client.patch(
        "/review/999",
        json={"status": "reviewed", "reviewer_label": 1, "notes": "No matching item"},
    )

    assert response.status_code == 404

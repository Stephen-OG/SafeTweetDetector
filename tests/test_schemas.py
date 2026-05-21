import pytest
from pydantic import ValidationError

from safetweet.schemas import ModerateRequest, ReviewRequest


def test_moderate_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        ModerateRequest(text="")


def test_moderate_request_accepts_normal_text():
    request = ModerateRequest(text="A short post to check.")

    assert request.text == "A short post to check."


def test_review_request_accepts_known_status():
    request = ReviewRequest(status="reviewed", reviewer_label=3, notes="Looks safe")

    assert request.status == "reviewed"


def test_review_request_rejects_unknown_label():
    with pytest.raises(ValidationError):
        ReviewRequest(status="reviewed", reviewer_label=7)

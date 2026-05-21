from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class ModerateResponse(BaseModel):
    text: str
    predicted_label: int
    predicted_label_name: str
    confidence: float
    probabilities: dict[str, float]
    action: Literal["allow", "flag_for_review", "block"]
    reason: str
    automation_level: Literal["assistive"] = "assistive"
    queued: bool
    queue_id: int | None = None
    warnings: list[str] = Field(default_factory=list)
    model_version: str


class ReviewRequest(BaseModel):
    status: Literal["reviewed", "dismissed"]
    reviewer_label: int | None = Field(default=None, ge=0, le=3)
    notes: str = ""


class QueueItem(BaseModel):
    id: int
    text: str
    predicted_label: int
    predicted_label_name: str
    confidence: float
    action: str
    reason: str
    status: str
    reviewer_label: int | None = None
    notes: str = ""
    created_at: str
    reviewed_at: str | None = None

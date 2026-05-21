from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    severe_block_threshold: float = 0.90
    harmful_review_threshold: float = 0.60
    safe_allow_threshold: float = 0.85
    low_confidence_threshold: float = 0.60


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str


def decide(label_id: int, confidence: float, thresholds: ThresholdConfig) -> PolicyDecision:
    if confidence < thresholds.low_confidence_threshold:
        return PolicyDecision("flag_for_review", "low_confidence")
    if label_id == 0 and confidence >= thresholds.severe_block_threshold:
        return PolicyDecision("block", "severe_harm_above_block_threshold")
    if label_id in {0, 1, 2} and confidence >= thresholds.harmful_review_threshold:
        return PolicyDecision("flag_for_review", "harmful_content_requires_review")
    if label_id == 3 and confidence >= thresholds.safe_allow_threshold:
        return PolicyDecision("allow", "safe_above_allow_threshold")
    return PolicyDecision("flag_for_review", "ambiguous_prediction")

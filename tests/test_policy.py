from safetweet.policy import ThresholdConfig, decide


def test_blocks_high_confidence_severe_harm():
    result = decide(label_id=0, confidence=0.96, thresholds=ThresholdConfig())

    assert result.action == "block"
    assert result.reason == "severe_harm_above_block_threshold"


def test_flags_medium_confidence_harmful_prediction():
    result = decide(label_id=1, confidence=0.74, thresholds=ThresholdConfig())

    assert result.action == "flag_for_review"
    assert result.reason == "harmful_content_requires_review"


def test_allows_safe_high_confidence_prediction():
    result = decide(label_id=3, confidence=0.94, thresholds=ThresholdConfig())

    assert result.action == "allow"
    assert result.reason == "safe_above_allow_threshold"


def test_reviews_low_confidence_prediction():
    result = decide(label_id=3, confidence=0.51, thresholds=ThresholdConfig())

    assert result.action == "flag_for_review"
    assert result.reason == "low_confidence"

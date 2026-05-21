from __future__ import annotations

LABELS: dict[int, str] = {
    0: "Severe Harm",
    1: "Non-violent Harm",
    2: "Social / Contextual Harm",
    3: "Safe",
}

SEVERE_CATEGORIES = frozenset(
    {
        "violence,aiding_and_abetting,incitement",
        "terrorism,organized_crime",
        "self_harm",
        "child_abuse",
        "sexually_explicit,adult_content",
    }
)

NON_VIOLENT_CATEGORIES = frozenset(
    {
        "non_violent_unethical_behavior",
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation",
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "animal_abuse",
    }
)

SOCIAL_CONTEXTUAL_CATEGORIES = frozenset(
    {
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
        "controversial_topics,politics",
    }
)


def label_from_categories(categories: dict[str, bool]) -> int:
    """Map BeaverTails category flags into the project's 4-class hierarchy."""
    if any(categories.get(category, False) for category in SEVERE_CATEGORIES):
        return 0
    if any(categories.get(category, False) for category in NON_VIOLENT_CATEGORIES):
        return 1
    if any(categories.get(category, False) for category in SOCIAL_CONTEXTUAL_CATEGORIES):
        return 2
    return 3


def label_name(label_id: int) -> str:
    try:
        return LABELS[label_id]
    except KeyError as exc:
        raise ValueError(f"Unknown label id: {label_id}") from exc

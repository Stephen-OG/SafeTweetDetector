from safetweet.labels import LABELS, label_from_categories, label_name


def test_label_names_match_project_scheme():
    assert LABELS == {
        0: "Severe Harm",
        1: "Non-violent Harm",
        2: "Social / Contextual Harm",
        3: "Safe",
    }


def test_severe_harm_has_priority_over_other_harm():
    categories = {
        "violence,aiding_and_abetting,incitement": True,
        "financial_crime,property_crime,theft": True,
        "hate_speech,offensive_language": True,
    }

    assert label_from_categories(categories) == 0


def test_non_violent_harm_is_second_priority():
    categories = {
        "financial_crime,property_crime,theft": True,
        "hate_speech,offensive_language": True,
    }

    assert label_from_categories(categories) == 1


def test_social_contextual_harm_is_third_priority():
    categories = {
        "controversial_topics,politics": True,
        "hate_speech,offensive_language": True,
    }

    assert label_from_categories(categories) == 2


def test_safe_label_when_no_category_is_true():
    categories = {
        "violence,aiding_and_abetting,incitement": False,
        "financial_crime,property_crime,theft": False,
        "hate_speech,offensive_language": False,
    }

    assert label_from_categories(categories) == 3
    assert label_name(3) == "Safe"

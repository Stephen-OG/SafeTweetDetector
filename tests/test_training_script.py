from safetweet.training.train_transformer import build_model_metadata, parse_args


def test_parse_args_defaults_to_distilroberta():
    args = parse_args([])

    assert args.model_name == "distilroberta-base"
    assert args.max_length == 256


def test_build_model_metadata_contains_expected_label_map():
    metadata = build_model_metadata(
        model_name="distilroberta-base",
        model_version="distilroberta-base-local-test",
    )

    assert metadata["model_name"] == "distilroberta-base"
    assert metadata["model_version"] == "distilroberta-base-local-test"
    assert metadata["label_map"]["0"] == "Severe Harm"
    assert metadata["label_map"]["3"] == "Safe"

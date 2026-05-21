import gzip
import json
import lzma
from pathlib import Path

import pytest


@pytest.fixture
def sample_rows():
    return [
        {
            "prompt": "How can I steal money?",
            "response": "I cannot help with theft.",
            "is_safe": False,
            "category": {
                "financial_crime,property_crime,theft": True,
                "violence,aiding_and_abetting,incitement": False,
            },
        },
        {
            "prompt": "Tell me a friendly greeting.",
            "response": "Hello, hope your day is going well.",
            "is_safe": True,
            "category": {
                "financial_crime,property_crime,theft": False,
                "violence,aiding_and_abetting,incitement": False,
            },
        },
    ]


@pytest.fixture
def compressed_dataset_files(tmp_path: Path, sample_rows):
    train_xz = tmp_path / "train.jsonl.xz"
    test_gz = tmp_path / "test.jsonl.gz"

    with lzma.open(train_xz, "wt", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row) + "\n")

    with gzip.open(test_gz, "wt", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row) + "\n")

    return train_xz, test_gz

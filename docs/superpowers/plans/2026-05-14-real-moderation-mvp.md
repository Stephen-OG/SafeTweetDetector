# Real Moderation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local production-style moderation MVP with transformer-ready data loading, thresholded inference, SQLite-backed review queues, FastAPI endpoints, analytics, and training/export entry points.

**Architecture:** Create a `src/safetweet` Python package around the existing notebook work. The first runnable API uses a deterministic mock inference backend for fast tests and local development, while the inference interface also supports loading a saved Hugging Face sequence classification model. Storage, thresholds, review workflow, and analytics are implemented independently so the real transformer can be plugged in without redesigning the app.

**Tech Stack:** Python 3.10+, pytest, FastAPI, Pydantic, SQLite, Hugging Face datasets, transformers, PyTorch, scikit-learn.

---

## File Structure

- Create: `pyproject.toml`  
  Project metadata, runtime dependencies, dev dependencies, pytest configuration, and console scripts.

- Create: `src/safetweet/__init__.py`  
  Package marker and version.

- Create: `src/safetweet/labels.py`  
  Canonical 4-label definitions and BeaverTails category mapping.

- Create: `src/safetweet/data/__init__.py`  
  Data package exports.

- Create: `src/safetweet/data/beavertails.py`  
  Compressed JSONL loading, text construction, label mapping, and Hugging Face dataset conversion.

- Create: `src/safetweet/policy.py`  
  Threshold configuration and decision policy.

- Create: `src/safetweet/schemas.py`  
  Shared Pydantic request and response models.

- Create: `src/safetweet/storage.py`  
  SQLite schema, prediction logging, queue operations, review updates, and analytics queries.

- Create: `src/safetweet/inference/__init__.py`  
  Inference package exports.

- Create: `src/safetweet/inference/service.py`  
  Model metadata validation, provider interface, mock provider, Hugging Face provider, and moderation orchestration.

- Create: `src/safetweet/api/__init__.py`  
  API package marker.

- Create: `src/safetweet/api/app.py`  
  FastAPI app factory and endpoints.

- Create: `src/safetweet/training/__init__.py`  
  Training package marker.

- Create: `src/safetweet/training/train_transformer.py`  
  Transformer fine-tuning/export script with smoke-test-friendly sample limits.

- Create: `tests/conftest.py`  
  Shared fixtures for temporary databases and sample BeaverTails rows.

- Create: `tests/test_labels.py`  
  Label mapping tests.

- Create: `tests/test_data.py`  
  Compressed local JSONL loading tests.

- Create: `tests/test_policy.py`  
  Threshold decision tests.

- Create: `tests/test_schemas.py`  
  Pydantic validation tests for API request and response models.

- Create: `tests/test_storage.py`  
  SQLite logging, threshold setting, queue, review, and analytics tests.

- Create: `tests/test_inference.py`  
  Mock inference and metadata validation tests.

- Create: `tests/test_api.py`  
  FastAPI endpoint tests.

- Create: `tests/test_training_script.py`  
  Transformer training CLI helper tests.

- Modify: `README.md`  
  Add local setup, test, API, and training usage for the MVP.

## Task 1: Project Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/safetweet/__init__.py`

- [ ] **Step 1: Create packaging and dependency configuration**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "safe-tweet-detector"
version = "0.1.0"
description = "Production-style moderation MVP for 4-class harm detection."
requires-python = ">=3.10"
dependencies = [
  "datasets>=2.19",
  "fastapi>=0.111",
  "httpx>=0.27",
  "numpy>=1.26",
  "pydantic>=2.7",
  "scikit-learn>=1.4",
  "torch>=2.2",
  "transformers>=4.41",
  "uvicorn[standard]>=0.30"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0"
]

[project.scripts]
safetweet-api = "safetweet.api.app:main"
safetweet-train-transformer = "safetweet.training.train_transformer:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

- [ ] **Step 2: Create the package marker**

Create `src/safetweet/__init__.py` with:

```python
"""Safe Tweet Detector production-style moderation MVP."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Install the project in editable mode**

Run:

```bash
python3 -m pip install -e ".[dev]"
```

Expected: command exits with status `0` and installs `safetweet`.

- [ ] **Step 4: Run an empty test discovery check**

Run:

```bash
python3 -m pytest
```

Expected: pytest reports no tests collected before test files exist; exit code may be `5` at this point.

- [ ] **Step 5: Commit the package skeleton**

Run:

```bash
git add pyproject.toml src/safetweet/__init__.py
git commit -m "chore: add package skeleton"
```

## Task 2: Canonical Labels And BeaverTails Mapping

**Files:**
- Create: `src/safetweet/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write failing label mapping tests**

Create `tests/test_labels.py` with:

```python
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
```

- [ ] **Step 2: Run the label tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_labels.py -q
```

Expected: fails with `ModuleNotFoundError` or missing functions from `safetweet.labels`.

- [ ] **Step 3: Implement label mapping**

Create `src/safetweet/labels.py` with:

```python
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
```

- [ ] **Step 4: Run the label tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_labels.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit labels**

Run:

```bash
git add src/safetweet/labels.py tests/test_labels.py
git commit -m "feat: add beavertails label mapping"
```

## Task 3: Local BeaverTails Data Loading

**Files:**
- Create: `src/safetweet/data/__init__.py`
- Create: `src/safetweet/data/beavertails.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write fixtures and failing data loader tests**

Create `tests/conftest.py` with:

```python
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
```

Create `tests/test_data.py` with:

```python
from safetweet.data.beavertails import build_text, iter_jsonl, load_records


def test_iter_jsonl_reads_xz_and_gz(compressed_dataset_files):
    train_xz, test_gz = compressed_dataset_files

    assert len(list(iter_jsonl(train_xz))) == 2
    assert len(list(iter_jsonl(test_gz))) == 2


def test_build_text_joins_prompt_and_response():
    row = {"prompt": "Prompt", "response": "Response"}

    assert build_text(row) == "Prompt\n\nResponse"


def test_load_records_adds_text_and_label(compressed_dataset_files):
    train_xz, _ = compressed_dataset_files

    records = load_records(train_xz)

    assert records[0]["text"] == "How can I steal money?\n\nI cannot help with theft."
    assert records[0]["label"] == 1
    assert records[1]["label"] == 3
```

- [ ] **Step 2: Run data tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_data.py -q
```

Expected: fails because `safetweet.data.beavertails` does not exist.

- [ ] **Step 3: Implement compressed JSONL loading**

Create `src/safetweet/data/__init__.py` with:

```python
"""Data loading helpers."""
```

Create `src/safetweet/data/beavertails.py` with:

```python
from __future__ import annotations

import gzip
import json
import lzma
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset

from safetweet.labels import label_from_categories


def _open_text(path: Path):
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".jsonl.xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    if suffixes.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    with _open_text(resolved) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_text(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    return f"{prompt}\n\n{response}".strip()


def load_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path)):
        if limit is not None and index >= limit:
            break
        records.append(
            {
                "text": build_text(row),
                "label": label_from_categories(row.get("category", {})),
                "is_safe": bool(row.get("is_safe", False)),
                "category": row.get("category", {}),
            }
        )
    return records


def load_dataset_from_jsonl(path: str | Path, limit: int | None = None) -> Dataset:
    return Dataset.from_list(load_records(path, limit=limit))
```

- [ ] **Step 4: Run data tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_data.py tests/test_labels.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit data loading**

Run:

```bash
git add src/safetweet/data tests/conftest.py tests/test_data.py
git commit -m "feat: load local beavertails data"
```

## Task 4: Threshold Decision Policy

**Files:**
- Create: `src/safetweet/policy.py`
- Create: `tests/test_policy.py`

- [ ] **Step 1: Write failing threshold policy tests**

Create `tests/test_policy.py` with:

```python
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
```

- [ ] **Step 2: Run policy tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_policy.py -q
```

Expected: fails because `safetweet.policy` does not exist.

- [ ] **Step 3: Implement threshold policy**

Create `src/safetweet/policy.py` with:

```python
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
```

- [ ] **Step 4: Run policy tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit threshold policy**

Run:

```bash
git add src/safetweet/policy.py tests/test_policy.py
git commit -m "feat: add moderation threshold policy"
```

## Task 5: Shared Schemas

**Files:**
- Create: `src/safetweet/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema validation tests**

Create `tests/test_schemas.py` with:

```python
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
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_schemas.py -q
```

Expected: fails because `safetweet.schemas` does not exist.

- [ ] **Step 3: Implement Pydantic schemas**

Create `src/safetweet/schemas.py` with:

```python
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
```

- [ ] **Step 4: Run schema tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_schemas.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit schemas**

Run:

```bash
git add src/safetweet/schemas.py tests/test_schemas.py
git commit -m "feat: add moderation schemas"
```

## Task 6: SQLite Storage And Analytics

**Files:**
- Create: `src/safetweet/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_storage.py` with:

```python
from safetweet.policy import ThresholdConfig
from safetweet.storage import ModerationStore


def test_logs_prediction_and_queues_review_item(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()

    prediction_id = store.log_prediction(
        text="Potentially harmful text",
        predicted_label=0,
        predicted_label_name="Severe Harm",
        confidence=0.95,
        probabilities={"0": 0.95, "1": 0.03, "2": 0.01, "3": 0.01},
        action="block",
        reason="severe_harm_above_block_threshold",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)

    items = store.list_queue()

    assert queue_id == 1
    assert items[0]["id"] == 1
    assert items[0]["predicted_label"] == 0
    assert items[0]["status"] == "pending"


def test_review_updates_queue_item(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    prediction_id = store.log_prediction(
        text="Friendly text",
        predicted_label=3,
        predicted_label_name="Safe",
        confidence=0.92,
        probabilities={"0": 0.01, "1": 0.02, "2": 0.05, "3": 0.92},
        action="allow",
        reason="safe_above_allow_threshold",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)

    updated = store.review(queue_id, status="reviewed", reviewer_label=3, notes="Confirmed")

    assert updated["status"] == "reviewed"
    assert updated["reviewer_label"] == 3
    assert updated["notes"] == "Confirmed"


def test_list_queue_returns_pending_items_first(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    first_prediction_id = store.log_prediction(
        text="First review item",
        predicted_label=1,
        predicted_label_name="Non-violent Harm",
        confidence=0.72,
        probabilities={"0": 0.05, "1": 0.72, "2": 0.1, "3": 0.13},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    second_prediction_id = store.log_prediction(
        text="Second review item",
        predicted_label=2,
        predicted_label_name="Social / Contextual Harm",
        confidence=0.68,
        probabilities={"0": 0.04, "1": 0.12, "2": 0.68, "3": 0.16},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    reviewed_queue_id = store.enqueue(first_prediction_id)
    pending_queue_id = store.enqueue(second_prediction_id)
    store.review(reviewed_queue_id, status="reviewed", reviewer_label=1, notes="Handled")

    items = store.list_queue()

    assert items[0]["id"] == pending_queue_id
    assert items[0]["status"] == "pending"
    assert items[1]["id"] == reviewed_queue_id


def test_analytics_counts_predictions_and_queue(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()
    prediction_id = store.log_prediction(
        text="Potentially harmful text",
        predicted_label=1,
        predicted_label_name="Non-violent Harm",
        confidence=0.77,
        probabilities={"0": 0.05, "1": 0.77, "2": 0.1, "3": 0.08},
        action="flag_for_review",
        reason="harmful_content_requires_review",
        model_version="mock-v1",
    )
    queue_id = store.enqueue(prediction_id)
    store.review(queue_id, status="reviewed", reviewer_label=1, notes="Confirmed")

    analytics = store.analytics()

    assert analytics["total_predictions"] == 1
    assert analytics["queued_items"] == 1
    assert analytics["by_label"]["1"] == 1
    assert analytics["by_model_version"]["mock-v1"] == 1
    assert analytics["review_outcomes"]["reviewed"] == 1
    assert analytics["confidence_bands"]["0.60-0.80"] == 1


def test_threshold_settings_are_versioned(tmp_path):
    store = ModerationStore(tmp_path / "moderation.db")
    store.initialize()

    settings_id = store.save_thresholds(
        ThresholdConfig(severe_block_threshold=0.92),
        model_version="mock-v1",
    )
    latest = store.get_latest_thresholds()

    assert settings_id == 1
    assert latest["model_version"] == "mock-v1"
    assert latest["severe_block_threshold"] == 0.92
```

- [ ] **Step 2: Run storage tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_storage.py -q
```

Expected: fails because `safetweet.storage` does not exist.

- [ ] **Step 3: Implement SQLite store**

Create `src/safetweet/storage.py` with:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ModerationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    predicted_label INTEGER NOT NULL,
                    predicted_label_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
                    status TEXT NOT NULL DEFAULT 'pending',
                    reviewer_label INTEGER,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS threshold_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    severe_block_threshold REAL NOT NULL,
                    harmful_review_threshold REAL NOT NULL,
                    safe_allow_threshold REAL NOT NULL,
                    low_confidence_threshold REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def log_prediction(
        self,
        *,
        text: str,
        predicted_label: int,
        predicted_label_name: str,
        confidence: float,
        probabilities: dict[str, float],
        action: str,
        reason: str,
        model_version: str,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    text, predicted_label, predicted_label_name, confidence,
                    probabilities_json, action, reason, model_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text,
                    predicted_label,
                    predicted_label_name,
                    confidence,
                    json.dumps(probabilities, sort_keys=True),
                    action,
                    reason,
                    model_version,
                ),
            )
            return int(cursor.lastrowid)

    def enqueue(self, prediction_id: int) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO queue_items (prediction_id) VALUES (?)",
                (prediction_id,),
            )
            return int(cursor.lastrowid)

    def list_queue(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    q.id, p.text, p.predicted_label, p.predicted_label_name,
                    p.confidence, p.action, p.reason, q.status,
                    q.reviewer_label, q.notes, q.created_at, q.reviewed_at
                FROM queue_items q
                JOIN predictions p ON p.id = q.prediction_id
                ORDER BY
                    CASE WHEN q.status = 'pending' THEN 0 ELSE 1 END,
                    q.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def review(
        self,
        queue_id: int,
        *,
        status: str,
        reviewer_label: int | None,
        notes: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE queue_items
                SET status = ?, reviewer_label = ?, notes = ?, reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, reviewer_label, notes, queue_id),
            )
            row = connection.execute(
                """
                SELECT
                    q.id, p.text, p.predicted_label, p.predicted_label_name,
                    p.confidence, p.action, p.reason, q.status,
                    q.reviewer_label, q.notes, q.created_at, q.reviewed_at
                FROM queue_items q
                JOIN predictions p ON p.id = q.prediction_id
                WHERE q.id = ?
                """,
                (queue_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Queue item not found: {queue_id}")
        return dict(row)

    def save_thresholds(self, thresholds, *, model_version: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO threshold_settings (
                    model_version, severe_block_threshold, harmful_review_threshold,
                    safe_allow_threshold, low_confidence_threshold
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_version,
                    thresholds.severe_block_threshold,
                    thresholds.harmful_review_threshold,
                    thresholds.safe_allow_threshold,
                    thresholds.low_confidence_threshold,
                ),
            )
            return int(cursor.lastrowid)

    def get_latest_thresholds(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id, model_version, severe_block_threshold, harmful_review_threshold,
                    safe_allow_threshold, low_confidence_threshold, created_at
                FROM threshold_settings
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def analytics(self) -> dict[str, Any]:
        with self.connect() as connection:
            total_predictions = connection.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
            queued_items = connection.execute(
                "SELECT COUNT(*) FROM queue_items"
            ).fetchone()[0]
            label_rows = connection.execute(
                "SELECT predicted_label, COUNT(*) AS count FROM predictions GROUP BY predicted_label"
            ).fetchall()
            action_rows = connection.execute(
                "SELECT action, COUNT(*) AS count FROM predictions GROUP BY action"
            ).fetchall()
            model_version_rows = connection.execute(
                "SELECT model_version, COUNT(*) AS count FROM predictions GROUP BY model_version"
            ).fetchall()
            review_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM queue_items GROUP BY status"
            ).fetchall()
            confidence_rows = connection.execute(
                """
                SELECT
                    CASE
                        WHEN confidence < 0.60 THEN '0.00-0.60'
                        WHEN confidence < 0.80 THEN '0.60-0.80'
                        WHEN confidence < 0.90 THEN '0.80-0.90'
                        ELSE '0.90-1.00'
                    END AS band,
                    COUNT(*) AS count
                FROM predictions
                GROUP BY band
                """
            ).fetchall()

        return {
            "total_predictions": total_predictions,
            "queued_items": queued_items,
            "by_label": {str(row["predicted_label"]): row["count"] for row in label_rows},
            "by_action": {row["action"]: row["count"] for row in action_rows},
            "by_model_version": {
                row["model_version"]: row["count"] for row in model_version_rows
            },
            "review_outcomes": {row["status"]: row["count"] for row in review_rows},
            "confidence_bands": {row["band"]: row["count"] for row in confidence_rows},
        }
```

- [ ] **Step 4: Run storage tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_storage.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit storage**

Run:

```bash
git add src/safetweet/storage.py tests/test_storage.py
git commit -m "feat: add sqlite moderation storage"
```

## Task 7: Inference Service With Mock And Transformer Providers

**Files:**
- Create: `src/safetweet/inference/__init__.py`
- Create: `src/safetweet/inference/service.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing inference tests**

Create `tests/test_inference.py` with:

```python
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
```

- [ ] **Step 2: Run inference tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_inference.py -q
```

Expected: fails because `safetweet.inference.service` does not exist.

- [ ] **Step 3: Implement inference service**

Create `src/safetweet/inference/__init__.py` with:

```python
"""Inference providers and moderation service."""
```

Create `src/safetweet/inference/service.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from safetweet.labels import LABELS, label_name
from safetweet.policy import ThresholdConfig, decide
from safetweet.schemas import ModerateResponse
from safetweet.storage import ModerationStore


class MetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    probabilities: dict[str, float]
    model_version: str

    @property
    def label_id(self) -> int:
        return int(max(self.probabilities, key=self.probabilities.get))

    @property
    def confidence(self) -> float:
        return float(self.probabilities[str(self.label_id)])


class PredictionProvider(Protocol):
    def predict(self, text: str) -> Prediction:
        pass


class MockProvider:
    model_version = "mock-v1"

    def predict(self, text: str) -> Prediction:
        lowered = text.lower()
        if any(term in lowered for term in ("steal", "bomb", "kill", "weapon")):
            probabilities = {"0": 0.05, "1": 0.78, "2": 0.08, "3": 0.09}
        else:
            probabilities = {"0": 0.01, "1": 0.02, "2": 0.03, "3": 0.94}
        return Prediction(probabilities=probabilities, model_version=self.model_version)


class HuggingFaceProvider:
    def __init__(self, model_dir: str | Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.model_dir = Path(model_dir)
        validate_model_metadata(self.model_dir / "model_metadata.json")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.torch = torch
        metadata = json.loads((self.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.model_version = metadata["model_version"]

    def predict(self, text: str) -> Prediction:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        probabilities_array = self.torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        probabilities = {
            str(index): float(probability)
            for index, probability in enumerate(np.asarray(probabilities_array))
        }
        return Prediction(probabilities=probabilities, model_version=self.model_version)


def validate_model_metadata(metadata_path: str | Path) -> dict[str, object]:
    path = Path(metadata_path)
    if not path.exists():
        raise MetadataError(f"Missing model metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    label_map = metadata.get("label_map")
    expected = {str(key): value for key, value in LABELS.items()}
    if label_map != expected:
        raise MetadataError("Model metadata label_map does not match SafeTweet labels")
    if not metadata.get("model_version"):
        raise MetadataError("Model metadata is missing model_version")
    return metadata


class ModerationService:
    def __init__(
        self,
        *,
        provider: PredictionProvider,
        store: ModerationStore,
        thresholds: ThresholdConfig | None = None,
    ):
        self.provider = provider
        self.store = store
        self.thresholds = thresholds or ThresholdConfig()

    def moderate(self, text: str) -> ModerateResponse:
        prediction = self.provider.predict(text)
        decision = decide(prediction.label_id, prediction.confidence, self.thresholds)
        predicted_label_name = label_name(prediction.label_id)
        warnings: list[str] = []
        queue_id: int | None = None
        queued = decision.action in {"flag_for_review", "block"}

        try:
            prediction_id = self.store.log_prediction(
                text=text,
                predicted_label=prediction.label_id,
                predicted_label_name=predicted_label_name,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                action=decision.action,
                reason=decision.reason,
                model_version=prediction.model_version,
            )
            if queued:
                queue_id = self.store.enqueue(prediction_id)
        except Exception as exc:
            warnings.append(f"logging_failed: {exc}")

        return ModerateResponse(
            text=text,
            predicted_label=prediction.label_id,
            predicted_label_name=predicted_label_name,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            action=decision.action,
            reason=decision.reason,
            queued=queued and queue_id is not None,
            queue_id=queue_id,
            warnings=warnings,
            model_version=prediction.model_version,
        )
```

- [ ] **Step 4: Run inference tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_inference.py tests/test_policy.py tests/test_storage.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit inference service**

Run:

```bash
git add src/safetweet/inference tests/test_inference.py
git commit -m "feat: add moderation inference service"
```

## Task 8: FastAPI Endpoints

**Files:**
- Create: `src/safetweet/api/__init__.py`
- Create: `src/safetweet/api/app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_api.py` with:

```python
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
```

- [ ] **Step 2: Run API tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_api.py -q
```

Expected: fails because `safetweet.api.app` does not exist.

- [ ] **Step 3: Implement FastAPI app**

Create `src/safetweet/api/__init__.py` with:

```python
"""FastAPI application package."""
```

Create `src/safetweet/api/app.py` with:

```python
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from safetweet.inference.service import HuggingFaceProvider, MockProvider, ModerationService
from safetweet.schemas import ModerateRequest, ModerateResponse, QueueItem, ReviewRequest
from safetweet.storage import ModerationStore


def create_app(
    *,
    store: ModerationStore | None = None,
    provider=None,
) -> FastAPI:
    db_path = Path(os.getenv("SAFETWEET_DB_PATH", "var/moderation.db"))
    model_dir = os.getenv("SAFETWEET_MODEL_DIR")
    resolved_store = store or ModerationStore(db_path)
    resolved_store.initialize()
    resolved_provider = provider or (
        HuggingFaceProvider(model_dir) if model_dir else MockProvider()
    )
    service = ModerationService(provider=resolved_provider, store=resolved_store)

    app = FastAPI(title="Safe Tweet Detector", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "model_version": resolved_provider.model_version}

    @app.post("/moderate", response_model=ModerateResponse)
    def moderate(request: ModerateRequest):
        return service.moderate(request.text)

    @app.get("/queue", response_model=list[QueueItem])
    def queue():
        return resolved_store.list_queue()

    @app.patch("/review/{item_id}", response_model=QueueItem)
    def review(item_id: int, request: ReviewRequest):
        try:
            return resolved_store.review(
                item_id,
                status=request.status,
                reviewer_label=request.reviewer_label,
                notes=request.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/analytics")
    def analytics():
        return resolved_store.analytics()

    return app


app = create_app()


def main() -> None:
    uvicorn.run("safetweet.api.app:app", host="127.0.0.1", port=8000, reload=True)
```

- [ ] **Step 4: Run API tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_api.py tests/test_inference.py tests/test_storage.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Smoke test the local API manually**

Run:

```bash
SAFETWEET_DB_PATH=var/dev-moderation.db python3 -m uvicorn safetweet.api.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, run:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text":"How can I steal something?"}'
```

Expected: `/health` returns `{"status":"ready","model_version":"mock-v1"}` and `/moderate` returns `flag_for_review`.

- [ ] **Step 6: Commit API endpoints**

Run:

```bash
git add src/safetweet/api tests/test_api.py
git commit -m "feat: add moderation api endpoints"
```

## Task 9: Transformer Training And Export Script

**Files:**
- Create: `src/safetweet/training/__init__.py`
- Create: `src/safetweet/training/train_transformer.py`
- Create: `tests/test_training_script.py`

- [ ] **Step 1: Write failing training helper tests**

Create `tests/test_training_script.py` with:

```python
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
```

- [ ] **Step 2: Run training helper tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_training_script.py -q
```

Expected: fails because `safetweet.training.train_transformer` does not exist.

- [ ] **Step 3: Implement transformer training script**

Create `src/safetweet/training/__init__.py` with:

```python
"""Training entry points."""
```

Create `src/safetweet/training/train_transformer.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from safetweet.data.beavertails import load_dataset_from_jsonl
from safetweet.labels import LABELS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer moderation model.")
    parser.add_argument("--train-path", default="dataset/train.jsonl.xz")
    parser.add_argument("--test-path", default="dataset/test.jsonl.xz")
    parser.add_argument("--output-dir", default="models/transformer")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--model-version", default="distilroberta-base-local")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def build_model_metadata(*, model_name: str, model_version: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "label_map": {str(key): value for key, value in LABELS.items()},
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = load_dataset_from_jsonl(args.train_path, limit=args.train_limit)
    eval_dataset = load_dataset_from_jsonl(args.test_path, limit=args.eval_limit)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "model_metadata.json").write_text(
        json.dumps(
            build_model_metadata(model_name=args.model_name, model_version=args.model_version),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (metrics_dir / "transformer_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run training helper tests and verify pass**

Run:

```bash
python3 -m pytest tests/test_training_script.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run a tiny training smoke command**

Run:

```bash
python3 -m safetweet.training.train_transformer \
  --train-limit 8 \
  --eval-limit 8 \
  --epochs 0.01 \
  --batch-size 2 \
  --output-dir models/transformer-smoke \
  --metrics-dir reports/metrics-smoke \
  --model-version smoke-test
```

Expected: command exits with status `0`, writes `models/transformer-smoke/model_metadata.json`, and writes `reports/metrics-smoke/transformer_metrics.json`.

- [ ] **Step 6: Commit training script**

Run:

```bash
git add src/safetweet/training tests/test_training_script.py
git commit -m "feat: add transformer training entry point"
```

## Task 10: README Usage Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add MVP setup and usage sections**

Append this section to `README.md` before the licence section:

````markdown
## Real Moderation MVP

The production-style MVP adds a Python package, FastAPI moderation service, SQLite moderation queue, analytics, and a transformer training/export path.

### Local Setup

```bash
python3 -m pip install -e ".[dev]"
```

### Run Tests

```bash
python3 -m pytest
```

### Start The API

By default, the API uses a deterministic mock model so the moderation queue and analytics can be tested without waiting for transformer training.

```bash
SAFETWEET_DB_PATH=var/moderation.db \
python3 -m uvicorn safetweet.api.app:app --host 127.0.0.1 --port 8000
```

### Moderate Text

```bash
curl -s -X POST http://127.0.0.1:8000/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text":"How can I steal something?"}'
```

### Inspect Queue And Analytics

```bash
curl -s http://127.0.0.1:8000/queue
curl -s http://127.0.0.1:8000/analytics
```

### Train A Transformer

Use small limits for a smoke test:

```bash
python3 -m safetweet.training.train_transformer \
  --train-limit 32 \
  --eval-limit 32 \
  --epochs 0.1 \
  --batch-size 4 \
  --output-dir models/transformer-smoke \
  --metrics-dir reports/metrics-smoke \
  --model-version smoke-test
```

Use the exported model in the API:

```bash
SAFETWEET_MODEL_DIR=models/transformer-smoke \
SAFETWEET_DB_PATH=var/moderation.db \
python3 -m uvicorn safetweet.api.app:app --host 127.0.0.1 --port 8000
```
````

- [ ] **Step 2: Run tests after docs update**

Run:

```bash
python3 -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: add moderation mvp usage"
```

## Task 11: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run full unit test suite**

Run:

```bash
python3 -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run a package import smoke test**

Run:

```bash
python3 -c "from safetweet.api.app import create_app; app = create_app(); print(app.title)"
```

Expected output contains:

```text
Safe Tweet Detector
```

- [ ] **Step 3: Run API smoke test**

Run the API:

```bash
SAFETWEET_DB_PATH=var/final-smoke.db python3 -m uvicorn safetweet.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hope you have a wonderful day."}'
curl -s http://127.0.0.1:8000/analytics
```

Expected: health reports `ready`, moderation returns `allow`, and analytics reports at least one prediction.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only known ignored or pre-existing local files remain outside commits.

- [ ] **Step 5: Record final verification result**

Add a short note to the implementation handoff or final message containing:

```text
Verified:
- python3 -m pytest
- python3 -c "from safetweet.api.app import create_app; app = create_app(); print(app.title)"
- API smoke test with /health, /moderate, and /analytics
```

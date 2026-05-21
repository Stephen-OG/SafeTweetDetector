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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

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

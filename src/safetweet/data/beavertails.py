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

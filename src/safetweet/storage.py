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

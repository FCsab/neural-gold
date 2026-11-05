"""SQLite model registry.

Two tables for now: ``models`` and ``training_runs``. They are deliberately a
subset of the plan's schema — predictions_log, adwin_state and retraining_jobs
arrive with the execution and drift loops, and inventing empty tables now
would hide which parts are actually implemented.

Writes happen only in the server process (the training child writes checkpoint
files, never SQLite). That keeps the "one writer" rule from the plan intact
without a queue: nothing else contends.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id                        TEXT PRIMARY KEY,
    name                      TEXT NOT NULL,
    created_at                TEXT NOT NULL,
    interval                  TEXT NOT NULL,
    architecture              TEXT NOT NULL,
    window                    INTEGER NOT NULL,
    status                    TEXT NOT NULL,
    parent_id                 TEXT,
    label_config              TEXT NOT NULL,
    feature_config            TEXT NOT NULL,
    checkpoint_path           TEXT,
    checkpoint_version        INTEGER NOT NULL DEFAULT 0,
    online_adaptation_enabled INTEGER NOT NULL DEFAULT 0,
    metrics                   TEXT
);
CREATE TABLE IF NOT EXISTS training_runs (
    id                 TEXT PRIMARY KEY,
    model_id           TEXT NOT NULL REFERENCES models(id),
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL,
    config             TEXT NOT NULL,
    final_metrics      TEXT,
    per_class_metrics  TEXT,
    confusion_matrix   TEXT,
    epochs_done        INTEGER,
    stop_reason        TEXT,
    duration_s         REAL,
    device             TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_model ON training_runs(model_id);
"""

JSON_COLUMNS = {
    "label_config",
    "feature_config",
    "metrics",
    "config",
    "final_metrics",
    "per_class_metrics",
    "confusion_matrix",
}


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- models ----------------------------------------------------------
    def create_model(
        self,
        *,
        name: str,
        interval: str,
        architecture: str,
        window: int,
        label_config: dict[str, Any],
        feature_config: dict[str, Any],
        status: str = "training",
        parent_id: str | None = None,
        model_id: str | None = None,
    ) -> str:
        model_id = model_id or new_id("model")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO models (id, name, created_at, interval, architecture,
                                    window, status, parent_id, label_config, feature_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    name,
                    _now(),
                    interval,
                    architecture,
                    int(window),
                    status,
                    parent_id,
                    json.dumps(label_config),
                    json.dumps(feature_config),
                ),
            )
        return model_id

    def update_model(self, model_id: str, **fields: Any) -> None:
        allowed = {
            "name",
            "status",
            "checkpoint_path",
            "checkpoint_version",
            "metrics",
            "online_adaptation_enabled",
            "parent_id",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update model fields: {sorted(unknown)}")

        payload = {
            key: json.dumps(value) if key in JSON_COLUMNS else value
            for key, value in fields.items()
        }
        assignments = ", ".join(f"{key} = ?" for key in payload)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE models SET {assignments} WHERE id = ?",  # noqa: S608 - keys are allowlisted
                (*payload.values(), model_id),
            )

    def list_models(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(dict(row)) for row in rows]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE id = ?", (model_id,)
            ).fetchone()
        return self._decode(dict(row)) if row else None

    def count_models(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])

    # -- runs ------------------------------------------------------------
    def record_run(
        self,
        *,
        model_id: str,
        started_at: str,
        status: str,
        config: dict[str, Any],
        finished_at: str | None = None,
        final_metrics: dict[str, Any] | None = None,
        per_class_metrics: dict[str, Any] | None = None,
        confusion_matrix: list[list[int]] | None = None,
        epochs_done: int | None = None,
        stop_reason: str | None = None,
        duration_s: float | None = None,
        device: str | None = None,
    ) -> str:
        run_id = new_id("run")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs
                    (id, model_id, started_at, finished_at, status, config,
                     final_metrics, per_class_metrics, confusion_matrix,
                     epochs_done, stop_reason, duration_s, device)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    started_at,
                    finished_at or _now(),
                    status,
                    json.dumps(config),
                    json.dumps(final_metrics) if final_metrics else None,
                    json.dumps(per_class_metrics) if per_class_metrics else None,
                    json.dumps(confusion_matrix) if confusion_matrix else None,
                    epochs_done,
                    stop_reason,
                    duration_s,
                    device,
                ),
            )
        return run_id

    def list_runs(self, model_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM training_runs"
        params: tuple[Any, ...] = ()
        if model_id:
            query += " WHERE model_id = ?"
            params = (model_id,)
        query += " ORDER BY started_at DESC LIMIT ?"
        params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._decode(dict(row)) for row in rows]

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        for key in JSON_COLUMNS & row.keys():
            if row[key]:
                try:
                    row[key] = json.loads(row[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return row


def registry_from_config(config) -> Registry:
    return Registry(config.data_root / "registry.sqlite")

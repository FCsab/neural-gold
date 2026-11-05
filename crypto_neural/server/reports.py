"""Small JSON report store under ``data/reports``.

The UI shows the *last* dataset, leakage, and verification results. Keeping
them on disk means a browser refresh — or a server restart — does not lose
the evidence, and the files can be diffed or committed elsewhere.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def reports_dir(config) -> Path:
    path = config.reports_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(config, name: str) -> Path:
    return reports_dir(config) / f"{name}.json"


def write_report(config, name: str, payload: dict[str, Any]) -> Path:
    path = report_path(config, name)
    body = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str))
    tmp.replace(path)
    return path


def read_report(config, name: str) -> dict[str, Any] | None:
    path = report_path(config, name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

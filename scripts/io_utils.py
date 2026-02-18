from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_stamp(iso8601: str) -> str:
    # For filenames: remove ":" and other problematic characters.
    return (
        iso8601.strip()
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

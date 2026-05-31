"""General utilities used across project scripts."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int = 42) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_markdown_entry(path: Path, heading: str, body: str) -> None:
    """Append a dated markdown entry to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# {path.stem.replace('_', ' ').title()}\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {utc_timestamp()} — {heading}\n\n{body.strip()}\n")


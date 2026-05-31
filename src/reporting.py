"""Experiment and report logging helpers."""

from __future__ import annotations

from .paths import EXPERIMENT_LOG_PATH, REPORT_NOTES_PATH
from .utils import append_markdown_entry


def log_experiment(model_id: str, metrics: dict[str, float], notes: str = "") -> None:
    """Append a compact experiment entry to `EXPERIMENT_LOG.md`."""
    metric_lines = "\n".join(f"- `{key}`: {value:.4f}" for key, value in metrics.items())
    body = f"**Model:** `{model_id}`\n\n{metric_lines}\n\n{notes}".strip()
    append_markdown_entry(EXPERIMENT_LOG_PATH, f"Experiment {model_id}", body)


def add_report_note(heading: str, body: str) -> None:
    """Append an observation to `REPORT_NOTES.md`."""
    append_markdown_entry(REPORT_NOTES_PATH, heading, body)


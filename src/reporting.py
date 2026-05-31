"""Experiment and report logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import EXPERIMENT_LOG_PATH, LOGS_DIR, REPORT_NOTES_PATH
from .utils import append_markdown_entry


def log_experiment(model_id: str, metrics: dict[str, float], notes: str = "") -> None:
    """Append a compact experiment entry to `EXPERIMENT_LOG.md`."""
    metric_lines = "\n".join(f"- `{key}`: {value:.4f}" for key, value in metrics.items())
    body = f"**Model:** `{model_id}`\n\n{metric_lines}\n\n{notes}".strip()
    append_markdown_entry(EXPERIMENT_LOG_PATH, f"Experiment {model_id}", body)


def add_report_note(heading: str, body: str) -> None:
    """Append an observation to `REPORT_NOTES.md`."""
    append_markdown_entry(REPORT_NOTES_PATH, heading, body)


def upsert_markdown_section(path: Path, title: str, body: str) -> None:
    """Create or replace a markdown section."""
    if not path.exists():
        path.write_text(f"# {path.stem.replace('_', ' ').title()}\n", encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    heading = f"## {title}"
    replacement = f"{heading}\n\n{body.strip()}\n"
    if heading not in content:
        path.write_text(content.rstrip() + "\n\n" + replacement, encoding="utf-8")
        return
    start = content.index(heading)
    next_heading = content.find("\n## ", start + len(heading))
    if next_heading == -1:
        updated = content[:start].rstrip() + "\n\n" + replacement
    else:
        updated = content[:start].rstrip() + "\n\n" + replacement + content[next_heading:]
    path.write_text(updated, encoding="utf-8")


def write_run_summary(
    version_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifact_paths: dict[str, str],
) -> Path:
    """Write a run summary markdown file and update the experiment log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{version_name}_run.md"
    lines = [
        f"# Run Summary: {version_name}",
        "",
        "## Config",
        "",
        "```text",
        *[f"{key}: {value}" for key, value in config.items()],
        "```",
        "",
        "## Metrics",
        "",
    ]
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "top_2_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
    ]:
        if key in metrics:
            lines.append(f"- `{key}`: {metrics[key]:.6f}")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in artifact_paths.items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    experiment_body = "\n".join(
        [
            f"**Version:** `{version_name}`",
            "",
            "Metrics:",
            *[
                f"- `{key}`: {metrics[key]:.6f}"
                for key in ["accuracy", "macro_f1", "weighted_f1"]
                if key in metrics
            ],
            "",
            f"Run summary: `{path}`",
        ]
    )
    upsert_markdown_section(EXPERIMENT_LOG_PATH, f"Run {version_name}", experiment_body)
    return path

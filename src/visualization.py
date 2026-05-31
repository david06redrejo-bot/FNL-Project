"""Plotting helpers used by notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_label_distribution(series, path: Path, title: str = "Label Distribution") -> None:
    """Save a bar plot for category counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = series.value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("ICD category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


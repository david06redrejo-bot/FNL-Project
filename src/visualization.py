"""Plotting helpers used by notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def _prepare_plot(path: Path, figsize: tuple[float, float]) -> None:
    """Create a plot directory and configure common style."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(PLOT_STYLE)
    plt.figure(figsize=figsize)


def _finish_plot(path: Path) -> None:
    """Save and close the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


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


def plot_category_distribution(category_df: pd.DataFrame, path: Path) -> None:
    """Question: which ICD categories dominate the training labels?"""
    _prepare_plot(path, (12, 5))
    plot_df = category_df.sort_values("y_category")
    sns.barplot(data=plot_df, x="y_category", y="count", color="#4c78a8")
    plt.title("Training label distribution by ICD category")
    plt.xlabel("y_category")
    plt.ylabel("Training rows")
    _finish_plot(path)


def plot_long_tail_distribution(train_df: pd.DataFrame, path: Path) -> None:
    """Question: how severe is the long tail at full ICD-code level?"""
    _prepare_plot(path, (8, 5))
    counts = train_df["Code"].value_counts().reset_index()
    counts.columns = ["Code", "count"]
    counts["rank"] = range(1, len(counts) + 1)
    sns.lineplot(data=counts, x="rank", y="count", color="#f58518")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Long-tail distribution of full ICD codes")
    plt.xlabel("ICD code rank by frequency (log scale)")
    plt.ylabel("Rows per code (log scale)")
    _finish_plot(path)


def plot_literal_length_distribution(train_df: pd.DataFrame, path: Path) -> None:
    """Question: how much text context does each training example contain?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.set_theme(style="whitegrid")
    sns.histplot(train_df["char_len"], bins=35, ax=axes[0], color="#4c78a8")
    axes[0].set_title("Training literal length in characters")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Rows")
    sns.histplot(train_df["whitespace_tokens"], bins=range(1, int(train_df["whitespace_tokens"].max()) + 3), ax=axes[1], color="#54a24b")
    axes[1].set_title("Training literal length in whitespace tokens")
    axes[1].set_xlabel("Whitespace tokens")
    axes[1].set_ylabel("Rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_length_by_category(train_df: pd.DataFrame, path: Path) -> None:
    """Question: are some categories described with longer literals?"""
    _prepare_plot(path, (12, 5))
    order = train_df.groupby("y_category")["whitespace_tokens"].median().sort_values().index
    sns.boxplot(
        data=train_df,
        x="y_category",
        y="whitespace_tokens",
        order=order,
        color="#72b7b2",
        showfliers=False,
    )
    plt.title("Literal length by category")
    plt.xlabel("y_category ordered by median token length")
    plt.ylabel("Whitespace tokens")
    _finish_plot(path)


def plot_train_vs_leaderboard_lengths(train_df: pd.DataFrame, leaderboard_df: pd.DataFrame, path: Path) -> None:
    """Question: do train and leaderboard literals have similar lengths?"""
    combined = pd.concat(
        [
            train_df[["char_len", "whitespace_tokens"]].assign(dataset="train"),
            leaderboard_df[["char_len", "whitespace_tokens"]].assign(dataset="leaderboard"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.set_theme(style="whitegrid")
    sns.kdeplot(data=combined, x="char_len", hue="dataset", common_norm=False, ax=axes[0])
    axes[0].set_title("Character length: train vs leaderboard")
    axes[0].set_xlabel("Characters")
    sns.kdeplot(
        data=combined,
        x="whitespace_tokens",
        hue="dataset",
        common_norm=False,
        ax=axes[1],
        bw_adjust=1.4,
    )
    axes[1].set_title("Whitespace tokens: train vs leaderboard")
    axes[1].set_xlabel("Whitespace tokens")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_text_pattern_presence(pattern_df: pd.DataFrame, path: Path) -> None:
    """Question: which noisy clinical text patterns appear in each split?"""
    _prepare_plot(path, (12, 5))
    plot_df = pattern_df.copy()
    plot_df["pattern"] = plot_df["pattern"].str.replace("has_", "", regex=False).str.replace("_", " ")
    plot_df["share_percent"] = plot_df["share"] * 100
    sns.barplot(data=plot_df, x="pattern", y="share_percent", hue="dataset")
    plt.title("Text pattern presence in clinical literals")
    plt.xlabel("Pattern")
    plt.ylabel("Rows with pattern (%)")
    plt.xticks(rotation=30, ha="right")
    _finish_plot(path)

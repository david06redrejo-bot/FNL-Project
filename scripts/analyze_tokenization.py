"""Analyze tokenizer lengths for the RoBERTa backbone without training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import REPORTS_DIR


MODEL_NAME = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
MAX_LENGTHS = [32, 64, 96, 128, 192, 256]
TEXT_COL = "Literal_required_clean"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def upsert_section(path: Path, title: str, body: str) -> None:
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


def load_tokenizer():
    """Load the requested Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def token_lengths(tokenizer, texts: pd.Series, batch_size: int = 512) -> list[int]:
    """Tokenize texts in batches and return sequence lengths with special tokens."""
    values = texts.fillna("").astype(str).tolist()
    lengths: list[int] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def summarize_lengths(df: pd.DataFrame, dataset: str) -> dict[str, float | int | str]:
    """Compute percentile token length summary for one split."""
    lengths = df["token_len"]
    percentiles = lengths.quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "mean": float(lengths.mean()),
        "p50": int(percentiles.loc[0.50]),
        "p75": int(percentiles.loc[0.75]),
        "p90": int(percentiles.loc[0.90]),
        "p95": int(percentiles.loc[0.95]),
        "p99": int(percentiles.loc[0.99]),
        "max": int(lengths.max()),
    }


def choose_default_max_length(length_summary: pd.DataFrame) -> int:
    """Choose the smallest candidate that covers the training p99."""
    train_p99 = int(length_summary.loc[length_summary["dataset"] == "train", "p99"].iloc[0])
    for value in MAX_LENGTHS:
        if value >= train_p99:
            return value
    return MAX_LENGTHS[-1]


def plot_token_distribution(combined: pd.DataFrame, path: Path) -> None:
    """Save train vs leaderboard token length histogram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=combined,
        x="token_len",
        hue="dataset",
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("RoBERTa tokenizer length distribution")
    plt.xlabel("Tokenizer tokens including special tokens")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_truncation(truncation: pd.DataFrame, path: Path) -> None:
    """Save truncation rate by max_length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_df = truncation.copy()
    plot_df["truncation_percent"] = plot_df["truncation_rate"] * 100
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=plot_df,
        x="max_length",
        y="truncation_percent",
        hue="dataset",
        marker="o",
    )
    plt.title("Examples truncated by tokenizer max_length")
    plt.xlabel("max_length")
    plt.ylabel("Truncated examples (%)")
    plt.xticks(MAX_LENGTHS)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROJECT_ROOT / "data" / "processed" / "train_required_clean.csv"
    leaderboard_path = PROJECT_ROOT / "data" / "processed" / "leaderboard_required_clean.csv"
    if not train_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError(
            "Required-clean files are missing. Run `python scripts/run_preprocessing.py` first."
        )

    try:
        tokenizer = load_tokenizer()
    except Exception as exc:  # noqa: BLE001 - exact error is intentionally documented.
        error_text = (
            f"Tokenizer download/load failed for `{MODEL_NAME}`.\n\n"
            f"Exact error:\n{type(exc).__name__}: {exc}\n\n"
            "Retry command:\npython scripts/analyze_tokenization.py\n"
        )
        error_path = TABLES_DIR / "tokenizer_download_error.txt"
        error_path.write_text(error_text, encoding="utf-8")
        upsert_section(
            PROJECT_ROOT / "REPORT_NOTES.md",
            "Tokenizer Analysis Blocked",
            error_text + "\nNo tokenization statistics were fabricated.",
        )
        print(error_text)
        return 2

    train = pd.read_csv(train_path)
    leaderboard = pd.read_csv(leaderboard_path)
    train["dataset"] = "train"
    leaderboard["dataset"] = "leaderboard"
    train["token_len"] = token_lengths(tokenizer, train[TEXT_COL])
    leaderboard["token_len"] = token_lengths(tokenizer, leaderboard[TEXT_COL])
    combined = pd.concat(
        [
            train[["dataset", TEXT_COL, "token_len"]],
            leaderboard[["dataset", TEXT_COL, "token_len"]],
        ],
        ignore_index=True,
    )

    length_summary = pd.DataFrame(
        [summarize_lengths(train, "train"), summarize_lengths(leaderboard, "leaderboard")]
    )
    default_max_length = choose_default_max_length(length_summary)
    length_summary["recommended_default_max_length"] = default_max_length
    length_summary["tokenizer"] = MODEL_NAME

    truncation_rows = []
    for dataset, frame in [("train", train), ("leaderboard", leaderboard)]:
        for max_length in MAX_LENGTHS:
            truncated = frame["token_len"] > max_length
            truncation_rows.append(
                {
                    "dataset": dataset,
                    "max_length": max_length,
                    "truncated_examples": int(truncated.sum()),
                    "total_examples": int(len(frame)),
                    "truncation_rate": float(truncated.mean()),
                    "recommended_default": max_length == default_max_length,
                }
            )
    truncation = pd.DataFrame(truncation_rows)

    length_summary.to_csv(TABLES_DIR / "token_length_summary.csv", index=False)
    truncation.to_csv(TABLES_DIR / "truncation_by_max_length.csv", index=False)
    combined.to_csv(TABLES_DIR / "token_lengths_by_split.csv", index=False)
    plot_token_distribution(combined, FIGURES_DIR / "fig_07_token_length_distribution.png")
    plot_truncation(truncation, FIGURES_DIR / "fig_08_truncation_rate_by_max_length.png")

    train_row = length_summary[length_summary["dataset"] == "train"].iloc[0]
    leaderboard_row = length_summary[length_summary["dataset"] == "leaderboard"].iloc[0]
    report_body = f"""
We analyzed tokenization with `{MODEL_NAME}` using the required-clean literals
and without training any model.

Subword tokenization matters because RoBERTa does not see whitespace words
directly. It sees subword pieces created by the pretrained tokenizer. Therefore,
`max_length` is not arbitrary: too small a value truncates information; too
large a value wastes compute.

Token length summary including special tokens:

- Train p50/p75/p90/p95/p99/max:
  {train_row['p50']}/{train_row['p75']}/{train_row['p90']}/{train_row['p95']}/{train_row['p99']}/{train_row['max']}
- Leaderboard p50/p75/p90/p95/p99/max:
  {leaderboard_row['p50']}/{leaderboard_row['p75']}/{leaderboard_row['p90']}/{leaderboard_row['p95']}/{leaderboard_row['p99']}/{leaderboard_row['max']}

Recommended default `max_length`: **{default_max_length}**.

This confirms that our task is much easier than long EMR ICD coding with full
discharge summaries from a sequence-length point of view. The inputs are short
literals, not multi-page documents. However, the same shortness creates another
problem: ambiguous literals often lack the surrounding context that would
disambiguate the correct ICD category.

No model has been trained yet; these are tokenizer-design conclusions only.
"""
    decision_body = f"""
Decision: use `{MODEL_NAME}` as the tokenizer for RoBERTa experiments and set
the initial default `max_length` to **{default_max_length}**. This value is based
on the observed token length percentiles of required-clean train and leaderboard
literals, not on an arbitrary default. Future ablations may compare nearby
values, but this is the default starting point.
"""
    experiment_body = (
        f"Analyzed RoBERTa tokenizer lengths with `{MODEL_NAME}`. "
        f"Recommended max_length={default_max_length}. No model training, "
        "validation, or prediction was run."
    )
    upsert_section(PROJECT_ROOT / "REPORT_NOTES.md", "RoBERTa Tokenization Analysis", report_body)
    upsert_section(PROJECT_ROOT / "DECISIONS.md", "RoBERTa Tokenizer max_length Decision", decision_body)
    upsert_section(PROJECT_ROOT / "EXPERIMENT_LOG.md", "No model yet; tokenizer analysis only", experiment_body)

    print(f"Wrote {TABLES_DIR / 'token_length_summary.csv'}")
    print(f"Wrote {TABLES_DIR / 'truncation_by_max_length.csv'}")
    print(f"Wrote {FIGURES_DIR / 'fig_07_token_length_distribution.png'}")
    print(f"Wrote {FIGURES_DIR / 'fig_08_truncation_rate_by_max_length.png'}")
    print(f"Recommended default max_length: {default_max_length}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

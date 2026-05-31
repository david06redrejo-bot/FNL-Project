"""Create required-clean processed datasets and preprocessing ablation artifacts.

This is a preprocessing/design phase only. It does not train any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import add_label_columns, load_csv
from src.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.preprocessing import clean_required, compare_preprocessing_effects


ABLATION_DIR = INTERIM_DATA_DIR / "preprocessing_ablation"


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


def build_ablation_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each preprocessing ablation changes literals."""
    total = len(effects)
    rows = []
    for col, label in [
        ("required_changed", "required whitespace cleanup"),
        ("lowercase_changed", "lowercase"),
        ("accent_removal_changed", "accent removal"),
        ("punctuation_removal_changed", "punctuation removal"),
    ]:
        count = int(effects[col].sum())
        rows.append(
            {
                "variant": label,
                "changed_rows": count,
                "changed_share": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_csv("codification")
    leaderboard_raw = load_csv("leaderboard")

    train_labeled, mapping = add_label_columns(train_raw)
    train_processed = train_labeled.copy()
    leaderboard_processed = leaderboard_raw.copy()

    train_processed["Literal_required_clean"] = train_processed["Literal"].map(clean_required)
    leaderboard_processed["Literal_required_clean"] = leaderboard_processed["Literal"].map(
        clean_required
    )

    train_path = PROCESSED_DATA_DIR / "train_required_clean.csv"
    leaderboard_path = PROCESSED_DATA_DIR / "leaderboard_required_clean.csv"
    train_processed.to_csv(train_path, index=False)
    leaderboard_processed.to_csv(leaderboard_path, index=False)

    effects = compare_preprocessing_effects(train_raw)
    examples = effects[
        effects[
            [
                "required_changed",
                "lowercase_changed",
                "accent_removal_changed",
                "punctuation_removal_changed",
            ]
        ].any(axis=1)
    ].head(80)
    summary = build_ablation_summary(effects)

    summary.to_csv(ABLATION_DIR / "preprocessing_ablation_summary.csv", index=False)
    examples.to_csv(ABLATION_DIR / "preprocessing_ablation_examples.csv", index=False)

    report_body = f"""
The final preprocessing decision is deliberately conservative. For the RoBERTa
pipeline, we use only required light cleanup: convert null-safe values to text,
strip leading/trailing spaces, and collapse repeated whitespace.

We do **not** lowercase, remove accents, or remove punctuation for the final
RoBERTa pipeline because the backbone tokenizer is pretrained on Spanish
biomedical and clinical text. Aggressive normalization could remove useful
signals from accents, uppercase abbreviations, punctuation, digits, and compact
clinical notation.

Processed files created:

- `data/processed/train_required_clean.csv`
- `data/processed/leaderboard_required_clean.csv`

Ablation artifacts for analysis/classical baselines:

- `data/interim/preprocessing_ablation/preprocessing_ablation_summary.csv`
- `data/interim/preprocessing_ablation/preprocessing_ablation_examples.csv`

Observed ablation impact:

```text
{summary.to_string(index=False)}
```

No model has been trained yet; these are preprocessing-design conclusions only.
"""
    upsert_section(
        PROJECT_ROOT / "REPORT_NOTES.md",
        "Preprocessing Design: Light Cleaning for RoBERTa",
        report_body,
    )
    upsert_section(
        PROJECT_ROOT / "DECISIONS.md",
        "Final Preprocessing Decision",
        "Decision: use `clean_required` for the final RoBERTa pipeline. This preserves case, accents, punctuation, digits, and abbreviations while stripping leading/trailing spaces and collapsing repeated whitespace. Stronger normalization is reserved for classical-baseline ablations only.",
    )
    upsert_section(
        PROJECT_ROOT / "EXPERIMENT_LOG.md",
        "No model yet; preprocessing conclusions only",
        "Created required-clean processed datasets and preprocessing ablation summaries. No model training, validation, or prediction was run.",
    )

    print(f"Wrote {train_path}")
    print(f"Wrote {leaderboard_path}")
    print(f"Wrote {ABLATION_DIR / 'preprocessing_ablation_summary.csv'}")
    print(f"Wrote {ABLATION_DIR / 'preprocessing_ablation_examples.csv'}")
    print(f"Label classes preserved: {mapping['n_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

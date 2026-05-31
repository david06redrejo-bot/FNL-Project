"""Run the visual EDA phase.

This script continues the first project phase: analyzing the data and the
annotations. It creates figures and tables only; it does not train models.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_csv
from src.eda import (
    add_annotation_columns,
    add_literal_features,
    build_eda_key_findings,
    category_distribution,
    code_distribution_by_category,
    duplicate_literal_analysis,
    literal_examples_by_category,
    normalization_collision_summary,
    text_pattern_summary,
    train_leaderboard_shift_summary,
)
from src.paths import REPORTS_DIR, ensure_project_dirs
from src.visualization import (
    plot_category_distribution,
    plot_length_by_category,
    plot_literal_length_distribution,
    plot_long_tail_distribution,
    plot_text_pattern_presence,
    plot_train_vs_leaderboard_lengths,
)


FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


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


def main() -> int:
    ensure_project_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    codification = load_csv("codification")
    leaderboard = load_csv("leaderboard")
    icd_catalog = load_csv("icd_catalog")

    train = add_literal_features(add_annotation_columns(codification))
    leaderboard_features = add_literal_features(leaderboard)

    category_df = category_distribution(train)
    code_category_df = code_distribution_by_category(train)
    examples_df = literal_examples_by_category(train)
    duplicate_df = duplicate_literal_analysis(train)
    pattern_df = text_pattern_summary(train, leaderboard_features)
    shift = train_leaderboard_shift_summary(train, leaderboard_features)
    normalization = normalization_collision_summary(train)
    key_findings = build_eda_key_findings(
        train,
        leaderboard_features,
        category_df,
        duplicate_df,
        shift,
        normalization,
    )

    # Extra rows requested by the brief, kept inside the category distribution table.
    category_table = category_df.merge(code_category_df, on="y_category", how="left")
    category_table = category_table.sort_values("count", ascending=False)
    category_table.to_csv(TABLES_DIR / "category_distribution.csv", index=False)
    examples_df.to_csv(TABLES_DIR / "literal_examples_by_category.csv", index=False)
    duplicate_df.to_csv(TABLES_DIR / "duplicate_literal_analysis.csv", index=False)
    key_findings.to_csv(TABLES_DIR / "eda_key_findings.csv", index=False)

    # Supporting tables that make the figures and conclusions traceable.
    pattern_df.to_csv(TABLES_DIR / "text_pattern_summary.csv", index=False)
    pd.DataFrame([shift]).to_csv(TABLES_DIR / "train_leaderboard_shift_summary.csv", index=False)
    pd.DataFrame([normalization]).to_csv(TABLES_DIR / "normalization_collision_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "training_rows": len(train),
                "leaderboard_rows": len(leaderboard_features),
                "icd_catalog_rows": len(icd_catalog),
                "unique_icd_codes_train": train["Code"].nunique(),
                "unique_y_categories_train": train["y_category"].nunique(),
                "unique_icd_codes_catalog": icd_catalog["Code"].nunique(),
                "duplicate_literals": int(duplicate_df["is_duplicate_literal"].sum()),
                "same_literal_multiple_codes": int(
                    duplicate_df["same_literal_multiple_codes"].sum()
                ),
                "same_literal_multiple_y_categories": int(
                    duplicate_df["same_literal_multiple_y_categories"].sum()
                ),
            }
        ]
    ).to_csv(TABLES_DIR / "eda_dataset_summary.csv", index=False)

    plot_category_distribution(category_table, FIGURES_DIR / "fig_01_category_distribution.png")
    plot_long_tail_distribution(train, FIGURES_DIR / "fig_02_long_tail_distribution.png")
    plot_literal_length_distribution(train, FIGURES_DIR / "fig_03_literal_length_distribution.png")
    plot_length_by_category(train, FIGURES_DIR / "fig_04_length_by_category.png")
    plot_train_vs_leaderboard_lengths(
        train,
        leaderboard_features,
        FIGURES_DIR / "fig_05_train_vs_leaderboard_lengths.png",
    )
    plot_text_pattern_presence(pattern_df, FIGURES_DIR / "fig_06_text_pattern_presence.png")

    top10 = category_table.head(10)[["y_category", "count"]].to_dict("records")
    bottom10 = category_table.sort_values("count").head(10)[
        ["y_category", "count"]
    ].to_dict("records")
    ambiguous_literals = int(duplicate_df["same_literal_multiple_y_categories"].sum())

    report_body = f"""
We continued the first phase with a visual EDA inspired by a Data Engineering
mindset: before modeling, we inspected the data-generating process, data
quality, distributions, missingness, duplicates, and possible leakage.

Core size and annotation facts:

- Training rows: {len(train)}
- Leaderboard rows: {len(leaderboard_features)}
- Unique full ICD codes in training: {train['Code'].nunique()}
- Unique `y_category` labels: {train['y_category'].nunique()}
- Unique full ICD codes in the ICD catalog: {icd_catalog['Code'].nunique()}

The top 10 categories by row count are: {top10}.
The bottom 10 categories by row count are: {bottom10}.

Duplicate and ambiguity findings:

- Duplicate literals: {int(duplicate_df['is_duplicate_literal'].sum())}
- Same literal with multiple full ICD codes: {int(duplicate_df['same_literal_multiple_codes'].sum())}
- Same literal with multiple `y_category` labels: {ambiguous_literals}

Train/leaderboard comparison:

- {shift['leaderboard_seen_in_train_share']:.1%} of unique normalized leaderboard
  literals appear in the training set.
- Mean character length is {shift['train_mean_chars']:.2f} in train and
  {shift['leaderboard_mean_chars']:.2f} in leaderboard.

Normalization risk:

- Raw unique literals: {normalization['raw_unique_literals']}
- Unique literals after lowercase/accent-stripped normalization:
  {normalization['normalized_no_accents_unique_literals']}
- Normalization collision keys:
  {normalization['normalization_collision_keys']}

No model has been trained yet; these are EDA conclusions only.
"""
    upsert_section(PROJECT_ROOT / "REPORT_NOTES.md", "Visual EDA Conclusions", report_body)
    upsert_section(
        PROJECT_ROOT / "EXPERIMENT_LOG.md",
        "No model yet; EDA conclusions only",
        "Generated the visual EDA figures and tables. No model training, validation, or prediction was run.",
    )
    upsert_section(
        PROJECT_ROOT / "DECISIONS.md",
        "EDA Before Modeling",
        "Decision: continue with EDA conclusions only before training. The observed imbalance, duplicate literals, and ambiguous literal-category mappings must shape the baseline design and evaluation plan.",
    )

    expected_outputs = [
        FIGURES_DIR / "fig_01_category_distribution.png",
        FIGURES_DIR / "fig_02_long_tail_distribution.png",
        FIGURES_DIR / "fig_03_literal_length_distribution.png",
        FIGURES_DIR / "fig_04_length_by_category.png",
        FIGURES_DIR / "fig_05_train_vs_leaderboard_lengths.png",
        FIGURES_DIR / "fig_06_text_pattern_presence.png",
        TABLES_DIR / "category_distribution.csv",
        TABLES_DIR / "literal_examples_by_category.csv",
        TABLES_DIR / "duplicate_literal_analysis.csv",
        TABLES_DIR / "eda_key_findings.csv",
    ]
    missing = [str(path) for path in expected_outputs if not path.exists()]
    if missing:
        print("Missing expected EDA outputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("Visual EDA complete. Generated:")
    for path in expected_outputs:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

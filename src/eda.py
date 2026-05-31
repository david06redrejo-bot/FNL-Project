"""Reusable EDA summaries for notebooks and scripts."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from .preprocessing import extract_y_category, normalize_literal


def summarize_codification(df: pd.DataFrame) -> dict[str, object]:
    """Return basic annotation statistics for the training pairs."""
    categories = df["Code"].map(extract_y_category)
    return {
        "rows": int(len(df)),
        "unique_codes": int(df["Code"].nunique()),
        "unique_literals": int(df["Literal"].nunique()),
        "missing_values": int(df[["Code", "Literal"]].isna().sum().sum()),
        "categories": sorted(categories.dropna().unique().tolist()),
        "n_categories": int(categories.nunique()),
    }


def literal_length_summary(df: pd.DataFrame, literal_col: str = "Literal") -> dict[str, float]:
    """Summarize clinical literal length in characters and whitespace tokens."""
    text = df[literal_col].fillna("").astype(str)
    token_counts = text.str.split().map(len)
    char_counts = text.str.len()
    return {
        "mean_chars": float(char_counts.mean()),
        "median_chars": float(char_counts.median()),
        "mean_tokens": float(token_counts.mean()),
        "median_tokens": float(token_counts.median()),
    }


def add_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the project target `y_category` to a codification dataframe."""
    result = df.copy()
    result["Code"] = result["Code"].astype(str)
    result["Literal"] = result["Literal"].astype(str)
    result["y_category"] = result["Code"].map(extract_y_category)
    return result


def add_literal_features(df: pd.DataFrame, literal_col: str = "Literal") -> pd.DataFrame:
    """Add length, normalization, and text-pattern features for EDA."""
    result = df.copy()
    literal = result[literal_col].fillna("").astype(str)
    normalized_no_accents = literal.map(lambda text: normalize_literal(text, strip_accents=True))
    normalized_keep_accents = literal.map(lambda text: normalize_literal(text, strip_accents=False))

    result["char_len"] = literal.str.len()
    result["whitespace_tokens"] = literal.str.split().map(len)
    result["tokenizer_tokens"] = pd.NA
    result["literal_lower"] = literal.str.lower()
    result["literal_norm_no_accents"] = normalized_no_accents
    result["literal_norm_keep_accents"] = normalized_keep_accents
    result["has_digit"] = literal.str.contains(r"\d", regex=True)
    result["is_all_upper"] = literal.str.contains(r"[A-ZÁÉÍÓÚÜÑÇ]", regex=True) & (
        literal == literal.str.upper()
    )
    result["has_slash"] = literal.str.contains("/", regex=False)
    result["has_hyphen"] = literal.str.contains("-", regex=False)
    result["has_parentheses"] = literal.str.contains(r"[()]", regex=True)
    result["has_accent"] = literal.map(has_accent)
    result["has_punctuation"] = literal.str.contains(r"[^\w\sáéíóúüñçàèìòùÁÉÍÓÚÜÑÇÀÈÌÒÙ]", regex=True)
    result["has_measurement_like"] = literal.str.contains(
        r"(?:\d+\s*(?:mg|ml|cm|mm|kg|g|mcg|ui|%|x|/|\\+)|\d+[.,]\d+)",
        case=False,
        regex=True,
    )
    return result


def has_accent(text: str) -> bool:
    """Return True when text contains a combining accent after decomposition."""
    decomposed = unicodedata.normalize("NFD", str(text))
    return any(unicodedata.category(ch) == "Mn" for ch in decomposed)


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-category training counts and shares."""
    counts = df["y_category"].value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame(
        {
            "y_category": counts.index,
            "count": counts.values,
            "share": counts.values / total,
            "rank_by_count": counts.rank(method="first", ascending=False).astype(int).values,
        }
    ).sort_values("count", ascending=False)


def code_distribution_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique full ICD codes per category."""
    return (
        df.groupby("y_category")["Code"]
        .nunique()
        .reset_index(name="unique_codes")
        .sort_values("unique_codes", ascending=False)
    )


def literal_examples_by_category(df: pd.DataFrame, examples_per_category: int = 5) -> pd.DataFrame:
    """Collect representative literal examples for each category."""
    rows = []
    for category, group in df.sort_values(["y_category", "Literal"]).groupby("y_category"):
        examples = group["Literal"].dropna().astype(str).drop_duplicates().head(examples_per_category)
        rows.append(
            {
                "y_category": category,
                "n_rows": int(len(group)),
                "n_unique_codes": int(group["Code"].nunique()),
                "example_literals": " || ".join(examples.tolist()),
            }
        )
    return pd.DataFrame(rows)


def duplicate_literal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze duplicated literals, including code/category ambiguity."""
    grouped = (
        df.groupby("Literal")
        .agg(
            row_count=("Literal", "size"),
            unique_codes=("Code", "nunique"),
            unique_y_categories=("y_category", "nunique"),
            codes=("Code", lambda values: " | ".join(sorted(set(map(str, values)))[:20])),
            y_categories=(
                "y_category",
                lambda values: " | ".join(sorted(set(map(str, values)))[:20]),
            ),
        )
        .reset_index()
    )
    grouped["is_duplicate_literal"] = grouped["row_count"] > 1
    grouped["same_literal_multiple_codes"] = grouped["unique_codes"] > 1
    grouped["same_literal_multiple_y_categories"] = grouped["unique_y_categories"] > 1
    return grouped.sort_values(
        ["same_literal_multiple_y_categories", "same_literal_multiple_codes", "row_count"],
        ascending=[False, False, False],
    )


def text_pattern_summary(train_df: pd.DataFrame, leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """Compare text-pattern prevalence in train and leaderboard literals."""
    pattern_cols = [
        "has_digit",
        "is_all_upper",
        "has_slash",
        "has_hyphen",
        "has_parentheses",
        "has_accent",
        "has_punctuation",
        "has_measurement_like",
    ]
    rows = []
    for name, frame in [("train", train_df), ("leaderboard", leaderboard_df)]:
        for col in pattern_cols:
            rows.append(
                {
                    "dataset": name,
                    "pattern": col,
                    "count": int(frame[col].sum()),
                    "share": float(frame[col].mean()),
                }
            )
    return pd.DataFrame(rows)


def normalization_collision_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Measure whether lowercasing/accent stripping collapses distinct literals."""
    raw_unique = df["Literal"].nunique()
    lower_unique = df["literal_lower"].nunique()
    keep_accents_unique = df["literal_norm_keep_accents"].nunique()
    no_accents_unique = df["literal_norm_no_accents"].nunique()
    collisions = (
        df.groupby("literal_norm_no_accents")["Literal"]
        .nunique()
        .reset_index(name="raw_literal_variants")
    )
    collisions = collisions[collisions["raw_literal_variants"] > 1]
    ambiguous_after_norm = (
        df.groupby("literal_norm_no_accents")["y_category"]
        .nunique()
        .reset_index(name="y_categories_after_norm")
    )
    ambiguous_after_norm = ambiguous_after_norm[
        ambiguous_after_norm["y_categories_after_norm"] > 1
    ]
    return {
        "raw_unique_literals": int(raw_unique),
        "lowercase_unique_literals": int(lower_unique),
        "normalized_keep_accents_unique_literals": int(keep_accents_unique),
        "normalized_no_accents_unique_literals": int(no_accents_unique),
        "normalization_collision_keys": int(len(collisions)),
        "normalized_forms_with_multiple_categories": int(len(ambiguous_after_norm)),
    }


def train_leaderboard_shift_summary(train_df: pd.DataFrame, leaderboard_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize simple train/leaderboard shift signals available without labels."""
    train_norm = set(train_df["literal_norm_no_accents"])
    lead_norm = set(leaderboard_df["literal_norm_no_accents"])
    overlap = train_norm & lead_norm
    return {
        "train_rows": int(len(train_df)),
        "leaderboard_rows": int(len(leaderboard_df)),
        "train_unique_normalized_literals": int(len(train_norm)),
        "leaderboard_unique_normalized_literals": int(len(lead_norm)),
        "leaderboard_normalized_literals_seen_in_train": int(len(overlap)),
        "leaderboard_seen_in_train_share": float(len(overlap) / len(lead_norm)) if lead_norm else 0.0,
        "train_mean_chars": float(train_df["char_len"].mean()),
        "leaderboard_mean_chars": float(leaderboard_df["char_len"].mean()),
        "train_mean_whitespace_tokens": float(train_df["whitespace_tokens"].mean()),
        "leaderboard_mean_whitespace_tokens": float(leaderboard_df["whitespace_tokens"].mean()),
    }


def build_eda_key_findings(
    train_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    category_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    shift: dict[str, Any],
    normalization: dict[str, Any],
) -> pd.DataFrame:
    """Create a compact table of EDA conclusions for the report."""
    top_category = category_df.sort_values("count", ascending=False).iloc[0]
    bottom_category = category_df.sort_values("count", ascending=True).iloc[0]
    rows = [
        {
            "topic": "Dataset size",
            "finding": f"{len(train_df)} training rows and {len(leaderboard_df)} leaderboard rows.",
            "modeling_implication": "Validation can be meaningful, but rare classes still need careful reporting.",
        },
        {
            "topic": "Label space",
            "finding": f"{train_df['y_category'].nunique()} y_category labels and {train_df['Code'].nunique()} unique full ICD codes.",
            "modeling_implication": "The target is broad category prediction, not full ICD-code assignment.",
        },
        {
            "topic": "Class imbalance",
            "finding": f"Top category {top_category['y_category']} has {int(top_category['count'])} rows; bottom category {bottom_category['y_category']} has {int(bottom_category['count'])} rows.",
            "modeling_implication": "Accuracy will be dominated by common categories; macro metrics and error analysis are needed.",
        },
        {
            "topic": "Short context",
            "finding": f"Median train literal length is {train_df['whitespace_tokens'].median():.0f} whitespace tokens.",
            "modeling_implication": "Models receive little context, so ambiguity and abbreviation handling matter.",
        },
        {
            "topic": "Duplicate literals",
            "finding": f"{int(duplicate_df['is_duplicate_literal'].sum())} literals appear more than once.",
            "modeling_implication": "Exact-match and leakage-aware splits must be considered.",
        },
        {
            "topic": "Ambiguous literals",
            "finding": f"{int(duplicate_df['same_literal_multiple_y_categories'].sum())} literals map to multiple y_category labels.",
            "modeling_implication": "Some errors may reflect missing clinical context, not only weak modeling.",
        },
        {
            "topic": "Normalization",
            "finding": f"Accent/case normalization creates {normalization['normalization_collision_keys']} collision keys.",
            "modeling_implication": "Normalization helps matching but can collapse distinct surface forms.",
        },
        {
            "topic": "Train/leaderboard overlap",
            "finding": f"{shift['leaderboard_seen_in_train_share']:.1%} of unique normalized leaderboard literals appear in train.",
            "modeling_implication": "A hybrid exact-match plus generalization strategy is plausible.",
        },
    ]
    return pd.DataFrame(rows)

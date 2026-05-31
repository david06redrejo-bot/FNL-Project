"""Reusable EDA summaries for notebooks and scripts."""

from __future__ import annotations

import pandas as pd

from .preprocessing import extract_y_category


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


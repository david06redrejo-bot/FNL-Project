"""Text and annotation preprocessing for ICD category prediction."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def strip_html(text: str) -> str:
    """Remove simple HTML tags sometimes present in literals."""
    return re.sub(r"<[^>]+>", " ", str(text))


def normalize_literal(text: str, strip_accents: bool = True) -> str:
    """Normalize a clinical literal for classical lexical models."""
    text = strip_html(text).lower().strip()
    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    else:
        text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^a-z0-9áéíóúüñçàèìòù\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_y_category(code: str) -> str:
    """Extract the first ICD code character as an uppercase category."""
    return str(code)[0].upper()


def build_category_dataset(
    codification_df: pd.DataFrame,
    literal_col: str = "Literal",
    code_col: str = "Code",
) -> pd.DataFrame:
    """Create one `(Literal, y_category)` row per unique literal.

    If a literal maps to multiple categories, the most frequent category is kept.
    This choice must be reported because it compresses genuine annotation
    ambiguity into a single-label training target.
    """
    df = codification_df[[literal_col, code_col]].dropna().copy()
    df["y_category"] = df[code_col].map(extract_y_category)
    grouped = (
        df.groupby(literal_col)["y_category"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index()
    )
    grouped.columns = ["Literal", "y_category"]
    grouped["normalized_literal"] = grouped["Literal"].map(normalize_literal)
    return grouped


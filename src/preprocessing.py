"""Text and annotation preprocessing for ICD category prediction."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


def strip_html(text: str) -> str:
    """Remove simple HTML tags sometimes present in literals."""
    return re.sub(r"<[^>]+>", " ", str(text))


def _safe_text(text: Any) -> str:
    """Convert input to text while treating null-like values as empty strings."""
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    return str(text)


def clean_required(text: Any) -> str:
    """Required light preprocessing for the final RoBERTa pipeline.

    This deliberately preserves case, accents, punctuation, digits, and
    abbreviations because the selected backbone tokenizer is pretrained on
    Spanish biomedical/clinical text.
    """
    text = _safe_text(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_expand_spaces_only(text: Any) -> str:
    """Alias for the required whitespace-only cleanup."""
    return clean_required(text)


def clean_lowercase(text: Any) -> str:
    """Ablation: apply required cleaning and lowercase."""
    return clean_required(text).lower()


def clean_remove_accents(text: Any) -> str:
    """Ablation: apply required cleaning and remove diacritical marks."""
    cleaned = clean_required(text)
    decomposed = unicodedata.normalize("NFD", cleaned)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def clean_remove_punctuation(text: Any) -> str:
    """Ablation: apply required cleaning and remove punctuation symbols."""
    cleaned = clean_required(text)
    cleaned = re.sub(r"[^\w\sáéíóúüñçàèìòùÁÉÍÓÚÜÑÇÀÈÌÒÙ]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_text_pattern_features(text: Any) -> dict[str, bool | int | str]:
    """Extract text-pattern features used to reason about preprocessing risk."""
    cleaned = clean_required(text)
    return {
        "text": cleaned,
        "char_len": len(cleaned),
        "whitespace_tokens": len(cleaned.split()) if cleaned else 0,
        "has_digit": bool(re.search(r"\d", cleaned)),
        "is_all_upper": bool(re.search(r"[A-ZÁÉÍÓÚÜÑÇ]", cleaned)) and cleaned == cleaned.upper(),
        "has_slash": "/" in cleaned,
        "has_hyphen": "-" in cleaned,
        "has_parentheses": bool(re.search(r"[()]", cleaned)),
        "has_accent": _has_accent(cleaned),
        "has_punctuation": bool(
            re.search(r"[^\w\sáéíóúüñçàèìòùÁÉÍÓÚÜÑÇÀÈÌÒÙ]", cleaned)
        ),
        "has_measurement_like": bool(
            re.search(
                r"(?:\d+\s*(?:mg|ml|cm|mm|kg|g|mcg|ui|%|x|/|\+)|\d+[.,]\d+)",
                cleaned,
                flags=re.IGNORECASE,
            )
        ),
    }


def _has_accent(text: str) -> bool:
    """Return True if a string contains a decomposable accent mark."""
    decomposed = unicodedata.normalize("NFD", str(text))
    return any(unicodedata.category(ch) == "Mn" for ch in decomposed)


def compare_preprocessing_effects(
    df: pd.DataFrame,
    literal_col: str = "Literal",
    max_examples: int | None = None,
) -> pd.DataFrame:
    """Compare preprocessing variants without choosing them as defaults."""
    source = df if max_examples is None else df.head(max_examples)
    result = source[[literal_col]].copy()
    result["required_clean"] = result[literal_col].map(clean_required)
    result["lowercase"] = result[literal_col].map(clean_lowercase)
    result["remove_accents"] = result[literal_col].map(clean_remove_accents)
    result["remove_punctuation"] = result[literal_col].map(clean_remove_punctuation)
    result["required_changed"] = result[literal_col].map(_safe_text) != result["required_clean"]
    result["lowercase_changed"] = result["required_clean"] != result["lowercase"]
    result["accent_removal_changed"] = result["required_clean"] != result["remove_accents"]
    result["punctuation_removal_changed"] = (
        result["required_clean"] != result["remove_punctuation"]
    )
    return result


def normalize_literal(text: str, strip_accents: bool = True) -> str:
    """Normalize a clinical literal for classical lexical models."""
    text = strip_html(clean_required(text)).lower().strip()
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

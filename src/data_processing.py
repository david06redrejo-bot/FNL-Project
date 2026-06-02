"""
Data processing utilities for the automated ICD coding project.

The notebooks use these helpers to clean short Spanish/Catalan clinical
literals, extract ICD categories, and prepare train/validation splits.
"""

import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean a clinical literal while preserving accented Unicode letters.

    This variant is useful when we want light cleanup but do not want to strip
    Spanish/Catalan accents.
    """
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Normalize a clinical literal for the TF-IDF + SVM pipeline.

    Steps:
        1. Lowercase.
        2. Strip HTML tags.
        3. Strip diacritics, e.g. accented letters become plain ASCII letters.
        4. Keep only ASCII letters, digits, and whitespace.
        5. Collapse whitespace.
    """
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_texts(texts) -> list:
    """Apply clean_text to a list or Series of strings."""
    return [clean_text(text) for text in texts]


def normalize_texts(texts) -> list:
    """Apply normalize_text to a list or Series of strings."""
    return [normalize_text(text) for text in texts]


def extract_category(code: str) -> str:
    """
    Extract the ICD category from a code, using its first character.

    Examples:
        "J9809" -> "J"
        "07CP0ZZ" -> "0"
        "N801" -> "N"
    """
    return str(code)[0].upper()


def prepare_category_dataset(
    df: pd.DataFrame,
    literal_col: str = "Literal",
    code_col: str = "Code",
) -> pd.DataFrame:
    """
    Prepare a single-label dataset for ICD category classification.

    If a literal appears with multiple categories, the most frequent category
    is kept so the task matches the required single-label submission format.
    """
    df = df.copy()
    df["y_category"] = df[code_col].apply(extract_category)

    unique_cats = sorted(df["y_category"].unique())
    print(f"Total rows: {len(df):,}")
    print(f"Unique categories: {len(unique_cats)} -> {unique_cats}")

    grouped = (
        df.groupby(literal_col)["y_category"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index()
    )
    grouped.columns = ["Literal", "y_category"]

    print(f"Unique literals: {len(grouped):,}")
    print("Category distribution:")
    print(grouped["y_category"].value_counts().sort_index().to_string())

    return grouped


def split_category_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split a category dataset into stratified train and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        df["Literal"].tolist(),
        df["y_category"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["y_category"],
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    return X_train, X_val, y_train, y_val

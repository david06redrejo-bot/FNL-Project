"""Dataset construction and splitting utilities."""

from __future__ import annotations

from sklearn.model_selection import train_test_split


def stratified_literal_split(df, test_size: float = 0.2, random_state: int = 42):
    """Return stratified train/validation splits for category prediction."""
    return train_test_split(
        df["Literal"],
        df["y_category"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["y_category"],
    )


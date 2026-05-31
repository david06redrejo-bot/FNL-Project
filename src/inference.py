"""Prediction and submission helpers."""

from __future__ import annotations

import pandas as pd


def build_submission(
    leaderboard_df: pd.DataFrame,
    y_pred,
    id_col: str = "id",
) -> pd.DataFrame:
    """Build the exact Kaggle submission contract: `id,y_category` only."""
    output = pd.DataFrame({"id": leaderboard_df[id_col].values, "y_category": y_pred})
    output["y_category"] = output["y_category"].astype(str).str.upper()
    return output[["id", "y_category"]]


def build_detailed_predictions(
    source_df: pd.DataFrame,
    y_pred,
    y_true=None,
    probabilities=None,
    labels: list[str] | None = None,
    id_col: str | None = None,
    literal_col: str = "Literal_required_clean",
) -> pd.DataFrame:
    """Build detailed predictions for analysis, not for Kaggle submission."""
    output = pd.DataFrame()
    if id_col and id_col in source_df.columns:
        output["id"] = source_df[id_col].values
    if literal_col in source_df.columns:
        output["Literal"] = source_df[literal_col].values
    elif "Literal" in source_df.columns:
        output["Literal"] = source_df["Literal"].values
    if y_true is not None:
        output["y_true"] = y_true
    output["y_pred"] = y_pred

    if probabilities is not None and labels is not None:
        for idx, label in enumerate(labels):
            output[f"proba_{label}"] = probabilities[:, idx]
    return output

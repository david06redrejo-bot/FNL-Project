"""Prediction and submission helpers."""

from __future__ import annotations

import pandas as pd


def build_submission(
    leaderboard_df: pd.DataFrame,
    y_pred,
    include_literal: bool = True,
    id_col: str = "id",
    literal_col: str = "Literal",
) -> pd.DataFrame:
    """Build a Kaggle-style submission dataframe.

    The project keeps `Literal` in intermediate submissions for traceability.
    If Kaggle requires only `id,y_category`, call with `include_literal=False`.
    """
    output = pd.DataFrame({"id": leaderboard_df[id_col].values, "y_category": y_pred})
    if include_literal:
        output.insert(1, "Literal", leaderboard_df[literal_col].values)
    output["y_category"] = output["y_category"].astype(str).str.upper()
    return output


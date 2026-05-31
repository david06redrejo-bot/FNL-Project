"""Lightweight interpretability and error-analysis helpers.

These helpers avoid claiming that probabilities or attention weights fully
explain a clinical decision. They provide practical diagnostics: confidence,
margin, confusion pairs, and representative examples.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def probability_columns(df: pd.DataFrame) -> list[str]:
    """Return probability columns from a detailed prediction file."""
    return [col for col in df.columns if col.startswith("proba_")]


def add_confidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence, margin, top labels, and correctness columns."""
    result = df.copy()
    cols = probability_columns(result)
    if not cols:
        result["confidence"] = np.nan
        result["confidence_margin"] = np.nan
        result["top1_label"] = result["y_pred"].astype(str).str.upper()
        result["top2_label"] = pd.NA
    else:
        labels = [col.replace("proba_", "", 1) for col in cols]
        proba = result[cols].to_numpy(dtype=float)
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = np.divide(proba, row_sums, out=np.zeros_like(proba), where=row_sums != 0)
        order = np.argsort(proba, axis=1)[:, ::-1]
        result["confidence"] = proba[np.arange(len(result)), order[:, 0]]
        result["top1_label"] = [labels[idx] for idx in order[:, 0]]
        result["top2_label"] = [labels[idx] for idx in order[:, 1]]
        result["top2_confidence"] = proba[np.arange(len(result)), order[:, 1]]
        result["confidence_margin"] = result["confidence"] - result["top2_confidence"]
    if "y_true" in result.columns:
        result["is_correct"] = result["y_true"].astype(str).str.upper() == result["y_pred"].astype(str).str.upper()
    return result


def build_error_examples(predictions: pd.DataFrame, n_per_group: int = 12) -> pd.DataFrame:
    """Collect representative correct/wrong high/low confidence examples."""
    df = add_confidence_columns(predictions)
    rows = []
    groups = [
        ("correct_high_confidence", df[df["is_correct"]].sort_values(["confidence", "confidence_margin"], ascending=False)),
        ("correct_low_confidence", df[df["is_correct"]].sort_values(["confidence", "confidence_margin"], ascending=True)),
        ("wrong_high_confidence", df[~df["is_correct"]].sort_values(["confidence", "confidence_margin"], ascending=False)),
        ("wrong_low_confidence", df[~df["is_correct"]].sort_values(["confidence", "confidence_margin"], ascending=True)),
    ]
    for group_name, group_df in groups:
        selected = group_df.head(n_per_group).copy()
        selected["example_group"] = group_name
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    keep_cols = [
        "example_group",
        "Literal",
        "y_true",
        "y_pred",
        "confidence",
        "confidence_margin",
        "top2_label",
        "top2_confidence",
        "is_correct",
    ]
    return pd.concat(rows, ignore_index=True)[keep_cols]


def build_top_confusions(predictions: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return most frequent true/predicted error pairs."""
    df = predictions.copy()
    wrong = df[df["y_true"].astype(str).str.upper() != df["y_pred"].astype(str).str.upper()]
    if wrong.empty:
        return pd.DataFrame(columns=["y_true", "y_pred", "count", "example_literals"])
    grouped = (
        wrong.groupby(["y_true", "y_pred"])
        .agg(
            count=("Literal", "size"),
            example_literals=("Literal", lambda values: " || ".join(values.astype(str).head(4))),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return grouped


def heuristic_error_reason(row: pd.Series) -> str:
    """Attach a cautious, heuristic possible error reason for report review."""
    literal = str(row.get("Literal", ""))
    reasons = []
    if len(literal.split()) <= 2:
        reasons.append("short literal / insufficient context")
    if re.search(r"\b(no|sin|nega|descarta)\b", literal, flags=re.IGNORECASE):
        reasons.append("possible negation or exclusion cue")
    if re.search(r"\b[A-ZÁÉÍÓÚÜÑ]{2,}\b", literal):
        reasons.append("abbreviation or uppercase clinical shorthand")
    if re.search(r"\d", literal):
        reasons.append("digit or measurement-like signal")
    if row.get("confidence", 0) >= 0.7 and not bool(row.get("is_correct", False)):
        reasons.append("high-confidence error / possible ambiguous mapping")
    if row.get("confidence_margin", 0) < 0.15:
        reasons.append("low confidence margin / similar categories")
    if not reasons:
        reasons.append("broad ICD prefix ambiguity")
    return "; ".join(reasons)

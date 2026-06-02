"""
Evaluation utilities for ICD category classification.

The official target is strict accuracy on the first ICD character, but the
notebooks also report F1/precision/recall to make class imbalance visible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_category_metrics(y_true, y_pred):
    """Calculate metrics for single-label ICD category classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def print_classification_report(y_true, y_pred):
    """Print sklearn's per-class classification report."""
    print(classification_report(y_true, y_pred, zero_division=0))


def print_comparison_table(results_dict):
    """Print and return a model-comparison table."""
    df = pd.DataFrame(results_dict).T
    print("\n--- Model comparison ---")
    print(df.to_string())
    return df


def plot_comparison(results_dict, save_path=None):
    """Plot model metrics and optionally save the figure."""
    df = pd.DataFrame(results_dict).T

    plot_cols = [
        col for col in ["accuracy", "weighted_f1", "macro_f1"] if col in df.columns
    ]
    if not plot_cols:
        plot_cols = df.columns.tolist()

    df[plot_cols].plot(kind="bar", figsize=(10, 5), title="Model comparison")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


def generate_submission(
    leaderboard_df,
    y_pred_categories,
    output_path: str,
    id_col: str = "id",
    literal_col: str = "Literal",
):
    """
    Build and validate the required submission CSV.

    Required columns, in order:
        id
        Literal
        y_category
    """
    submission_df = pd.DataFrame(
        {
            "id": leaderboard_df[id_col].values,
            "Literal": leaderboard_df[literal_col].values,
            "y_category": y_pred_categories,
        }
    )

    submission_df["y_category"] = submission_df["y_category"].fillna("null")
    submission_df["y_category"] = submission_df["y_category"].replace("", "null")
    submission_df = submission_df[["id", "Literal", "y_category"]]

    assert len(submission_df) == len(leaderboard_df)
    assert submission_df["id"].notna().all()
    assert submission_df["Literal"].notna().all()
    assert submission_df["y_category"].notna().all()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(f"  Total rows: {len(submission_df):,}")
    print(f"  Unique categories predicted: {submission_df['y_category'].nunique()}")
    print(f"  Empty values filled as null: {(submission_df['y_category'] == 'null').sum()}")
    return submission_df

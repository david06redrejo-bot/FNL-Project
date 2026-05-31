"""Final evaluation utilities for the ICD category project.

The functions in this module read completed experiment artifacts, compare model
versions, and generate the final analysis tables and figures used by notebook
07 and the report. They do not train models and they do not use leaderboard
labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .interpretability import (
    add_confidence_columns,
    build_error_examples,
    build_top_confusions,
    heuristic_error_reason,
)
from .paths import METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR


FINAL_MODEL = "v09_ensemble"

MODEL_METRIC_FILES = {
    "v00_majority": METRICS_DIR / "v00_majority_baseline_metrics.json",
    "v01_char_tfidf_logreg": METRICS_DIR / "v01_tfidf_char_logreg_metrics.json",
    "v02_word_tfidf_svm": METRICS_DIR / "v02_tfidf_word_svm_metrics.json",
    "v03_similarity_retrieval": METRICS_DIR / "v03_similarity_retrieval_baseline_metrics.json",
    "v04_roberta_cls": METRICS_DIR / "v04_roberta_cls_metrics.json",
    "v05_roberta_mean": METRICS_DIR / "v05_roberta_mean_metrics.json",
    "v06_class_weighted": METRICS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_metrics.json",
    "v06_focal_gamma1": METRICS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma1_metrics.json",
    "v06_focal_gamma2": METRICS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_metrics.json",
    "v07_mean_tuning": METRICS_DIR / "v07_roberta_mean_tuning_metrics.json",
    "v08_safe_dedupe": METRICS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_metrics.json",
    "v08_weighted_sampler": METRICS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_metrics.json",
    "v09_ensemble": METRICS_DIR / "v09_ensemble_metrics.json",
}

MODEL_FAMILIES = {
    "v00_majority": "sanity baseline",
    "v01_char_tfidf_logreg": "classical baseline",
    "v02_word_tfidf_svm": "classical baseline",
    "v03_similarity_retrieval": "retrieval baseline",
    "v04_roberta_cls": "RoBERTa",
    "v05_roberta_mean": "RoBERTa",
    "v06_class_weighted": "advanced RoBERTa",
    "v06_focal_gamma1": "advanced RoBERTa",
    "v06_focal_gamma2": "advanced RoBERTa",
    "v07_mean_tuning": "advanced RoBERTa",
    "v08_safe_dedupe": "safe data strategy",
    "v08_weighted_sampler": "safe data strategy",
    "v09_ensemble": "ensemble",
}

TRAINING_HISTORY_FILES = {
    "v04_roberta_cls": REPORTS_DIR / "figures" / "fig_11_roberta_cls_training_curves.png",
    "v05_roberta_mean": REPORTS_DIR / "figures" / "fig_12_roberta_mean_training_curves.png",
    "v07_mean_tuning": REPORTS_DIR / "figures" / "v07_roberta_mean_tuning_c_recommended_32_lr2e5_warmup006_clip_training_curves.png",
}


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_comparison(metric_files: dict[str, Path] | None = None) -> pd.DataFrame:
    """Load metrics for the final comparison table."""
    metric_files = metric_files or MODEL_METRIC_FILES
    rows = []
    for model_id, path in metric_files.items():
        if not path.exists():
            rows.append(
                {
                    "model_id": model_id,
                    "family": MODEL_FAMILIES.get(model_id, "unknown"),
                    "status": "missing_metrics",
                }
            )
            continue
        metrics = load_json(path)
        rows.append(
            {
                "model_id": model_id,
                "family": MODEL_FAMILIES.get(model_id, "unknown"),
                "status": "available",
                "accuracy": metrics.get("accuracy"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "top_2_accuracy": metrics.get("top_2_accuracy"),
                "top_3_accuracy": metrics.get("top_3_accuracy"),
                "top_5_accuracy": metrics.get("top_5_accuracy"),
                "log_loss": metrics.get("log_loss"),
                "best_epoch": metrics.get("best_epoch"),
                "selected_candidate": metrics.get("selected_candidate"),
                "selected_recipe": metrics.get("selected_recipe"),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=[False, False], na_position="last")


def load_final_predictions(final_model: str = FINAL_MODEL) -> pd.DataFrame:
    """Load final validation predictions with confidence columns."""
    path = PREDICTIONS_DIR / f"{final_model}_val_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Final validation predictions not found: {path}")
    return add_confidence_columns(pd.read_csv(path))


def load_final_per_class(final_model: str = FINAL_MODEL) -> pd.DataFrame:
    """Load final per-class metrics."""
    path = METRICS_DIR / f"{final_model}_per_class_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Final per-class metrics not found: {path}")
    return pd.read_csv(path).sort_values("y_category")


def final_confusion_matrix(final_model: str = FINAL_MODEL) -> tuple[pd.DataFrame, list[str]]:
    """Return the final model confusion matrix as a dataframe."""
    metrics = load_json(METRICS_DIR / f"{final_model}_metrics.json")
    labels = metrics["labels"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    return matrix, labels


def save_final_tables() -> dict[str, Path]:
    """Save final evaluation tables."""
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_experiment_comparison()
    comparison_path = tables_dir / "final_experiment_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    per_class = load_final_per_class()
    per_class_path = tables_dir / "final_per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False)

    predictions = load_final_predictions()
    error_examples = build_error_examples(predictions)
    error_examples["possible_error_reason"] = error_examples.apply(heuristic_error_reason, axis=1)
    error_examples_path = tables_dir / "final_error_examples.csv"
    error_examples.to_csv(error_examples_path, index=False)

    top_confusions = build_top_confusions(predictions)
    top_confusions_path = tables_dir / "final_top_confusions.csv"
    top_confusions.to_csv(top_confusions_path, index=False)
    return {
        "comparison": comparison_path,
        "per_class": per_class_path,
        "error_examples": error_examples_path,
        "top_confusions": top_confusions_path,
    }


def _style() -> None:
    """Apply plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Save bar chart comparing model versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = comparison[comparison["status"].eq("available")].copy()
    plot_df = plot_df.sort_values("accuracy", ascending=True)
    _style()
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["accuracy"], height=0.18, label="accuracy", color="#4c78a8")
    ax.barh(y, plot_df["macro_f1"], height=0.18, label="macro F1", color="#f58518")
    ax.barh(y + 0.18, plot_df["weighted_f1"], height=0.18, label="weighted F1", color="#54a24b")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_id"])
    ax.set_xlim(0, max(0.65, float(plot_df["accuracy"].max()) + 0.05))
    ax.set_xlabel("Validation score")
    ax.set_title("Model comparison on the shared validation split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: pd.DataFrame, path: Path) -> None:
    """Save final confusion matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="Blues", cbar_kws={"label": "validation rows"}, ax=ax)
    ax.set_title("Final model confusion matrix")
    ax.set_xlabel("Predicted y_category")
    ax.set_ylabel("True y_category")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_recall(per_class: pd.DataFrame, path: Path) -> None:
    """Save per-class recall chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = per_class.sort_values("recall")
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="y_category", y="recall", color="#4c78a8", ax=ax)
    ax.set_title("Final model recall by ICD category")
    ax.set_xlabel("y_category ordered by recall")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_correct_vs_wrong(predictions: pd.DataFrame, path: Path) -> None:
    """Save confidence distribution split by correctness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = predictions.copy()
    plot_df["prediction_status"] = np.where(plot_df["is_correct"], "correct", "wrong")
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(data=plot_df, x="confidence", hue="prediction_status", common_norm=False, ax=axes[0])
    axes[0].set_title("Top-1 confidence")
    axes[0].set_xlabel("Confidence")
    sns.kdeplot(data=plot_df, x="confidence_margin", hue="prediction_status", common_norm=False, ax=axes[1])
    axes[1].set_title("Top-1 minus top-2 confidence margin")
    axes[1].set_xlabel("Confidence margin")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(path: Path) -> None:
    """Save a compact training-curve comparison for best model families."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history_files = {
        "v04_roberta_cls": Path("outputs/logs/v04_roberta_cls_history.csv"),
        "v05_roberta_mean": Path("outputs/logs/v05_roberta_mean_history.csv"),
        "v08_safe_dedupe": Path("outputs/logs/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_history.csv"),
        "v08_weighted_sampler": Path("outputs/logs/v08_roberta_mean_augmented_weighted_random_sampler_history.csv"),
    }
    frames = []
    for model_id, history_path in history_files.items():
        if history_path.exists():
            frame = pd.read_csv(history_path)
            frame["model_id"] = model_id
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No training history files found for final curve plot.")
    history = pd.concat(frames, ignore_index=True)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.lineplot(data=history, x="epoch", y="val_accuracy", hue="model_id", ax=axes[0])
    axes[0].set_title("Validation accuracy during training")
    axes[0].set_ylabel("Validation accuracy")
    sns.lineplot(data=history, x="epoch", y="val_loss", hue="model_id", ax=axes[1], legend=False)
    axes[1].set_title("Validation loss during training")
    axes[1].set_ylabel("Validation loss")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_final_figures() -> dict[str, Path]:
    """Save all final evaluation figures."""
    figures_dir = REPORTS_DIR / "figures"
    comparison = load_experiment_comparison()
    per_class = load_final_per_class()
    predictions = load_final_predictions()
    matrix, _ = final_confusion_matrix()
    paths = {
        "model_comparison": figures_dir / "fig_10_model_comparison.png",
        "confusion_matrix": figures_dir / "fig_11_final_confusion_matrix.png",
        "per_class_recall": figures_dir / "fig_12_per_class_recall.png",
        "confidence": figures_dir / "fig_13_confidence_correct_vs_wrong.png",
        "training_curves": figures_dir / "fig_14_training_curves.png",
    }
    plot_model_comparison(comparison, paths["model_comparison"])
    plot_confusion_matrix(matrix, paths["confusion_matrix"])
    plot_per_class_recall(per_class, paths["per_class_recall"])
    plot_confidence_correct_vs_wrong(predictions, paths["confidence"])
    plot_training_curves(paths["training_curves"])
    return paths


def run_final_evaluation() -> dict[str, dict[str, Path]]:
    """Generate final evaluation tables and figures."""
    table_paths = save_final_tables()
    figure_paths = save_final_figures()
    return {"tables": table_paths, "figures": figure_paths}


def main() -> None:
    """CLI entry point."""
    paths = run_final_evaluation()
    print("Final evaluation artifacts:")
    for group, group_paths in paths.items():
        for name, path in group_paths.items():
            print(f"- {group}.{name}: {path}")


if __name__ == "__main__":
    main()

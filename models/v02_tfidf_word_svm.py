"""v02: word-level TF-IDF baselines with LinearSVC and LogisticRegression.

This model version tests whether clinical literals contain enough lexical
signal at the word level. It complements the character n-gram model by checking
how far explicit word and word-pair evidence can go before moving to
Transformer representations.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

from src.inference import build_detailed_predictions, build_submission
from src.metrics import compute_full_metrics, per_class_metrics_table
from src.paths import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    ensure_project_dirs,
)
from src.preprocessing import normalize_literal
from src.reporting import write_run_summary
from src.training import (
    ModelRunConfig,
    load_required_clean_data,
    predict_proba_if_available,
    split_train_validation,
)
from src.utils import save_json, set_seed


VERSION_NAME = "v02_tfidf_word_svm"
SEED = 42


def make_pipeline(
    ngram_range: tuple[int, int],
    classifier_name: str,
    min_df: int,
    sublinear_tf: bool,
    seed: int,
) -> Pipeline:
    """Create a word-level TF-IDF baseline pipeline."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        preprocessor=normalize_literal,
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    if classifier_name == "logreg":
        classifier = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        )
    elif classifier_name == "linear_svc":
        classifier = LinearSVC(C=1.0, max_iter=10_000, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def candidate_grid() -> list[dict[str, Any]]:
    """Grid for the word-level baseline."""
    rows = []
    for ngram_range in [(1, 1), (1, 2), (1, 3)]:
        for classifier_name in ["linear_svc", "logreg"]:
            for min_df in [1, 2]:
                rows.append(
                    {
                        "ngram_range": ngram_range,
                        "classifier": classifier_name,
                        "min_df": min_df,
                        "sublinear_tf": True,
                    }
                )
    return rows


def update_classical_comparison(v02_metrics: dict[str, Any]) -> tuple[Path, Path]:
    """Write a compact v00/v01/v02 comparison table and plot."""
    tables_dir = REPORTS_DIR / "tables"
    figures_dir = REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in [
        "v00_majority_baseline",
        "v01_tfidf_char_logreg",
    ]:
        metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )
    rows.append(
        {
            "model": VERSION_NAME,
            "accuracy": v02_metrics["accuracy"],
            "macro_f1": v02_metrics["macro_f1"],
            "weighted_f1": v02_metrics["weighted_f1"],
        }
    )
    comparison = pd.DataFrame(rows)
    comparison_path = tables_dir / "classical_baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_df = comparison.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Classical Baseline Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure_path = figures_dir / "fig_10_classical_baseline_comparison.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return comparison_path, figure_path


def main() -> None:
    """Run the word TF-IDF grid, save the best model, and generate outputs."""
    ensure_project_dirs()
    set_seed(SEED)
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = ModelRunConfig(
        version_name=VERSION_NAME,
        model_family="tfidf_word",
        seed=SEED,
    )
    train, leaderboard = load_required_clean_data()
    labels = sorted(train[config.target_col].astype(str).unique().tolist())
    train_df, val_df = split_train_validation(train, config)

    grid_rows = []
    trained_candidates: list[tuple[dict[str, Any], Pipeline, dict[str, Any]]] = []
    for candidate in candidate_grid():
        model = make_pipeline(
            ngram_range=candidate["ngram_range"],
            classifier_name=candidate["classifier"],
            min_df=candidate["min_df"],
            sublinear_tf=candidate["sublinear_tf"],
            seed=SEED,
        )
        model.fit(train_df[config.text_col], train_df[config.target_col])
        val_pred = model.predict(val_df[config.text_col])
        val_proba = predict_proba_if_available(model, val_df[config.text_col])
        metrics = compute_full_metrics(
            val_df[config.target_col],
            val_pred,
            labels=labels,
            y_proba=val_proba,
        )
        grid_rows.append(
            {
                "version_name": VERSION_NAME,
                "ngram_range": f"{candidate['ngram_range'][0]}-{candidate['ngram_range'][1]}",
                "classifier": candidate["classifier"],
                "min_df": candidate["min_df"],
                "sublinear_tf": candidate["sublinear_tf"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "log_loss": metrics.get("log_loss"),
            }
        )
        trained_candidates.append((candidate, model, metrics))

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    )
    grid_path = tables_dir / "v02_tfidf_word_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_index = grid_df.index[0]
    best_candidate, best_model, best_metrics = trained_candidates[best_index]

    val_pred = best_model.predict(val_df[config.text_col])
    val_proba = predict_proba_if_available(best_model, val_df[config.text_col])
    val_detailed = build_detailed_predictions(
        val_df,
        val_pred,
        y_true=val_df[config.target_col].values,
        probabilities=val_proba,
        labels=labels,
        literal_col=config.text_col,
    )
    val_predictions_path = PREDICTIONS_DIR / f"{VERSION_NAME}_val_predictions.csv"
    val_detailed.to_csv(val_predictions_path, index=False)

    metrics = dict(best_metrics)
    metrics["best_params"] = {
        "ngram_range": list(best_candidate["ngram_range"]),
        "classifier": best_candidate["classifier"],
        "min_df": best_candidate["min_df"],
        "sublinear_tf": best_candidate["sublinear_tf"],
    }
    metrics["grid_results_path"] = str(grid_path)
    metrics_path = METRICS_DIR / f"{VERSION_NAME}_metrics.json"
    save_json(metrics, metrics_path)

    per_class_path = METRICS_DIR / f"{VERSION_NAME}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)

    final_model = make_pipeline(
        ngram_range=best_candidate["ngram_range"],
        classifier_name=best_candidate["classifier"],
        min_df=best_candidate["min_df"],
        sublinear_tf=best_candidate["sublinear_tf"],
        seed=SEED,
    )
    final_model.fit(train[config.text_col], train[config.target_col])
    checkpoint_path = CHECKPOINTS_DIR / f"{VERSION_NAME}.joblib"
    joblib.dump(final_model, checkpoint_path)

    leaderboard_pred = final_model.predict(leaderboard[config.text_col])
    leaderboard_proba = predict_proba_if_available(final_model, leaderboard[config.text_col])
    leaderboard_detailed = build_detailed_predictions(
        leaderboard,
        leaderboard_pred,
        probabilities=leaderboard_proba,
        labels=labels,
        id_col="id",
        literal_col=config.text_col,
    )
    leaderboard_detailed_path = PREDICTIONS_DIR / f"{VERSION_NAME}_leaderboard_detailed.csv"
    leaderboard_detailed.to_csv(leaderboard_detailed_path, index=False)

    submission = build_submission(leaderboard, leaderboard_pred)
    submission_path = SUBMISSIONS_DIR / f"{VERSION_NAME}_submission.csv"
    submission.to_csv(submission_path, index=False)

    comparison_path, comparison_figure_path = update_classical_comparison(metrics)

    config_dict = asdict(config)
    config_dict["grid"] = [
        {
            "ngram_range": row["ngram_range"],
            "classifier": row["classifier"],
            "min_df": row["min_df"],
            "sublinear_tf": row["sublinear_tf"],
        }
        for row in grid_rows
    ]
    config_dict["selected_by"] = "highest validation accuracy; ties by macro_f1 and weighted_f1"
    config_dict["best_params"] = metrics["best_params"]
    config_path = LOGS_DIR / f"{VERSION_NAME}_config.json"
    save_json(config_dict, config_path)

    artifact_paths = {
        "config": str(config_path),
        "grid_results": str(grid_path),
        "metrics": str(metrics_path),
        "validation_predictions": str(val_predictions_path),
        "per_class_metrics": str(per_class_path),
        "model_artifact": str(checkpoint_path),
        "leaderboard_detailed": str(leaderboard_detailed_path),
        "submission": str(submission_path),
        "classical_comparison": str(comparison_path),
        "classical_comparison_figure": str(comparison_figure_path),
    }
    summary_path = write_run_summary(VERSION_NAME, config_dict, metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)

    print(
        {
            "version_name": VERSION_NAME,
            "best_params": metrics["best_params"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "artifacts": artifact_paths,
        }
    )


if __name__ == "__main__":
    main()

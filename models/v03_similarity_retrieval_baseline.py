"""v03 optional baseline: TF-IDF similarity retrieval for ICD categories.

This script implements an information-retrieval view of ICD coding. Instead of
learning a classifier directly, it retrieves the nearest known clinical literal
or ICD description and copies the retrieved item's category.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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
    RAW_DATA_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    ensure_project_dirs,
)
from src.preprocessing import normalize_literal
from src.reporting import write_run_summary
from src.training import ModelRunConfig, load_required_clean_data, split_train_validation
from src.utils import save_json, set_seed


VERSION_NAME = "v03_similarity_retrieval_baseline"
SEED = 42


def make_vectorizer(analyzer: str, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for retrieval."""
    if analyzer == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            preprocessor=normalize_literal,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            sublinear_tf=True,
        )
    if analyzer == "char_wb":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            preprocessor=normalize_literal,
            sublinear_tf=True,
        )
    raise ValueError(f"Unknown analyzer: {analyzer}")


def majority_vote(labels: list[str]) -> str:
    """Return deterministic majority label for nearest-neighbor labels."""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def knn_predict(
    index_texts: pd.Series,
    index_labels: pd.Series,
    query_texts: pd.Series,
    analyzer: str,
    ngram_range: tuple[int, int],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, NearestNeighbors]:
    """Predict by retrieving nearest training literals."""
    vectorizer = make_vectorizer(analyzer, ngram_range)
    index_matrix = vectorizer.fit_transform(index_texts)
    query_matrix = vectorizer.transform(query_texts)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(index_matrix)
    distances, indices = nn.kneighbors(query_matrix)
    labels = index_labels.astype(str).to_numpy()
    predictions = []
    for row_indices in indices:
        predictions.append(majority_vote(labels[row_indices].tolist()))
    similarities = 1.0 - distances
    return np.asarray(predictions), indices, similarities, vectorizer, nn


def description_predict(
    descriptions: pd.DataFrame,
    query_texts: pd.Series,
    analyzer: str,
    ngram_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, NearestNeighbors]:
    """Predict by retrieving the nearest ICD description."""
    vectorizer = make_vectorizer(analyzer, ngram_range)
    description_matrix = vectorizer.fit_transform(descriptions["Description_clean"])
    query_matrix = vectorizer.transform(query_texts)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(description_matrix)
    distances, indices = nn.kneighbors(query_matrix)
    labels = descriptions["y_category"].astype(str).to_numpy()
    predictions = labels[indices[:, 0]]
    similarities = 1.0 - distances
    return predictions, indices, similarities, vectorizer, nn


def load_icd_descriptions() -> pd.DataFrame | None:
    """Load ICD descriptions if the optional description CSV is available."""
    path = RAW_DATA_DIR / "icd_d_p_pairs.csv"
    if not path.exists():
        return None
    descriptions = pd.read_csv(path)
    required = {"Code", "Description"}
    if not required.issubset(descriptions.columns):
        return None
    descriptions = descriptions.dropna(subset=["Code", "Description"]).copy()
    descriptions["Code"] = descriptions["Code"].astype(str)
    descriptions["y_category"] = descriptions["Code"].str[0].str.upper()
    descriptions["Description_clean"] = descriptions["Description"].map(normalize_literal)
    descriptions = descriptions[descriptions["Description_clean"].str.len() > 0]
    return descriptions.reset_index(drop=True)


def build_neighbor_examples(
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    index_df: pd.DataFrame,
    method_name: str,
    literal_col: str,
    n: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build correct and wrong nearest-neighbor examples for analysis."""
    rows = []
    for row_id, val_row in val_df.reset_index(drop=True).iterrows():
        nearest_idx = int(neighbor_indices[row_id, 0])
        neighbor = index_df.iloc[nearest_idx]
        rows.append(
            {
                "method": method_name,
                "query_literal": val_row[literal_col],
                "true_y_category": val_row["y_category"],
                "predicted_y_category": y_pred[row_id],
                "nearest_text": neighbor[literal_col]
                if literal_col in neighbor.index
                else neighbor.get("Description_clean", ""),
                "nearest_code": neighbor.get("Code", ""),
                "nearest_y_category": neighbor.get("y_category", ""),
                "cosine_similarity": float(similarities[row_id, 0]),
                "is_correct": bool(val_row["y_category"] == y_pred[row_id]),
            }
        )
    examples = pd.DataFrame(rows)
    correct = examples[examples["is_correct"]].sort_values(
        "cosine_similarity", ascending=False
    ).head(n)
    wrong = examples[~examples["is_correct"]].sort_values(
        "cosine_similarity", ascending=False
    ).head(n)
    return correct, wrong


def update_comparison(metrics: dict[str, Any]) -> tuple[Path, Path]:
    """Update the classical baseline comparison table and plot."""
    tables_dir = REPORTS_DIR / "tables"
    figures_dir = REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in [
        "v00_majority_baseline",
        "v01_tfidf_char_logreg",
        "v02_tfidf_word_svm",
    ]:
        metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model_name,
                    "accuracy": data["accuracy"],
                    "macro_f1": data["macro_f1"],
                    "weighted_f1": data["weighted_f1"],
                }
            )
    rows.append(
        {
            "model": VERSION_NAME,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
    )
    comparison = pd.DataFrame(rows)
    comparison_path = tables_dir / "classical_baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    ax = comparison.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=20,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Classical and Retrieval Baseline Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure_path = figures_dir / "fig_10_classical_baseline_comparison.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return comparison_path, figure_path


def main() -> None:
    """Run training-example and ICD-description retrieval baselines."""
    ensure_project_dirs()
    set_seed(SEED)
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = ModelRunConfig(
        version_name=VERSION_NAME,
        model_family="similarity_retrieval",
        seed=SEED,
    )
    train, leaderboard = load_required_clean_data()
    labels = sorted(train[config.target_col].astype(str).unique().tolist())
    train_df, val_df = split_train_validation(train, config)

    grid_rows: list[dict[str, Any]] = []
    trained_candidates: list[dict[str, Any]] = []
    for analyzer, ngram_range in [("word", (1, 2)), ("char_wb", (3, 5))]:
        for k in [1, 3, 5]:
            y_pred, indices, similarities, vectorizer, nn = knn_predict(
                train_df[config.text_col],
                train_df[config.target_col],
                val_df[config.text_col],
                analyzer=analyzer,
                ngram_range=ngram_range,
                k=k,
            )
            metrics = compute_full_metrics(val_df[config.target_col], y_pred, labels=labels)
            method_name = f"train_literal_{analyzer}_{ngram_range[0]}-{ngram_range[1]}_k{k}"
            grid_rows.append(
                {
                    "method": method_name,
                    "retrieval_index": "training_literals",
                    "analyzer": analyzer,
                    "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                    "k": k,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )
            trained_candidates.append(
                {
                    "method": method_name,
                    "retrieval_index": "training_literals",
                    "analyzer": analyzer,
                    "ngram_range": ngram_range,
                    "k": k,
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "indices": indices,
                    "similarities": similarities,
                    "index_df": train_df,
                    "vectorizer": vectorizer,
                    "nn": nn,
                }
            )

    descriptions = load_icd_descriptions()
    if descriptions is not None:
        for analyzer, ngram_range in [("word", (1, 2)), ("char_wb", (3, 5))]:
            y_pred, indices, similarities, vectorizer, nn = description_predict(
                descriptions,
                val_df[config.text_col],
                analyzer=analyzer,
                ngram_range=ngram_range,
            )
            metrics = compute_full_metrics(val_df[config.target_col], y_pred, labels=labels)
            method_name = f"icd_description_{analyzer}_{ngram_range[0]}-{ngram_range[1]}_k1"
            grid_rows.append(
                {
                    "method": method_name,
                    "retrieval_index": "icd_descriptions",
                    "analyzer": analyzer,
                    "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                    "k": 1,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )
            description_index = descriptions.rename(
                columns={"Description_clean": config.text_col}
            )
            trained_candidates.append(
                {
                    "method": method_name,
                    "retrieval_index": "icd_descriptions",
                    "analyzer": analyzer,
                    "ngram_range": ngram_range,
                    "k": 1,
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "indices": indices,
                    "similarities": similarities,
                    "index_df": description_index,
                    "vectorizer": vectorizer,
                    "nn": nn,
                }
            )

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    )
    grid_path = tables_dir / "v03_similarity_retrieval_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_method = grid_df.iloc[0]["method"]
    best = next(candidate for candidate in trained_candidates if candidate["method"] == best_method)
    metrics = dict(best["metrics"])
    metrics["best_params"] = {
        "method": best["method"],
        "retrieval_index": best["retrieval_index"],
        "analyzer": best["analyzer"],
        "ngram_range": list(best["ngram_range"]),
        "k": best["k"],
    }
    metrics["grid_results_path"] = str(grid_path)
    metrics_path = METRICS_DIR / f"{VERSION_NAME}_metrics.json"
    save_json(metrics, metrics_path)

    val_detailed = build_detailed_predictions(
        val_df,
        best["y_pred"],
        y_true=val_df[config.target_col].values,
        literal_col=config.text_col,
    )
    val_predictions_path = PREDICTIONS_DIR / f"{VERSION_NAME}_val_predictions.csv"
    val_detailed.to_csv(val_predictions_path, index=False)

    per_class_path = METRICS_DIR / f"{VERSION_NAME}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)

    correct_examples, wrong_examples = build_neighbor_examples(
        val_df,
        best["y_pred"],
        best["indices"],
        best["similarities"],
        best["index_df"],
        best["method"],
        config.text_col,
    )
    correct_path = tables_dir / "v03_similarity_correct_neighbors.csv"
    wrong_path = tables_dir / "v03_similarity_wrong_neighbors.csv"
    correct_examples.to_csv(correct_path, index=False)
    wrong_examples.to_csv(wrong_path, index=False)

    if best["retrieval_index"] == "training_literals":
        final_pred, _, _, final_vectorizer, final_nn = knn_predict(
            train[config.text_col],
            train[config.target_col],
            leaderboard[config.text_col],
            analyzer=best["analyzer"],
            ngram_range=best["ngram_range"],
            k=best["k"],
        )
        artifact = {
            "retrieval_index": "training_literals",
            "vectorizer": final_vectorizer,
            "nearest_neighbors": final_nn,
            "index_texts": train[config.text_col].tolist(),
            "index_labels": train[config.target_col].astype(str).tolist(),
            "best_params": metrics["best_params"],
        }
    else:
        if descriptions is None:
            raise RuntimeError("Best method requires ICD descriptions, but descriptions are missing.")
        final_pred, _, _, final_vectorizer, final_nn = description_predict(
            descriptions,
            leaderboard[config.text_col],
            analyzer=best["analyzer"],
            ngram_range=best["ngram_range"],
        )
        artifact = {
            "retrieval_index": "icd_descriptions",
            "vectorizer": final_vectorizer,
            "nearest_neighbors": final_nn,
            "index_texts": descriptions["Description_clean"].tolist(),
            "index_labels": descriptions["y_category"].astype(str).tolist(),
            "best_params": metrics["best_params"],
        }

    checkpoint_path = CHECKPOINTS_DIR / f"{VERSION_NAME}.joblib"
    joblib.dump(artifact, checkpoint_path)

    leaderboard_detailed = build_detailed_predictions(
        leaderboard,
        final_pred,
        id_col="id",
        literal_col=config.text_col,
    )
    leaderboard_detailed_path = PREDICTIONS_DIR / f"{VERSION_NAME}_leaderboard_detailed.csv"
    leaderboard_detailed.to_csv(leaderboard_detailed_path, index=False)

    submission = build_submission(leaderboard, final_pred)
    submission_path = SUBMISSIONS_DIR / f"{VERSION_NAME}_submission.csv"
    submission.to_csv(submission_path, index=False)

    comparison_path, comparison_figure_path = update_comparison(metrics)

    config_dict = asdict(config)
    config_dict["description_file_available"] = descriptions is not None
    config_dict["selected_by"] = "highest validation accuracy; ties by macro_f1 and weighted_f1"
    config_dict["best_params"] = metrics["best_params"]
    config_dict["grid"] = grid_rows
    config_path = LOGS_DIR / f"{VERSION_NAME}_config.json"
    save_json(config_dict, config_path)

    artifact_paths = {
        "config": str(config_path),
        "grid_results": str(grid_path),
        "metrics": str(metrics_path),
        "validation_predictions": str(val_predictions_path),
        "per_class_metrics": str(per_class_path),
        "correct_neighbor_examples": str(correct_path),
        "wrong_neighbor_examples": str(wrong_path),
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

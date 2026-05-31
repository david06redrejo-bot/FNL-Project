"""v01: character TF-IDF + logistic regression baseline.

This baseline tests whether surface form evidence in clinical literals is
already strong enough before using neural pretrained models. Character n-grams
are useful for abbreviations, morphology, digits, punctuation, and compact
clinical fragments.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.training import ModelRunConfig, load_required_clean_data, split_train_validation
from src.utils import save_json, set_seed


VERSION_NAME = "v01_tfidf_char_logreg"
SEED = 42


def lowercase_literal(text: object) -> str:
    """Ablation-only lowercase variant; not the default preprocessing."""
    return normalize_literal(text).lower()


def make_pipeline(
    ngram_range: tuple[int, int],
    class_weight: str | None,
    preprocessing_variant: str,
    seed: int,
) -> Pipeline:
    """Build a char n-gram TF-IDF logistic regression pipeline."""
    preprocessor = lowercase_literal if preprocessing_variant == "lowercase" else normalize_literal
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    preprocessor=preprocessor,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def candidate_grid() -> list[dict[str, Any]]:
    """Small internal grid requested for the first traditional ML baseline."""
    rows = []
    for preprocessing_variant in ["required_clean", "lowercase"]:
        for ngram_range in [(2, 4), (3, 5), (2, 6)]:
            for class_weight in [None, "balanced"]:
                rows.append(
                    {
                        "preprocessing_variant": preprocessing_variant,
                        "ngram_range": ngram_range,
                        "class_weight": class_weight,
                    }
                )
    return rows


def extract_top_ngrams_per_class(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract the largest positive logistic-regression coefficients per class."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for class_index, label in enumerate(classifier.classes_):
        coefs = classifier.coef_[class_index]
        top_indices = coefs.argsort()[::-1][:top_n]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "y_category": label,
                    "rank": rank,
                    "char_ngram": feature_names[feature_index],
                    "coefficient": float(coefs[feature_index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run grid search, select the best model, and write all deliverables."""
    ensure_project_dirs()
    set_seed(SEED)
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = ModelRunConfig(
        version_name=VERSION_NAME,
        model_family="tfidf_char_logreg",
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
            class_weight=candidate["class_weight"],
            preprocessing_variant=candidate["preprocessing_variant"],
            seed=SEED,
        )
        model.fit(train_df[config.text_col], train_df[config.target_col])
        val_pred = model.predict(val_df[config.text_col])
        val_proba = model.predict_proba(val_df[config.text_col])
        metrics = compute_full_metrics(
            val_df[config.target_col],
            val_pred,
            labels=labels,
            y_proba=val_proba,
        )
        row = {
            "version_name": VERSION_NAME,
            "preprocessing_variant": candidate["preprocessing_variant"],
            "ngram_range": f"{candidate['ngram_range'][0]}-{candidate['ngram_range'][1]}",
            "class_weight": "none" if candidate["class_weight"] is None else candidate["class_weight"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "log_loss": metrics.get("log_loss"),
        }
        grid_rows.append(row)
        trained_candidates.append((candidate, model, metrics))

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    )
    grid_path = tables_dir / "v01_tfidf_char_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_row = grid_df.iloc[0]
    best_index = grid_df.index[0]
    best_candidate, best_model, best_metrics = trained_candidates[best_index]

    # Refit the selected configuration on the validation split basis for final
    # leaderboard prediction after metrics have been computed honestly above.
    checkpoint_path = CHECKPOINTS_DIR / f"{VERSION_NAME}.joblib"
    joblib.dump(best_model, checkpoint_path)

    val_pred = best_model.predict(val_df[config.text_col])
    val_proba = best_model.predict_proba(val_df[config.text_col])
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
        "preprocessing_variant": best_candidate["preprocessing_variant"],
        "ngram_range": list(best_candidate["ngram_range"]),
        "class_weight": best_candidate["class_weight"],
    }
    metrics["grid_results_path"] = str(grid_path)
    metrics_path = METRICS_DIR / f"{VERSION_NAME}_metrics.json"
    save_json(metrics, metrics_path)

    per_class_path = METRICS_DIR / f"{VERSION_NAME}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)

    top_ngrams = extract_top_ngrams_per_class(best_model)
    top_ngrams_path = tables_dir / "v01_tfidf_char_top_ngrams.csv"
    top_ngrams.to_csv(top_ngrams_path, index=False)

    final_model = make_pipeline(
        ngram_range=best_candidate["ngram_range"],
        class_weight=best_candidate["class_weight"],
        preprocessing_variant=best_candidate["preprocessing_variant"],
        seed=SEED,
    )
    final_model.fit(train[config.text_col], train[config.target_col])
    joblib.dump(final_model, checkpoint_path)

    leaderboard_pred = final_model.predict(leaderboard[config.text_col])
    leaderboard_proba = final_model.predict_proba(leaderboard[config.text_col])
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

    config_dict = asdict(config)
    config_dict["grid"] = [
        {
            "preprocessing_variant": row["preprocessing_variant"],
            "ngram_range": row["ngram_range"],
            "class_weight": row["class_weight"],
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
        "top_ngrams": str(top_ngrams_path),
        "model_artifact": str(checkpoint_path),
        "leaderboard_detailed": str(leaderboard_detailed_path),
        "submission": str(submission_path),
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

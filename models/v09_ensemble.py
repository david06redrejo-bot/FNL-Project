"""v09: validation-driven ensemble over completed model versions.

The ensemble uses only validation predictions and leaderboard predictions from
previous model runs. It never reads leaderboard labels and it does not tune
against public leaderboard feedback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_full_metrics, per_class_metrics_table
from src.paths import LOGS_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SUBMISSIONS_DIR, ensure_project_dirs
from src.reporting import write_run_summary
from src.utils import save_json


VERSION_NAME = "v09_ensemble"


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one candidate model's saved predictions."""

    name: str
    family: str
    val_path: Path
    leaderboard_path: Path
    metrics_path: Path | None = None
    use_in_probability_ensemble: bool = True


MODEL_SPECS = [
    ModelSpec(
        name="v04_roberta_cls",
        family="roberta_cls",
        val_path=PREDICTIONS_DIR / "v04_roberta_cls_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v04_roberta_cls_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v04_roberta_cls_metrics.json",
    ),
    ModelSpec(
        name="v05_roberta_mean",
        family="roberta_mean",
        val_path=PREDICTIONS_DIR / "v05_roberta_mean_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v05_roberta_mean_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v05_roberta_mean_metrics.json",
    ),
    ModelSpec(
        name="v08_roberta_mean_dedupe",
        family="roberta_mean_safe_data",
        val_path=PREDICTIONS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_metrics.json",
    ),
    ModelSpec(
        name="v08_roberta_mean_weighted_sampler",
        family="roberta_mean_safe_data",
        val_path=PREDICTIONS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_metrics.json",
    ),
    ModelSpec(
        name="v06_roberta_mean_class_weighted",
        family="roberta_mean_imbalance",
        val_path=PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_metrics.json",
    ),
    ModelSpec(
        name="v06_roberta_mean_focal_gamma2",
        family="roberta_mean_imbalance",
        val_path=PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_metrics.json",
    ),
    ModelSpec(
        name="v01_tfidf_char_logreg",
        family="tfidf_char",
        val_path=PREDICTIONS_DIR / "v01_tfidf_char_logreg_val_predictions.csv",
        leaderboard_path=PREDICTIONS_DIR / "v01_tfidf_char_logreg_leaderboard_detailed.csv",
        metrics_path=METRICS_DIR / "v01_tfidf_char_logreg_metrics.json",
    ),
]


def proba_columns(df: pd.DataFrame) -> list[str]:
    """Return probability columns in their stored order."""
    cols = [col for col in df.columns if col.startswith("proba_")]
    if not cols:
        raise ValueError("Prediction file does not contain probability columns.")
    return cols


def labels_from_proba_columns(cols: list[str]) -> list[str]:
    """Map `proba_X` columns to labels."""
    return [col.replace("proba_", "", 1) for col in cols]


def load_prediction_bundle(spec: ModelSpec) -> dict[str, Any]:
    """Load validation and leaderboard prediction files for one model."""
    missing = [path for path in [spec.val_path, spec.leaderboard_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for {spec.name}: {missing}")
    val = pd.read_csv(spec.val_path)
    leaderboard = pd.read_csv(spec.leaderboard_path)
    cols = proba_columns(val)
    labels = labels_from_proba_columns(cols)
    leaderboard_cols = proba_columns(leaderboard)
    if leaderboard_cols != cols:
        raise ValueError(f"Probability columns differ for {spec.name} validation vs leaderboard.")
    return {
        "spec": spec,
        "val": val,
        "leaderboard": leaderboard,
        "labels": labels,
        "proba_cols": cols,
        "val_proba": normalize_probabilities(val[cols].to_numpy(dtype=float)),
        "leaderboard_proba": normalize_probabilities(leaderboard[cols].to_numpy(dtype=float)),
    }


def validate_alignment(bundles: dict[str, dict[str, Any]]) -> tuple[list[str], pd.Series, pd.DataFrame]:
    """Ensure all validation files refer to the same split and label space."""
    names = list(bundles)
    reference = bundles[names[0]]
    labels = reference["labels"]
    y_true = reference["val"]["y_true"].astype(str).str.upper()
    literal = reference["val"].get("Literal")
    for name in names[1:]:
        bundle = bundles[name]
        if bundle["labels"] != labels:
            raise ValueError(f"Label/probability order mismatch between {names[0]} and {name}.")
        other_true = bundle["val"]["y_true"].astype(str).str.upper()
        if not y_true.equals(other_true):
            raise ValueError(f"Validation y_true alignment mismatch for {name}.")
        if literal is not None and "Literal" in bundle["val"].columns:
            if not literal.astype(str).equals(bundle["val"]["Literal"].astype(str)):
                raise ValueError(f"Validation literal alignment mismatch for {name}.")
    leaderboard = reference["leaderboard"][["id"]].copy()
    for name in names[1:]:
        other_ids = bundles[name]["leaderboard"]["id"]
        if not leaderboard["id"].equals(other_ids):
            raise ValueError(f"Leaderboard id alignment mismatch for {name}.")
    return labels, y_true, leaderboard


def normalize_probabilities(proba: np.ndarray) -> np.ndarray:
    """Normalize rows defensively after numeric combination."""
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def predict_from_proba(proba: np.ndarray, labels: list[str]) -> np.ndarray:
    """Convert probabilities to label predictions."""
    labels_array = np.asarray(labels)
    return labels_array[proba.argmax(axis=1)]


def average_proba(bundles: dict[str, dict[str, Any]], model_names: list[str], split: str) -> np.ndarray:
    """Unweighted probability average for a split."""
    key = "val_proba" if split == "val" else "leaderboard_proba"
    return normalize_probabilities(np.mean([bundles[name][key] for name in model_names], axis=0))


def weighted_average_proba(
    bundles: dict[str, dict[str, Any]],
    model_weights: dict[str, float],
    split: str,
) -> np.ndarray:
    """Weighted probability average for a split."""
    key = "val_proba" if split == "val" else "leaderboard_proba"
    total = sum(model_weights.values())
    if total <= 0:
        raise ValueError("Model weights must sum to a positive value.")
    proba = None
    for name, weight in model_weights.items():
        weighted = bundles[name][key] * (weight / total)
        proba = weighted if proba is None else proba + weighted
    return normalize_probabilities(proba)


def majority_vote_predictions(bundles: dict[str, dict[str, Any]], model_names: list[str], split: str, labels: list[str]) -> np.ndarray:
    """Majority vote with confidence tie-break from average probabilities."""
    pred_key = "y_pred"
    rows = []
    for name in model_names:
        frame = bundles[name]["val" if split == "val" else "leaderboard"]
        rows.append(frame[pred_key].astype(str).str.upper().to_numpy())
    votes = np.vstack(rows).T
    avg_proba = average_proba(bundles, model_names, split)
    output = []
    label_order = {label: idx for idx, label in enumerate(labels)}
    for row_idx, row_votes in enumerate(votes):
        counts = pd.Series(row_votes).value_counts()
        top_count = counts.max()
        tied = sorted(counts[counts == top_count].index, key=lambda label: label_order.get(label, 10_000))
        if len(tied) == 1:
            output.append(tied[0])
        else:
            tied_indices = [label_order[label] for label in tied]
            best = tied_indices[int(np.argmax(avg_proba[row_idx, tied_indices]))]
            output.append(labels[best])
    return np.asarray(output)


def fallback_tfidf_predictions(
    bundles: dict[str, dict[str, Any]],
    neural_weights: dict[str, float],
    labels: list[str],
    split: str,
    neural_threshold: float,
    tfidf_threshold: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use TF-IDF only when neural confidence is low and TF-IDF is high."""
    neural_proba = weighted_average_proba(bundles, neural_weights, split)
    tfidf_proba = bundles["v01_tfidf_char_logreg"]["val_proba" if split == "val" else "leaderboard_proba"]
    neural_conf = neural_proba.max(axis=1)
    tfidf_conf = tfidf_proba.max(axis=1)
    neural_pred = predict_from_proba(neural_proba, labels)
    tfidf_pred = predict_from_proba(tfidf_proba, labels)
    use_tfidf = (neural_conf < neural_threshold) & (tfidf_conf >= tfidf_threshold) & ((tfidf_conf - neural_conf) >= margin)
    output = neural_pred.copy()
    output[use_tfidf] = tfidf_pred[use_tfidf]
    hybrid_proba = neural_proba.copy()
    hybrid_proba[use_tfidf] = tfidf_proba[use_tfidf]
    return output, normalize_probabilities(hybrid_proba)


def evaluate_recipe(name: str, y_true: pd.Series, labels: list[str], y_pred, proba: np.ndarray | None, recipe: str) -> dict[str, Any]:
    """Compute metrics and add recipe metadata."""
    metrics = compute_full_metrics(y_true, y_pred, labels=labels, y_proba=proba)
    return {
        "candidate_version": name,
        "recipe": recipe,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "log_loss": metrics.get("log_loss"),
        "top_3_accuracy": metrics.get("top_3_accuracy"),
        "metrics": metrics,
    }


def build_detailed_predictions(reference_df: pd.DataFrame, y_pred, proba: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Create detailed validation/leaderboard predictions for the ensemble."""
    output = pd.DataFrame()
    if "id" in reference_df.columns:
        output["id"] = reference_df["id"].values
    if "Literal" in reference_df.columns:
        output["Literal"] = reference_df["Literal"].values
    if "y_true" in reference_df.columns:
        output["y_true"] = reference_df["y_true"].values
    output["y_pred"] = y_pred
    for idx, label in enumerate(labels):
        output[f"proba_{label}"] = proba[:, idx]
    return output


def run() -> dict[str, Any]:
    """Run ensemble recipes and write artifacts."""
    ensure_project_dirs()
    bundles = {spec.name: load_prediction_bundle(spec) for spec in MODEL_SPECS}
    labels, y_true, leaderboard_ids = validate_alignment(bundles)

    neural_core = ["v04_roberta_cls", "v05_roberta_mean", "v08_roberta_mean_dedupe"]
    neural_balanced = [
        "v04_roberta_cls",
        "v05_roberta_mean",
        "v08_roberta_mean_dedupe",
        "v08_roberta_mean_weighted_sampler",
    ]
    all_probability = [
        "v04_roberta_cls",
        "v05_roberta_mean",
        "v08_roberta_mean_dedupe",
        "v06_roberta_mean_class_weighted",
        "v06_roberta_mean_focal_gamma2",
        "v01_tfidf_char_logreg",
    ]
    recipes: list[dict[str, Any]] = []

    individual_rows = []
    for name, bundle in bundles.items():
        pred = bundle["val"]["y_pred"].astype(str).str.upper().to_numpy()
        row = evaluate_recipe(name, y_true, labels, pred, bundle["val_proba"], "individual model")
        row["is_individual"] = True
        individual_rows.append(row)

    avg_core = average_proba(bundles, neural_core, "val")
    recipes.append(
        evaluate_recipe(
            "v09_neural_probability_average",
            y_true,
            labels,
            predict_from_proba(avg_core, labels),
            avg_core,
            "unweighted probability average over CLS, mean, and safe-dedupe mean",
        )
    )

    weighted_core_weights = {"v04_roberta_cls": 0.36, "v08_roberta_mean_dedupe": 0.34, "v05_roberta_mean": 0.30}
    weighted_core = weighted_average_proba(bundles, weighted_core_weights, "val")
    recipes.append(
        evaluate_recipe(
            "v09_weighted_neural_probability_average",
            y_true,
            labels,
            predict_from_proba(weighted_core, labels),
            weighted_core,
            f"weighted neural probability average: {weighted_core_weights}",
        )
    )

    balanced_weights = {
        "v04_roberta_cls": 0.34,
        "v08_roberta_mean_dedupe": 0.30,
        "v05_roberta_mean": 0.22,
        "v08_roberta_mean_weighted_sampler": 0.14,
    }
    balanced_proba = weighted_average_proba(bundles, balanced_weights, "val")
    recipes.append(
        evaluate_recipe(
            "v09_weighted_neural_with_sampler",
            y_true,
            labels,
            predict_from_proba(balanced_proba, labels),
            balanced_proba,
            f"weighted neural average including sampler model: {balanced_weights}",
        )
    )

    all_avg = average_proba(bundles, all_probability, "val")
    recipes.append(
        evaluate_recipe(
            "v09_all_probability_average",
            y_true,
            labels,
            predict_from_proba(all_avg, labels),
            all_avg,
            "unweighted probability average over selected RoBERTa variants and TF-IDF char",
        )
    )

    vote_names = ["v04_roberta_cls", "v05_roberta_mean", "v08_roberta_mean_dedupe", "v08_roberta_mean_weighted_sampler", "v01_tfidf_char_logreg"]
    vote_pred = majority_vote_predictions(bundles, vote_names, "val", labels)
    recipes.append(
        evaluate_recipe(
            "v09_majority_vote",
            y_true,
            labels,
            vote_pred,
            None,
            f"majority vote over {vote_names} with average-probability tie-break",
        )
    )

    fallback_base_weights = weighted_core_weights
    for neural_threshold, tfidf_threshold, margin in [(0.35, 0.55, 0.10), (0.40, 0.60, 0.10), (0.45, 0.65, 0.15)]:
        pred, proba = fallback_tfidf_predictions(
            bundles,
            fallback_base_weights,
            labels,
            "val",
            neural_threshold=neural_threshold,
            tfidf_threshold=tfidf_threshold,
            margin=margin,
        )
        recipes.append(
            evaluate_recipe(
                f"v09_low_confidence_tfidf_fallback_{neural_threshold}_{tfidf_threshold}_{margin}",
                y_true,
                labels,
                pred,
                proba,
                (
                    "weighted neural average, but use TF-IDF char if neural confidence "
                    f"< {neural_threshold}, TF-IDF confidence >= {tfidf_threshold}, margin >= {margin}"
                ),
            )
        )

    rows = []
    for row in individual_rows + recipes:
        rows.append(
            {
                "candidate_version": row["candidate_version"],
                "kind": "individual" if row.get("is_individual") else "ensemble",
                "recipe": row["recipe"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
                "macro_precision": row["macro_precision"],
                "macro_recall": row["macro_recall"],
                "log_loss": row.get("log_loss"),
                "top_3_accuracy": row.get("top_3_accuracy"),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1", "weighted_f1"], ascending=[False, False, False])
    comparison_path = REPORTS_DIR / "tables" / "v09_ensemble_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    ensemble_rows = [row for row in recipes]
    best = sorted(ensemble_rows, key=lambda row: (row["accuracy"], row["macro_f1"], row["weighted_f1"]), reverse=True)[0]
    best_name = best["candidate_version"]

    if best_name == "v09_neural_probability_average":
        leaderboard_proba = average_proba(bundles, neural_core, "leaderboard")
    elif best_name == "v09_weighted_neural_probability_average":
        leaderboard_proba = weighted_average_proba(bundles, weighted_core_weights, "leaderboard")
    elif best_name == "v09_weighted_neural_with_sampler":
        leaderboard_proba = weighted_average_proba(bundles, balanced_weights, "leaderboard")
    elif best_name == "v09_all_probability_average":
        leaderboard_proba = average_proba(bundles, all_probability, "leaderboard")
    elif best_name == "v09_majority_vote":
        leaderboard_pred = majority_vote_predictions(bundles, vote_names, "leaderboard", labels)
        leaderboard_proba = average_proba(bundles, vote_names, "leaderboard")
    elif best_name.startswith("v09_low_confidence_tfidf_fallback"):
        parts = best_name.replace("v09_low_confidence_tfidf_fallback_", "").split("_")
        leaderboard_pred, leaderboard_proba = fallback_tfidf_predictions(
            bundles,
            fallback_base_weights,
            labels,
            "leaderboard",
            neural_threshold=float(parts[0]),
            tfidf_threshold=float(parts[1]),
            margin=float(parts[2]),
        )
    else:
        raise ValueError(f"Unknown best recipe: {best_name}")

    if best_name != "v09_majority_vote" and not best_name.startswith("v09_low_confidence_tfidf_fallback"):
        leaderboard_pred = predict_from_proba(leaderboard_proba, labels)
    val_reference = bundles["v04_roberta_cls"]["val"]
    if best_name == "v09_majority_vote":
        best_val_pred = majority_vote_predictions(bundles, vote_names, "val", labels)
        best_val_proba = average_proba(bundles, vote_names, "val")
    elif best_name.startswith("v09_low_confidence_tfidf_fallback"):
        parts = best_name.replace("v09_low_confidence_tfidf_fallback_", "").split("_")
        best_val_pred, best_val_proba = fallback_tfidf_predictions(
            bundles,
            fallback_base_weights,
            labels,
            "val",
            neural_threshold=float(parts[0]),
            tfidf_threshold=float(parts[1]),
            margin=float(parts[2]),
        )
    else:
        best_val_proba = {
            "v09_neural_probability_average": avg_core,
            "v09_weighted_neural_probability_average": weighted_core,
            "v09_weighted_neural_with_sampler": balanced_proba,
            "v09_all_probability_average": all_avg,
        }[best_name]
        best_val_pred = predict_from_proba(best_val_proba, labels)

    metrics = best["metrics"]
    metrics["selected_recipe"] = best["recipe"]
    metrics["selected_candidate"] = best_name
    metrics["selection_policy"] = "best validation accuracy among predefined ensemble recipes"
    metrics_path = METRICS_DIR / f"{VERSION_NAME}_metrics.json"
    save_json(metrics, metrics_path)
    per_class_path = METRICS_DIR / f"{VERSION_NAME}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)

    val_predictions_path = PREDICTIONS_DIR / f"{VERSION_NAME}_val_predictions.csv"
    build_detailed_predictions(val_reference, best_val_pred, best_val_proba, labels).to_csv(val_predictions_path, index=False)
    leaderboard_reference = bundles["v04_roberta_cls"]["leaderboard"]
    leaderboard_detailed_path = PREDICTIONS_DIR / f"{VERSION_NAME}_leaderboard_detailed.csv"
    build_detailed_predictions(leaderboard_reference, leaderboard_pred, leaderboard_proba, labels).to_csv(
        leaderboard_detailed_path, index=False
    )
    submission = pd.DataFrame({"id": leaderboard_ids["id"], "y_category": leaderboard_pred})[["id", "y_category"]]
    submission_path = SUBMISSIONS_DIR / f"{VERSION_NAME}_submission.csv"
    submission.to_csv(submission_path, index=False)

    recipe_path = LOGS_DIR / f"{VERSION_NAME}_recipe.md"
    recipe_path.write_text(
        "\n".join(
            [
                "# v09 Ensemble Recipe",
                "",
                "Selection policy: choose the best predefined ensemble by validation accuracy.",
                "No leaderboard labels or public leaderboard feedback were used.",
                "",
                f"Selected candidate: `{best_name}`",
                f"Recipe: {best['recipe']}",
                "",
                "Inputs:",
                *[f"- `{spec.name}`: {spec.family}" for spec in MODEL_SPECS],
                "",
                "Important caution: ensembling can reduce variance only if models make partially different errors. "
                "If model errors are highly correlated, averaging may not improve over the best single model.",
            ]
        ),
        encoding="utf-8",
    )

    artifact_paths = {
        "comparison_table": str(comparison_path),
        "recipe": str(recipe_path),
        "metrics": str(metrics_path),
        "per_class_metrics": str(per_class_path),
        "validation_predictions": str(val_predictions_path),
        "leaderboard_detailed": str(leaderboard_detailed_path),
        "submission": str(submission_path),
    }
    run_summary = write_run_summary(VERSION_NAME, {"version_name": VERSION_NAME, "inputs": [spec.name for spec in MODEL_SPECS]}, metrics, artifact_paths)
    artifact_paths["run_summary"] = str(run_summary)
    return {"selected_candidate": best_name, "metrics": metrics, "artifacts": artifact_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v09 ensemble over completed model predictions.")
    parser.add_argument("--dry-run", action="store_true", help="Validate prediction inputs and print available candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        bundles = {spec.name: load_prediction_bundle(spec) for spec in MODEL_SPECS}
        labels, y_true, leaderboard = validate_alignment(bundles)
        print({"models": list(bundles), "num_labels": len(labels), "val_rows": len(y_true), "leaderboard_rows": len(leaderboard)})
        return
    result = run()
    print(
        {
            "selected_candidate": result["selected_candidate"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "weighted_f1": result["metrics"]["weighted_f1"],
            "artifacts": result["artifacts"],
        }
    )


if __name__ == "__main__":
    main()

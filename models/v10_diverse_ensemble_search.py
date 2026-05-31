"""v10: diverse ensemble search over completed model predictions.

This script intentionally combines different model families: TF-IDF, retrieval,
RoBERTa pooling variants, imbalance-aware models, safe-data models, and the
previous ensemble. It does not train a new neural model and it does not use
leaderboard labels.
"""

from __future__ import annotations

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
from src.paths import METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SUBMISSIONS_DIR
from src.reporting import write_run_summary
from src.utils import save_json


VERSION_NAME = "v10_diverse_ensemble_search"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    val_path: Path
    leaderboard_path: Path
    validation_accuracy: float
    has_probabilities: bool = True


CANDIDATES = [
    Candidate(
        "v01_char_tfidf_logreg",
        "tfidf_char",
        PREDICTIONS_DIR / "v01_tfidf_char_logreg_val_predictions.csv",
        PREDICTIONS_DIR / "v01_tfidf_char_logreg_leaderboard_detailed.csv",
        0.522628,
        True,
    ),
    Candidate(
        "v02_word_tfidf_svm",
        "tfidf_word_svm",
        PREDICTIONS_DIR / "v02_tfidf_word_svm_val_predictions.csv",
        PREDICTIONS_DIR / "v02_tfidf_word_svm_leaderboard_detailed.csv",
        0.520073,
        False,
    ),
    Candidate(
        "v03_similarity_retrieval",
        "retrieval",
        PREDICTIONS_DIR / "v03_similarity_retrieval_baseline_val_predictions.csv",
        PREDICTIONS_DIR / "v03_similarity_retrieval_baseline_leaderboard_detailed.csv",
        0.497445,
        False,
    ),
    Candidate(
        "v04_roberta_cls",
        "roberta_cls",
        PREDICTIONS_DIR / "v04_roberta_cls_val_predictions.csv",
        PREDICTIONS_DIR / "v04_roberta_cls_leaderboard_detailed.csv",
        0.569343,
        True,
    ),
    Candidate(
        "v05_roberta_mean",
        "roberta_mean",
        PREDICTIONS_DIR / "v05_roberta_mean_val_predictions.csv",
        PREDICTIONS_DIR / "v05_roberta_mean_leaderboard_detailed.csv",
        0.564599,
        True,
    ),
    Candidate(
        "v06_class_weighted",
        "roberta_imbalance",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_val_predictions.csv",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_class_weight_balanced_leaderboard_detailed.csv",
        0.544526,
        True,
    ),
    Candidate(
        "v06_focal_gamma1",
        "roberta_imbalance",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma1_val_predictions.csv",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma1_leaderboard_detailed.csv",
        0.557299,
        True,
    ),
    Candidate(
        "v06_focal_gamma2",
        "roberta_imbalance",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_val_predictions.csv",
        PREDICTIONS_DIR / "v06_roberta_mean_imbalance_aware_focal_gamma2_leaderboard_detailed.csv",
        0.555474,
        True,
    ),
    Candidate(
        "v07_mean_tuning",
        "roberta_tuned",
        PREDICTIONS_DIR / "v07_roberta_mean_tuning_c_recommended_32_lr2e5_warmup006_clip_val_predictions.csv",
        PREDICTIONS_DIR / "v07_roberta_mean_tuning_c_recommended_32_lr2e5_warmup006_clip_leaderboard_detailed.csv",
        0.564599,
        True,
    ),
    Candidate(
        "v08_safe_dedupe",
        "roberta_safe_data",
        PREDICTIONS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_val_predictions.csv",
        PREDICTIONS_DIR / "v08_roberta_mean_augmented_dedupe_non_conflicting_literals_leaderboard_detailed.csv",
        0.568613,
        True,
    ),
    Candidate(
        "v08_weighted_sampler",
        "roberta_safe_data",
        PREDICTIONS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_val_predictions.csv",
        PREDICTIONS_DIR / "v08_roberta_mean_augmented_weighted_random_sampler_leaderboard_detailed.csv",
        0.542336,
        True,
    ),
    Candidate(
        "v09_ensemble",
        "previous_ensemble",
        PREDICTIONS_DIR / "v09_ensemble_val_predictions.csv",
        PREDICTIONS_DIR / "v09_ensemble_leaderboard_detailed.csv",
        0.576642,
        True,
    ),
]


def probability_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("proba_")]


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def load_candidates() -> tuple[dict[str, dict[str, Any]], list[str], pd.Series, pd.DataFrame]:
    bundles = {}
    for candidate in CANDIDATES:
        if not candidate.val_path.exists() or not candidate.leaderboard_path.exists():
            raise FileNotFoundError(f"Missing predictions for {candidate.name}")
        val = pd.read_csv(candidate.val_path)
        leaderboard = pd.read_csv(candidate.leaderboard_path)
        cols = probability_columns(val)
        if candidate.has_probabilities and not cols:
            raise ValueError(f"{candidate.name} expected probability columns")
        proba = normalize(val[cols].to_numpy(dtype=float)) if cols else None
        leaderboard_cols = probability_columns(leaderboard)
        leaderboard_proba = normalize(leaderboard[leaderboard_cols].to_numpy(dtype=float)) if leaderboard_cols else None
        labels = [col.replace("proba_", "", 1) for col in cols] if cols else None
        bundles[candidate.name] = {
            "candidate": candidate,
            "val": val,
            "leaderboard": leaderboard,
            "labels": labels,
            "proba_cols": cols,
            "val_proba": proba,
            "leaderboard_proba": leaderboard_proba,
            "val_pred": val["y_pred"].astype(str).str.upper().to_numpy(),
            "leaderboard_pred": leaderboard["y_pred"].astype(str).str.upper().to_numpy(),
        }
    reference = bundles["v04_roberta_cls"]
    labels = reference["labels"]
    y_true = reference["val"]["y_true"].astype(str).str.upper()
    for name, bundle in bundles.items():
        if not y_true.equals(bundle["val"]["y_true"].astype(str).str.upper()):
            raise ValueError(f"Validation alignment mismatch: {name}")
        if bundle["labels"] is not None and bundle["labels"] != labels:
            raise ValueError(f"Label order mismatch: {name}")
    leaderboard_reference = reference["leaderboard"][["id", "Literal"]].copy()
    for name, bundle in bundles.items():
        if not leaderboard_reference["id"].equals(bundle["leaderboard"]["id"]):
            raise ValueError(f"Leaderboard id mismatch: {name}")
    return bundles, labels, y_true, leaderboard_reference


def pred_from_proba(proba: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.asarray(labels)[proba.argmax(axis=1)]


def weighted_proba(bundles, weights: dict[str, float], split: str) -> np.ndarray:
    key = "val_proba" if split == "val" else "leaderboard_proba"
    total = sum(weights.values())
    out = None
    for name, weight in weights.items():
        p = bundles[name][key]
        if p is None:
            raise ValueError(f"{name} has no probabilities")
        out = p * (weight / total) if out is None else out + p * (weight / total)
    return normalize(out)


def vote_predictions(
    bundles,
    weights: dict[str, float],
    split: str,
    labels: list[str],
    tie_break_weights: dict[str, float] | None = None,
) -> np.ndarray:
    pred_key = "val_pred" if split == "val" else "leaderboard_pred"
    proba_key = "val_proba" if split == "val" else "leaderboard_proba"
    label_order = {label: idx for idx, label in enumerate(labels)}
    avg_proba = None
    if tie_break_weights:
        avg_proba = weighted_proba(bundles, tie_break_weights, split)
    output = []
    n = len(next(iter(bundles.values()))[pred_key])
    for row_idx in range(n):
        scores = {label: 0.0 for label in labels}
        for name, weight in weights.items():
            pred = bundles[name][pred_key][row_idx]
            scores[pred] = scores.get(pred, 0.0) + weight
        best_score = max(scores.values())
        tied = [label for label, score in scores.items() if score == best_score]
        if len(tied) == 1:
            output.append(tied[0])
        elif avg_proba is not None:
            tied_indices = [label_order[label] for label in tied]
            output.append(labels[tied_indices[int(np.argmax(avg_proba[row_idx, tied_indices]))]])
        else:
            output.append(sorted(tied)[0])
    return np.asarray(output)


def confidence_fallback(
    bundles,
    base: str,
    fallback: str,
    labels: list[str],
    split: str,
    base_threshold: float,
    fallback_threshold: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    key = "val_proba" if split == "val" else "leaderboard_proba"
    base_proba = bundles[base][key]
    fallback_proba = bundles[fallback][key]
    base_conf = base_proba.max(axis=1)
    fallback_conf = fallback_proba.max(axis=1)
    base_pred = pred_from_proba(base_proba, labels)
    fallback_pred = pred_from_proba(fallback_proba, labels)
    use_fallback = (base_conf < base_threshold) & (fallback_conf >= fallback_threshold) & ((fallback_conf - base_conf) >= margin)
    pred = base_pred.copy()
    pred[use_fallback] = fallback_pred[use_fallback]
    proba = base_proba.copy()
    proba[use_fallback] = fallback_proba[use_fallback]
    return pred, normalize(proba)


def recipe_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # Diverse votes: include different ML families.
    vote_sets = {
        "vote_diverse_core": ["v08_safe_dedupe", "v04_roberta_cls", "v01_char_tfidf_logreg", "v02_word_tfidf_svm", "v03_similarity_retrieval"],
        "vote_diverse_no_retrieval": ["v08_safe_dedupe", "v04_roberta_cls", "v01_char_tfidf_logreg", "v02_word_tfidf_svm"],
        "vote_all_families": [c.name for c in CANDIDATES],
        "vote_public_top_diverse": ["v08_safe_dedupe", "v04_roberta_cls", "v07_mean_tuning", "v01_char_tfidf_logreg", "v02_word_tfidf_svm"],
        "vote_roberta_diverse": ["v08_safe_dedupe", "v04_roberta_cls", "v05_roberta_mean", "v06_class_weighted", "v08_weighted_sampler"],
        "vote_validation_top_plus_tfidf": ["v09_ensemble", "v08_safe_dedupe", "v04_roberta_cls", "v01_char_tfidf_logreg", "v02_word_tfidf_svm"],
    }
    for name, models in vote_sets.items():
        specs.append({"name": name, "method": "vote", "weights": {model: 1.0 for model in models}})
        specs.append(
            {
                "name": f"{name}_val_weighted",
                "method": "vote",
                "weights": {model: next(c.validation_accuracy for c in CANDIDATES if c.name == model) for model in models},
            }
        )
    # Probability averages: only models with probabilities.
    proba_sets = {
        "proba_public_top_roberta": ["v08_safe_dedupe", "v04_roberta_cls", "v07_mean_tuning", "v05_roberta_mean"],
        "proba_public_top_plus_tfidf": ["v08_safe_dedupe", "v04_roberta_cls", "v07_mean_tuning", "v01_char_tfidf_logreg"],
        "proba_validation_top": ["v09_ensemble", "v08_safe_dedupe", "v04_roberta_cls", "v05_roberta_mean"],
        "proba_imbalance_mix": ["v08_safe_dedupe", "v06_class_weighted", "v08_weighted_sampler", "v01_char_tfidf_logreg"],
        "proba_all_available": [c.name for c in CANDIDATES if c.has_probabilities],
    }
    for name, models in proba_sets.items():
        specs.append({"name": name, "method": "proba", "weights": {model: 1.0 for model in models}})
        specs.append(
            {
                "name": f"{name}_val_weighted",
                "method": "proba",
                "weights": {model: next(c.validation_accuracy for c in CANDIDATES if c.name == model) for model in models},
            }
        )
    # Public-best anchored probability blends.
    for other in ["v04_roberta_cls", "v05_roberta_mean", "v07_mean_tuning", "v09_ensemble", "v01_char_tfidf_logreg", "v06_focal_gamma1"]:
        for base_weight in [0.55, 0.65, 0.75, 0.85]:
            specs.append(
                {
                    "name": f"blend_v08_{int(base_weight*100)}_{other}",
                    "method": "proba",
                    "weights": {"v08_safe_dedupe": base_weight, other: 1 - base_weight},
                }
            )
    # Confidence fallback from public-best model to diverse models.
    for fallback in ["v04_roberta_cls", "v01_char_tfidf_logreg", "v09_ensemble", "v06_class_weighted"]:
        for base_threshold, fallback_threshold, margin in [(0.35, 0.55, 0.10), (0.45, 0.60, 0.10), (0.55, 0.65, 0.15)]:
            specs.append(
                {
                    "name": f"fallback_v08_to_{fallback}_{base_threshold}_{fallback_threshold}_{margin}",
                    "method": "fallback",
                    "base": "v08_safe_dedupe",
                    "fallback": fallback,
                    "base_threshold": base_threshold,
                    "fallback_threshold": fallback_threshold,
                    "margin": margin,
                }
            )
    return specs


def evaluate_recipe(spec, bundles, labels, y_true):
    if spec["method"] == "proba":
        proba = weighted_proba(bundles, spec["weights"], "val")
        pred = pred_from_proba(proba, labels)
    elif spec["method"] == "vote":
        pred = vote_predictions(
            bundles,
            spec["weights"],
            "val",
            labels,
            tie_break_weights={name: 1.0 for name in spec["weights"] if bundles[name]["val_proba"] is not None},
        )
        proba = None
    elif spec["method"] == "fallback":
        pred, proba = confidence_fallback(
            bundles,
            spec["base"],
            spec["fallback"],
            labels,
            "val",
            spec["base_threshold"],
            spec["fallback_threshold"],
            spec["margin"],
        )
    else:
        raise ValueError(spec["method"])
    metrics = compute_full_metrics(y_true, pred, labels=labels, y_proba=proba)
    return pred, proba, metrics


def leaderboard_for_recipe(spec, bundles, labels):
    if spec["method"] == "proba":
        proba = weighted_proba(bundles, spec["weights"], "leaderboard")
        pred = pred_from_proba(proba, labels)
    elif spec["method"] == "vote":
        pred = vote_predictions(
            bundles,
            spec["weights"],
            "leaderboard",
            labels,
            tie_break_weights={name: 1.0 for name in spec["weights"] if bundles[name]["leaderboard_proba"] is not None},
        )
        proba = None
    elif spec["method"] == "fallback":
        pred, proba = confidence_fallback(
            bundles,
            spec["base"],
            spec["fallback"],
            labels,
            "leaderboard",
            spec["base_threshold"],
            spec["fallback_threshold"],
            spec["margin"],
        )
    else:
        raise ValueError(spec["method"])
    return pred, proba


def run(top_n_submissions: int = 16) -> dict[str, Any]:
    bundles, labels, y_true, leaderboard = load_candidates()
    rows = []
    evaluated = {}
    for spec in recipe_specs():
        pred, proba, metrics = evaluate_recipe(spec, bundles, labels, y_true)
        families = sorted({bundles[name]["candidate"].family for name in spec.get("weights", {})})
        if spec["method"] == "fallback":
            families = sorted({bundles[spec["base"]]["candidate"].family, bundles[spec["fallback"]]["candidate"].family})
        rows.append(
            {
                "recipe_name": spec["name"],
                "method": spec["method"],
                "models": " | ".join(spec.get("weights", {}).keys()) if "weights" in spec else f"{spec['base']} -> {spec['fallback']}",
                "families": " | ".join(families),
                "n_families": len(families),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "top_3_accuracy": metrics.get("top_3_accuracy"),
                "log_loss": metrics.get("log_loss"),
            }
        )
        evaluated[spec["name"]] = {"spec": spec, "pred": pred, "proba": proba, "metrics": metrics}

    results = pd.DataFrame(rows).sort_values(["accuracy", "n_families", "macro_f1"], ascending=[False, False, False])
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "v10_diverse_ensemble_search.csv"
    results.to_csv(results_path, index=False)

    selected = results.head(top_n_submissions).copy()
    # Add public-score-inspired candidates even if validation is not top.
    extra_names = [
        "blend_v08_85_v04_roberta_cls",
        "blend_v08_75_v04_roberta_cls",
        "blend_v08_85_v09_ensemble",
        "blend_v08_75_v09_ensemble",
        "vote_public_top_diverse",
        "vote_public_top_diverse_val_weighted",
    ]
    selected = pd.concat([selected, results[results["recipe_name"].isin(extra_names)]], ignore_index=True)
    selected = selected.drop_duplicates("recipe_name").head(top_n_submissions)
    submission_rows = []
    for _, row in selected.iterrows():
        name = row["recipe_name"]
        pred, proba = leaderboard_for_recipe(evaluated[name]["spec"], bundles, labels)
        submission = leaderboard.copy()
        submission["y_category"] = pred
        safe_name = name.replace(".", "p").replace(" ", "_")
        submission_path = SUBMISSIONS_DIR / f"v10_{safe_name}_kaggle.csv"
        submission[["id", "Literal", "y_category"]].to_csv(submission_path, index=False)
        detailed = submission.copy()
        if proba is not None:
            for idx, label in enumerate(labels):
                detailed[f"proba_{label}"] = proba[:, idx]
        detailed_path = PREDICTIONS_DIR / f"v10_{safe_name}_leaderboard_detailed.csv"
        detailed.to_csv(detailed_path, index=False)
        submission_rows.append({**row.to_dict(), "submission_path": str(submission_path), "detailed_path": str(detailed_path)})

    submission_plan = pd.DataFrame(submission_rows)
    submission_plan_path = tables_dir / "v10_diverse_ensemble_submission_plan.csv"
    submission_plan.to_csv(submission_plan_path, index=False)

    best_name = results.iloc[0]["recipe_name"]
    best = evaluated[best_name]
    metrics = best["metrics"]
    metrics["selected_recipe"] = best_name
    metrics_path = METRICS_DIR / f"{VERSION_NAME}_metrics.json"
    save_json(metrics, metrics_path)
    per_class_metrics_table(metrics).to_csv(METRICS_DIR / f"{VERSION_NAME}_per_class_metrics.csv", index=False)
    run_summary = write_run_summary(
        VERSION_NAME,
        {"selected_recipe": best_name, "selection": "best validation accuracy among diverse predefined recipes"},
        metrics,
        {"search_table": str(results_path), "submission_plan": str(submission_plan_path), "metrics": str(metrics_path)},
    )
    return {"best_validation_recipe": best_name, "metrics": metrics, "results_path": results_path, "submission_plan_path": submission_plan_path, "run_summary": run_summary}


def main() -> None:
    result = run()
    print(
        {
            "best_validation_recipe": result["best_validation_recipe"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "weighted_f1": result["metrics"]["weighted_f1"],
            "results_path": str(result["results_path"]),
            "submission_plan_path": str(result["submission_plan_path"]),
        }
    )


if __name__ == "__main__":
    main()

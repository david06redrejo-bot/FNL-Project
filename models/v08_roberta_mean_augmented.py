"""v08: safe data-strategy experiments for RoBERTa mean pooling.

This file deliberately avoids unsafe clinical text augmentation. It does not
delete medical words, replace terminology with unverified synonyms, alter
negation, or fabricate clinical variants.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v05_roberta_mean import (
    RobertaMeanClassifier,
    evaluate,
    load_model_state,
    predict_leaderboard,
    train_epoch,
)
from src.datasets import (
    BACKBONE_CHECKPOINT,
    DEFAULT_MAX_LENGTH,
    ICDDatasetConfig,
    ICDLiteralDataset,
    build_label_mappings,
    load_backbone_tokenizer,
)
from src.inference import build_detailed_predictions, build_submission
from src.metrics import per_class_metrics_table
from src.paths import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SUBMISSIONS_DIR, ensure_project_dirs
from src.reporting import write_run_summary
from src.training import ModelRunConfig as SplitConfig
from src.training import load_required_clean_data, save_torch_checkpoint, split_train_validation
from src.utils import get_device, save_json, set_seed


VERSION_NAME = "v08_roberta_mean_augmented"


@dataclass
class SafeAugmentationConfig:
    """Configuration for safe data-strategy experiments."""

    version_name: str = VERSION_NAME
    backbone: str = BACKBONE_CHECKPOINT
    seed: int = 42
    max_length: int = DEFAULT_MAX_LENGTH
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    dropout: float = 0.1
    validation_size: float = 0.2
    use_amp: bool = False
    debug: bool = False
    debug_train_examples: int = 32
    debug_val_examples: int = 32
    debug_leaderboard_examples: int = 32


def candidate_grid(debug: bool = False) -> list[dict[str, Any]]:
    """Safe strategies to test."""
    candidates = [
        {
            "candidate_id": "dedupe_non_conflicting_literals",
            "strategy": "drop duplicate train literals only when all copies share y_category",
            "use_weighted_sampler": False,
        },
        {
            "candidate_id": "weighted_random_sampler",
            "strategy": "keep all train rows and sample inversely to y_category frequency",
            "use_weighted_sampler": True,
        },
    ]
    if debug:
        return candidates
    return candidates


def standard_split(config: SafeAugmentationConfig):
    """Load standard processed data and create the shared stratified split."""
    train, leaderboard = load_required_clean_data()
    split_config = SplitConfig(
        version_name=config.version_name,
        model_family="roberta_mean_safe_data_strategy",
        seed=config.seed,
        validation_size=config.validation_size,
    )
    train_df, val_df = split_train_validation(train, split_config)
    return train, train_df, val_df, leaderboard


def dedupe_non_conflicting_literals(train_df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate train literals only when their labels are not conflicting."""
    label_counts = train_df.groupby("Literal_required_clean")["y_category"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflicting_rows = train_df[train_df["Literal_required_clean"].isin(conflicting)]
    safe_rows = train_df[~train_df["Literal_required_clean"].isin(conflicting)]
    safe_deduped = safe_rows.drop_duplicates(subset=["Literal_required_clean", "y_category"])
    return pd.concat([safe_deduped, conflicting_rows], ignore_index=True)


def duplicate_report(train_df: pd.DataFrame, output_path: Path) -> Path:
    """Save duplicate statistics for the training split."""
    label_counts = train_df.groupby("Literal_required_clean")["y_category"].nunique()
    report = pd.DataFrame(
        [
            {"measure": "train_rows", "value": len(train_df)},
            {"measure": "exact_duplicate_rows", "value": int(train_df.duplicated().sum())},
            {
                "measure": "duplicate_literal_rows",
                "value": int(train_df.duplicated(subset=["Literal_required_clean"]).sum()),
            },
            {
                "measure": "conflicting_duplicate_literals",
                "value": int((label_counts > 1).sum()),
            },
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path


def make_datasets_from_frames(train_df, val_df, leaderboard_df, config):
    """Build datasets from already decided train/validation frames."""
    full_train, _ = load_required_clean_data()
    labels, label2id, id2label = build_label_mappings(full_train["y_category"])
    tokenizer = load_backbone_tokenizer(config.backbone)
    dataset_config = ICDDatasetConfig(max_length=config.max_length)
    train_dataset = ICDLiteralDataset(train_df, tokenizer, "train", label2id=label2id, config=dataset_config)
    val_dataset = ICDLiteralDataset(val_df, tokenizer, "validation", label2id=label2id, config=dataset_config)
    leaderboard_dataset = ICDLiteralDataset(leaderboard_df, tokenizer, "leaderboard", config=dataset_config)
    return train_dataset, val_dataset, leaderboard_dataset, labels, label2id, id2label


def make_sampler(train_df: pd.DataFrame, label2id: dict[str, int]) -> WeightedRandomSampler:
    """Create inverse-frequency weighted sampler from training split only."""
    counts = train_df["y_category"].astype(str).str.upper().value_counts()
    row_weights = train_df["y_category"].astype(str).str.upper().map(lambda label: 1.0 / counts[label])
    return WeightedRandomSampler(
        weights=torch.tensor(row_weights.values, dtype=torch.float32),
        num_samples=len(train_df),
        replacement=True,
    )


def train_candidate(config: SafeAugmentationConfig, candidate: dict[str, Any], train_df, val_df, leaderboard_df):
    """Train/evaluate/predict one safe data-strategy candidate."""
    set_seed(config.seed)
    device = get_device()
    candidate_version = f"{config.version_name}_{candidate['candidate_id']}"
    train_dataset, val_dataset, leaderboard_dataset, labels, label2id, id2label = make_datasets_from_frames(
        train_df, val_df, leaderboard_df, config
    )
    sampler = make_sampler(train_df, label2id) if candidate["use_weighted_sampler"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=config.batch_size, shuffle=False)

    model = RobertaMeanClassifier(config.backbone, len(labels), dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.use_amp and device.type == "cuda"))
    checkpoint_path = CHECKPOINTS_DIR / f"{candidate_version}.pt"

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config.use_amp, scaler)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, labels, id2label)
        row = {
            "candidate_id": candidate["candidate_id"],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        }
        print(row)
        history.append(row)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                config={**asdict(config), **candidate},
                metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    load_model_state(checkpoint_path, model, device)
    metrics, y_true, y_pred, val_probabilities = evaluate(model, val_loader, device, labels, id2label)
    metrics["best_epoch"] = best_epoch
    metrics["device"] = str(device)
    metrics["candidate"] = candidate
    metrics["train_rows_after_strategy"] = len(train_df)
    metrics["num_classes"] = len(labels)

    metrics_path = METRICS_DIR / f"{candidate_version}_metrics.json"
    save_json(metrics, metrics_path)
    per_class_path = METRICS_DIR / f"{candidate_version}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)
    history_path = LOGS_DIR / f"{candidate_version}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    config_path = LOGS_DIR / f"{candidate_version}_config.json"
    save_json({**asdict(config), **candidate}, config_path)

    val_detailed = build_detailed_predictions(
        val_df,
        y_pred,
        y_true=y_true,
        probabilities=val_probabilities,
        labels=labels,
        literal_col="Literal_required_clean",
    )
    val_predictions_path = PREDICTIONS_DIR / f"{candidate_version}_val_predictions.csv"
    val_detailed.to_csv(val_predictions_path, index=False)

    _, leaderboard_pred, leaderboard_probabilities = predict_leaderboard(model, leaderboard_loader, device, id2label)
    leaderboard_detailed = build_detailed_predictions(
        leaderboard_df,
        leaderboard_pred,
        probabilities=leaderboard_probabilities,
        labels=labels,
        id_col="id",
        literal_col="Literal_required_clean",
    )
    leaderboard_detailed_path = PREDICTIONS_DIR / f"{candidate_version}_leaderboard_detailed.csv"
    leaderboard_detailed.to_csv(leaderboard_detailed_path, index=False)
    submission_path = SUBMISSIONS_DIR / f"{candidate_version}_submission.csv"
    build_submission(leaderboard_df, leaderboard_pred).to_csv(submission_path, index=False)

    artifact_paths = {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "history": str(history_path),
        "validation_predictions": str(val_predictions_path),
        "per_class_metrics": str(per_class_path),
        "model_artifact": str(checkpoint_path),
        "leaderboard_detailed": str(leaderboard_detailed_path),
        "submission": str(submission_path),
    }
    summary_path = write_run_summary(candidate_version, {**asdict(config), **candidate}, metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)
    return {"candidate_version": candidate_version, "candidate": candidate, "metrics": metrics, "artifacts": artifact_paths}


def safe_augmentation_note(output_path: Path) -> Path:
    """Write the safety rationale for avoided augmentations."""
    text = """# Safe Clinical Text Augmentation Decision

We did not use random deletion, unverified synonym replacement, negation changes,
or back-translation in the final experiments. Clinical literals can be short and
semantically dense, so a small wording change can alter the correct ICD category.

The only tested strategies are data-handling strategies that do not invent new
clinical meaning: non-conflicting duplicate handling and class-balanced sampling.
Back-translation and synonym augmentation remain future work unless reviewed by
domain experts or controlled with verified medical terminology resources.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def run(config: SafeAugmentationConfig) -> dict[str, Any]:
    """Run safe data-strategy experiments."""
    ensure_project_dirs()
    set_seed(config.seed)
    _, train_df, val_df, leaderboard_df = standard_split(config)
    if config.debug:
        config = SafeAugmentationConfig(
            **{
                **asdict(config),
                "version_name": f"{VERSION_NAME}_debug",
                "batch_size": min(config.batch_size, 4),
                "max_epochs": min(config.max_epochs, 1),
                "patience": 1,
                "use_amp": False,
            }
        )
        train_df = train_df.sample(min(config.debug_train_examples, len(train_df)), random_state=config.seed)
        val_df = val_df.sample(min(config.debug_val_examples, len(val_df)), random_state=config.seed + 1)
        leaderboard_df = leaderboard_df.head(config.debug_leaderboard_examples)

    tables_dir = REPORTS_DIR / "tables"
    duplicate_report_path = duplicate_report(train_df, tables_dir / f"{config.version_name}_duplicate_report.csv")
    safety_note_path = safe_augmentation_note(REPORTS_DIR / "safe_augmentation_note.md")

    results = []
    for candidate in candidate_grid(debug=config.debug):
        candidate_train_df = train_df.copy()
        if candidate["candidate_id"] == "dedupe_non_conflicting_literals":
            candidate_train_df = dedupe_non_conflicting_literals(candidate_train_df)
        results.append(train_candidate(config, candidate, candidate_train_df, val_df, leaderboard_df))

    rows = []
    if not config.debug:
        v05_path = METRICS_DIR / "v05_roberta_mean_metrics.json"
        if v05_path.exists():
            v05 = json.loads(v05_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "candidate_version": "v05_roberta_mean",
                    "strategy": "original train reference",
                    "accuracy": v05["accuracy"],
                    "macro_f1": v05["macro_f1"],
                    "weighted_f1": v05["weighted_f1"],
                    "best_epoch": v05.get("best_epoch"),
                    "train_rows_after_strategy": 10960,
                    "is_reference": True,
                }
            )
    for result in results:
        m = result["metrics"]
        rows.append(
            {
                "candidate_version": result["candidate_version"],
                "strategy": result["candidate"]["strategy"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "best_epoch": m["best_epoch"],
                "train_rows_after_strategy": m["train_rows_after_strategy"],
                "is_reference": False,
            }
        )
    results_df = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1", "weighted_f1"], ascending=[False, False, False])
    results_path = tables_dir / "v08_data_strategy_results.csv"
    results_df.to_csv(results_path, index=False)

    best_non_reference = results_df[~results_df["is_reference"]].iloc[0]
    best_result = next(result for result in results if result["candidate_version"] == best_non_reference["candidate_version"])
    canonical_submission = SUBMISSIONS_DIR / f"{config.version_name}_submission.csv"
    pd.read_csv(best_result["artifacts"]["submission"]).to_csv(canonical_submission, index=False)
    canonical_metrics = METRICS_DIR / f"{config.version_name}_metrics.json"
    save_json(best_result["metrics"], canonical_metrics)

    artifact_paths = {
        "data_strategy_results": str(results_path),
        "duplicate_report": str(duplicate_report_path),
        "safety_note": str(safety_note_path),
        "canonical_submission": str(canonical_submission),
        "canonical_metrics": str(canonical_metrics),
    }
    summary_path = write_run_summary(config.version_name, asdict(config), best_result["metrics"], artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)
    return {
        "version_name": config.version_name,
        "best_candidate": best_result["candidate_version"],
        "metrics": best_result["metrics"],
        "artifacts": artifact_paths,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run safe RoBERTa data-strategy experiments.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SafeAugmentationConfig(
        debug=args.debug,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    result = run(config)
    print(
        {
            "version_name": result["version_name"],
            "best_candidate": result["best_candidate"],
            "accuracy": result["metrics"].get("accuracy"),
            "macro_f1": result["metrics"].get("macro_f1"),
            "weighted_f1": result["metrics"].get("weighted_f1"),
            "artifacts": result["artifacts"],
        }
    )


if __name__ == "__main__":
    main()

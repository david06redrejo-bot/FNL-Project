"""v07: controlled hyperparameter tuning for RoBERTa mean pooling.

The search is staged on purpose:

- Stage A: quick small-subset checks.
- Stage B: medium promising settings.
- Stage C: final full run for the recommended configuration.

This avoids an uncontrolled grid over expensive Transformer runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

from models.v05_roberta_mean import (
    RobertaMeanClassifier,
    RobertaMeanConfig,
    evaluate,
    load_model_state,
    make_datasets,
    predict_leaderboard,
)
from src.datasets import BACKBONE_CHECKPOINT, DEFAULT_MAX_LENGTH
from src.inference import build_detailed_predictions, build_submission
from src.metrics import per_class_metrics_table
from src.paths import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SUBMISSIONS_DIR, ensure_project_dirs
from src.reporting import write_run_summary
from src.training import save_torch_checkpoint
from src.utils import get_device, save_json, set_seed


VERSION_NAME = "v07_roberta_mean_tuning"


@dataclass
class TuningConfig:
    """Top-level tuning configuration."""

    version_name: str = VERSION_NAME
    backbone: str = BACKBONE_CHECKPOINT
    seed: int = 42
    validation_size: float = 0.2
    stage: str = "stage_a"
    use_amp: bool = False


def stage_candidates(stage: str) -> list[dict[str, Any]]:
    """Return controlled candidates for a stage."""
    stage_a = [
        {
            "candidate_id": "a_baseline_32_lr2e5",
            "stage": "stage_a",
            "max_length": 32,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "scheduler": "none",
            "gradient_clip": 1.0,
            "max_epochs": 1,
            "patience": 1,
            "train_limit": 512,
            "val_limit": 256,
            "use_amp": False,
        },
        {
            "candidate_id": "a_warmup_dropout_32_lr2e5",
            "stage": "stage_a",
            "max_length": 32,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "dropout": 0.2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 1,
            "patience": 1,
            "train_limit": 512,
            "val_limit": 256,
            "use_amp": False,
        },
        {
            "candidate_id": "a_longer_64_lr3e5",
            "stage": "stage_a",
            "max_length": 64,
            "learning_rate": 3e-5,
            "batch_size": 32,
            "dropout": 0.1,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 1,
            "patience": 1,
            "train_limit": 512,
            "val_limit": 256,
            "use_amp": False,
        },
        {
            "candidate_id": "a_amp_32_lr2e5",
            "stage": "stage_a",
            "max_length": 32,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 1,
            "patience": 1,
            "train_limit": 512,
            "val_limit": 256,
            "use_amp": True,
        },
    ]
    stage_b = [
        {
            "candidate_id": "b_warmup_32_lr2e5",
            "stage": "stage_b",
            "max_length": 32,
            "learning_rate": 2e-5,
            "batch_size": 64,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 4,
            "patience": 2,
            "train_limit": 4096,
            "val_limit": 1024,
            "use_amp": False,
        },
        {
            "candidate_id": "b_lr3e5_dropout2",
            "stage": "stage_b",
            "max_length": 32,
            "learning_rate": 3e-5,
            "batch_size": 64,
            "dropout": 0.2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 4,
            "patience": 2,
            "train_limit": 4096,
            "val_limit": 1024,
            "use_amp": False,
        },
    ]
    stage_c = [
        {
            "candidate_id": "c_recommended_32_lr2e5_warmup006_clip",
            "stage": "stage_c",
            "max_length": 32,
            "learning_rate": 2e-5,
            "batch_size": 128,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "scheduler": "linear",
            "gradient_clip": 1.0,
            "max_epochs": 50,
            "patience": 10,
            "train_limit": None,
            "val_limit": None,
            "use_amp": False,
        }
    ]
    if stage == "stage_a":
        return stage_a
    if stage == "stage_b":
        return stage_b
    if stage == "stage_c":
        return stage_c
    if stage == "all":
        return stage_a + stage_b + stage_c
    raise ValueError(f"Unknown stage: {stage}")


def make_candidate_datasets(candidate: dict[str, Any], config: TuningConfig):
    """Build datasets while preserving the standard split before sub-sampling."""
    roberta_config = RobertaMeanConfig(
        version_name=VERSION_NAME,
        backbone=config.backbone,
        seed=config.seed,
        max_length=int(candidate["max_length"]),
        batch_size=int(candidate["batch_size"]),
        validation_size=config.validation_size,
        dropout=float(candidate["dropout"]),
        learning_rate=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
        max_epochs=int(candidate["max_epochs"]),
        patience=int(candidate["patience"]),
        use_amp=bool(candidate["use_amp"]),
    )
    train_df, val_df, leaderboard_df, *_ = make_datasets(roberta_config)
    if candidate["train_limit"]:
        train_df = train_df.sample(
            min(int(candidate["train_limit"]), len(train_df)),
            random_state=config.seed,
        ).reset_index(drop=True)
    if candidate["val_limit"]:
        val_df = val_df.sample(
            min(int(candidate["val_limit"]), len(val_df)),
            random_state=config.seed + 1,
        ).reset_index(drop=True)
    # Rebuild datasets after stage-specific sub-sampling.
    from src.datasets import ICDDatasetConfig, ICDLiteralDataset, build_label_mappings, load_backbone_tokenizer

    full_train, _ = __import__("src.training", fromlist=["load_required_clean_data"]).load_required_clean_data()
    labels, label2id, id2label = build_label_mappings(full_train["y_category"])
    tokenizer = load_backbone_tokenizer(config.backbone)
    dataset_config = ICDDatasetConfig(max_length=int(candidate["max_length"]))
    train_dataset = ICDLiteralDataset(train_df, tokenizer, "train", label2id=label2id, config=dataset_config)
    val_dataset = ICDLiteralDataset(val_df, tokenizer, "validation", label2id=label2id, config=dataset_config)
    leaderboard_dataset = ICDLiteralDataset(leaderboard_df, tokenizer, "leaderboard", config=dataset_config)
    return train_df, val_df, leaderboard_df, train_dataset, val_dataset, leaderboard_dataset, labels, label2id, id2label


def train_epoch_with_controls(model, dataloader, optimizer, scheduler, device, candidate, scaler):
    """Train one epoch with optional AMP, scheduler, and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    amp_enabled = bool(candidate["use_amp"] and device.type == "cuda")
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(candidate["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(candidate["gradient_clip"]))
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = int(batch["labels"].shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
    return total_loss / max(total_examples, 1)


def plot_history(history_path: Path, figure_path: Path, title: str) -> Path:
    """Plot training curves for a candidate."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_accuracy"], marker="o", label="accuracy")
    axes[1].plot(history["epoch"], history["macro_f1"], marker="o", label="macro F1")
    axes[1].plot(history["epoch"], history["weighted_f1"], marker="o", label="weighted F1")
    axes[1].set_title(f"{title} Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 0.7)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_candidate(candidate: dict[str, Any], config: TuningConfig, device):
    """Train/evaluate/predict one tuning candidate."""
    set_seed(config.seed)
    candidate_version = f"{config.version_name}_{candidate['candidate_id']}"
    (
        _,
        val_df,
        leaderboard_df,
        train_dataset,
        val_dataset,
        leaderboard_dataset,
        labels,
        label2id,
        id2label,
    ) = make_candidate_datasets(candidate, config)
    train_loader = DataLoader(train_dataset, batch_size=int(candidate["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(candidate["batch_size"]), shuffle=False)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=int(candidate["batch_size"]), shuffle=False)

    model = RobertaMeanClassifier(
        backbone_name=config.backbone,
        num_classes=len(labels),
        dropout=float(candidate["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    total_steps = len(train_loader) * int(candidate["max_epochs"])
    scheduler = None
    if candidate["scheduler"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * float(candidate["warmup_ratio"])),
            num_training_steps=total_steps,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(candidate["use_amp"] and device.type == "cuda"))
    checkpoint_path = CHECKPOINTS_DIR / f"{candidate_version}.pt"

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, int(candidate["max_epochs"]) + 1):
        train_loss = train_epoch_with_controls(model, train_loader, optimizer, scheduler, device, candidate, scaler)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, labels, id2label)
        row = {
            "epoch": epoch,
            "candidate_id": candidate["candidate_id"],
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        print(row)
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
        if epochs_without_improvement >= int(candidate["patience"]):
            break

    load_model_state(checkpoint_path, model, device)
    metrics, y_true, y_pred, val_probabilities = evaluate(model, val_loader, device, labels, id2label)
    metrics["best_epoch"] = best_epoch
    metrics["device"] = str(device)
    metrics["candidate"] = candidate
    metrics["hidden_size"] = int(model.backbone.config.hidden_size)
    metrics["num_classes"] = len(labels)
    metrics["label2id"] = label2id
    metrics["id2label"] = {str(key): value for key, value in id2label.items()}

    metrics_path = METRICS_DIR / f"{candidate_version}_metrics.json"
    save_json(metrics, metrics_path)
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

    figure_path = REPORTS_DIR / "figures" / f"{candidate_version}_training_curves.png"
    plot_history(history_path, figure_path, candidate["candidate_id"])

    return {
        "candidate_version": candidate_version,
        "candidate": candidate,
        "metrics": metrics,
        "artifacts": {
            "config": str(config_path),
            "metrics": str(metrics_path),
            "history": str(history_path),
            "validation_predictions": str(val_predictions_path),
            "model_artifact": str(checkpoint_path),
            "leaderboard_detailed": str(leaderboard_detailed_path),
            "submission": str(submission_path),
            "training_curves": str(figure_path),
        },
    }


def run(config: TuningConfig) -> dict[str, Any]:
    """Run a staged tuning search."""
    ensure_project_dirs()
    set_seed(config.seed)
    device = get_device()
    results = [run_candidate(candidate, config, device) for candidate in stage_candidates(config.stage)]

    rows = []
    for result in results:
        metrics = result["metrics"]
        candidate = result["candidate"]
        rows.append(
            {
                "candidate_version": result["candidate_version"],
                **candidate,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "best_epoch": metrics["best_epoch"],
                "device": metrics["device"],
            }
        )
    results_df = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    )
    results_path = REPORTS_DIR / "tables" / "v07_tuning_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        previous = pd.read_csv(results_path)
        results_df = pd.concat([previous, results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(subset=["candidate_version"], keep="last")
        results_df = results_df.sort_values(["accuracy", "macro_f1", "weighted_f1"], ascending=[False, False, False])
    results_df.to_csv(results_path, index=False)

    best_version = results_df.iloc[0]["candidate_version"]
    best_result = next((result for result in results if result["candidate_version"] == best_version), None)
    if best_result is None:
        best_row = results_df.iloc[0].to_dict()
        best_submission = SUBMISSIONS_DIR / f"{best_version}_submission.csv"
        best_detailed = PREDICTIONS_DIR / f"{best_version}_leaderboard_detailed.csv"
        best_metrics_path = METRICS_DIR / f"{best_version}_metrics.json"
        best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    else:
        best_row = results_df.iloc[0].to_dict()
        best_submission = Path(best_result["artifacts"]["submission"])
        best_detailed = Path(best_result["artifacts"]["leaderboard_detailed"])
        best_metrics = best_result["metrics"]

    canonical_submission = SUBMISSIONS_DIR / f"{config.version_name}_submission.csv"
    pd.read_csv(best_submission).to_csv(canonical_submission, index=False)
    canonical_detailed = PREDICTIONS_DIR / f"{config.version_name}_leaderboard_detailed.csv"
    pd.read_csv(best_detailed).to_csv(canonical_detailed, index=False)
    canonical_metrics = METRICS_DIR / f"{config.version_name}_metrics.json"
    save_json(best_metrics, canonical_metrics)

    artifact_paths = {
        "tuning_results": str(results_path),
        "canonical_submission": str(canonical_submission),
        "canonical_leaderboard_detailed": str(canonical_detailed),
        "canonical_metrics": str(canonical_metrics),
    }
    summary_path = write_run_summary(config.version_name, asdict(config), best_metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)

    return {
        "version_name": config.version_name,
        "best_candidate": best_row,
        "metrics": best_metrics,
        "artifacts": artifact_paths,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Controlled RoBERTa mean-pooling tuning.")
    parser.add_argument("--stage", choices=["stage_a", "stage_b", "stage_c", "all"], default="stage_a")
    parser.add_argument("--amp", action="store_true", help="Enable AMP where candidate allows it.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = TuningConfig(stage=args.stage, use_amp=args.amp)
    result = run(config)
    print(
        {
            "version_name": result["version_name"],
            "best_candidate": result["best_candidate"].get("candidate_version"),
            "accuracy": result["metrics"].get("accuracy"),
            "macro_f1": result["metrics"].get("macro_f1"),
            "weighted_f1": result["metrics"].get("weighted_f1"),
            "artifacts": result["artifacts"],
        }
    )


if __name__ == "__main__":
    main()

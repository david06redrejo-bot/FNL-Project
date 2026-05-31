"""v06: imbalance-aware RoBERTa mean-pooling variants.

This script starts from the v05 mean-pooling baseline and compares loss
functions designed for long-tail category distributions:

1. standard CrossEntropyLoss reference,
2. class-weighted CrossEntropyLoss,
3. focal loss with configurable gamma.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v05_roberta_mean import (
    RobertaMeanClassifier,
    RobertaMeanConfig,
    evaluate,
    load_model_state,
    make_datasets,
    predict_leaderboard,
    train_epoch,
)
from src.datasets import BACKBONE_CHECKPOINT, DEFAULT_MAX_LENGTH
from src.inference import build_detailed_predictions, build_submission
from src.metrics import per_class_metrics_table
from src.paths import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    ensure_project_dirs,
)
from src.reporting import write_run_summary
from src.training import save_torch_checkpoint
from src.utils import get_device, save_json, set_seed


VERSION_NAME = "v06_roberta_mean_imbalance_aware"


@dataclass
class ImbalanceAwareConfig:
    """Configuration for imbalance-aware mean-pooling experiments."""

    version_name: str = VERSION_NAME
    backbone: str = BACKBONE_CHECKPOINT
    pooling: str = "mean"
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
    debug_train_examples: int = 16
    debug_val_examples: int = 16
    debug_leaderboard_examples: int = 16
    selection_metric: str = "accuracy"
    focal_gammas: tuple[float, ...] = (1.0, 2.0)


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def candidate_grid(focal_gammas: tuple[float, ...] = (1.0, 2.0)) -> list[dict[str, Any]]:
    """Loss-function grid for imbalance-aware experiments."""
    candidates = [
        {
            "candidate_id": "class_weight_balanced",
            "loss_name": "class_weighted_cross_entropy",
            "focal_gamma": None,
            "use_class_weights": True,
        },
    ]
    for gamma in focal_gammas:
        gamma_label = str(gamma).replace(".", "p")
        if gamma_label.endswith("p0"):
            gamma_label = gamma_label[:-2]
        candidates.append(
            {
                "candidate_id": f"focal_gamma{gamma_label}",
                "loss_name": "focal_loss",
                "focal_gamma": float(gamma),
                "use_class_weights": False,
            }
        )
    return candidates


def make_roberta_config(config: ImbalanceAwareConfig, version_name: str) -> RobertaMeanConfig:
    """Create the compatible v05 config object used by shared dataset helpers."""
    return RobertaMeanConfig(
        version_name=version_name,
        backbone=config.backbone,
        seed=config.seed,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        dropout=config.dropout,
        validation_size=config.validation_size,
        use_amp=config.use_amp,
        debug=config.debug,
        debug_train_examples=config.debug_train_examples,
        debug_val_examples=config.debug_val_examples,
        debug_leaderboard_examples=config.debug_leaderboard_examples,
    )


def compute_balanced_class_weights(train_df: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    """Compute balanced class weights from the training split only."""
    total = len(train_df)
    num_classes = len(label2id)
    counts = train_df["y_category"].astype(str).str.upper().value_counts()
    rows = []
    for label, label_id in sorted(label2id.items(), key=lambda item: item[1]):
        count = int(counts.get(label, 0))
        weight = float(total / (num_classes * count)) if count > 0 else 0.0
        rows.append(
            {
                "y_category": label,
                "label_id": label_id,
                "train_count": count,
                "class_weight": weight,
            }
        )
    return pd.DataFrame(rows)


def loss_for_candidate(candidate: dict[str, Any], weights_df: pd.DataFrame, device):
    """Build the criterion for a candidate."""
    if candidate["loss_name"] == "class_weighted_cross_entropy":
        weights = torch.tensor(weights_df["class_weight"].values, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    if candidate["loss_name"] == "focal_loss":
        return FocalLoss(gamma=float(candidate["focal_gamma"]))
    return nn.CrossEntropyLoss()


def save_confusion_matrix(metrics: dict[str, Any], labels: list[str], path: Path) -> Path:
    """Save confusion matrix as a labeled CSV."""
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    matrix.index.name = "true_y_category"
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)
    return path


def recall_comparison(best_per_class_path: Path, output_path: Path) -> Path:
    """Compare per-class recall for v05 mean vs the best v06 candidate."""
    v05_path = METRICS_DIR / "v05_roberta_mean_per_class_metrics.csv"
    best = pd.read_csv(best_per_class_path)
    best = best.rename(columns={"recall": "v06_recall", "f1_score": "v06_f1"})
    if v05_path.exists():
        v05 = pd.read_csv(v05_path).rename(
            columns={"recall": "v05_recall", "f1_score": "v05_f1"}
        )
        comparison = v05[["y_category", "v05_recall", "v05_f1"]].merge(
            best[["y_category", "v06_recall", "v06_f1"]],
            on="y_category",
            how="outer",
        )
        comparison["recall_delta_v06_minus_v05"] = comparison["v06_recall"] - comparison["v05_recall"]
        comparison["f1_delta_v06_minus_v05"] = comparison["v06_f1"] - comparison["v05_f1"]
    else:
        comparison = best[["y_category", "v06_recall", "v06_f1"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def run_candidate(
    config: ImbalanceAwareConfig,
    candidate: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    train_dataset,
    val_dataset,
    leaderboard_dataset,
    labels: list[str],
    label2id: dict[str, int],
    id2label: dict[int, str],
    weights_df: pd.DataFrame,
    device,
) -> dict[str, Any]:
    """Train, evaluate, and predict one imbalance-aware candidate."""
    candidate_version = f"{config.version_name}_{candidate['candidate_id']}"
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=config.batch_size, shuffle=False)

    model = RobertaMeanClassifier(
        backbone_name=config.backbone,
        num_classes=len(labels),
        dropout=config.dropout,
    ).to(device)
    model.loss_fn = loss_for_candidate(candidate, weights_df, device)
    hidden_size = int(model.backbone.config.hidden_size)
    if hidden_size != 768:
        raise ValueError(f"Expected hidden size 768, found {hidden_size}.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config.use_amp and device.type == "cuda"),
    )
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
        print(row)
        if epochs_without_improvement >= config.patience:
            break

    load_model_state(checkpoint_path, model, device)
    metrics, y_true, y_pred, val_probabilities = evaluate(model, val_loader, device, labels, id2label)
    metrics["best_epoch"] = best_epoch
    metrics["device"] = str(device)
    metrics["hidden_size"] = hidden_size
    metrics["num_classes"] = len(labels)
    metrics["label2id"] = label2id
    metrics["id2label"] = {str(key): value for key, value in id2label.items()}
    metrics["candidate"] = candidate

    metrics_path = METRICS_DIR / f"{candidate_version}_metrics.json"
    save_json(metrics, metrics_path)
    per_class_path = METRICS_DIR / f"{candidate_version}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)
    confusion_path = METRICS_DIR / f"{candidate_version}_confusion_matrix.csv"
    save_confusion_matrix(metrics, labels, confusion_path)

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

    _, leaderboard_pred, leaderboard_probabilities = predict_leaderboard(
        model,
        leaderboard_loader,
        device,
        id2label,
    )
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

    submission = build_submission(leaderboard_df, leaderboard_pred)
    submission_path = SUBMISSIONS_DIR / f"{candidate_version}_submission.csv"
    submission.to_csv(submission_path, index=False)

    artifact_paths = {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "history": str(history_path),
        "validation_predictions": str(val_predictions_path),
        "per_class_metrics": str(per_class_path),
        "confusion_matrix": str(confusion_path),
        "model_artifact": str(checkpoint_path),
        "leaderboard_detailed": str(leaderboard_detailed_path),
        "submission": str(submission_path),
    }
    summary_path = write_run_summary(candidate_version, {**asdict(config), **candidate}, metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)

    return {
        "candidate_version": candidate_version,
        "candidate": candidate,
        "metrics": metrics,
        "artifacts": artifact_paths,
    }


def run(config: ImbalanceAwareConfig) -> dict[str, Any]:
    """Run the imbalance-aware candidate grid."""
    ensure_project_dirs()
    set_seed(config.seed)
    if config.debug:
        config = ImbalanceAwareConfig(
            **{
                **asdict(config),
                "version_name": f"{VERSION_NAME}_debug",
                "batch_size": min(config.batch_size, 4),
                "max_epochs": min(config.max_epochs, 1),
                "patience": 1,
                "use_amp": False,
            }
        )

    device = get_device()
    roberta_config = make_roberta_config(config, config.version_name)
    (
        train_df,
        val_df,
        leaderboard_df,
        train_dataset,
        val_dataset,
        leaderboard_dataset,
        labels,
        label2id,
        id2label,
    ) = make_datasets(roberta_config)
    if len(labels) != 36:
        raise ValueError(f"Expected 36 labels, found {len(labels)}.")

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    weights_df = compute_balanced_class_weights(train_df, label2id)
    weights_csv_path = tables_dir / f"{config.version_name}_class_weights.csv"
    weights_json_path = LOGS_DIR / f"{config.version_name}_class_weights.json"
    weights_df.to_csv(weights_csv_path, index=False)
    save_json(
        {
            row["y_category"]: {
                "label_id": int(row["label_id"]),
                "train_count": int(row["train_count"]),
                "class_weight": float(row["class_weight"]),
            }
            for _, row in weights_df.iterrows()
        },
        weights_json_path,
    )

    results = []
    for candidate in candidate_grid(config.focal_gammas):
        results.append(
            run_candidate(
                config,
                candidate,
                train_df,
                val_df,
                leaderboard_df,
                train_dataset,
                val_dataset,
                leaderboard_dataset,
                labels,
                label2id,
                id2label,
                weights_df,
                device,
            )
        )

    grid_rows = []
    v05_metrics_path = METRICS_DIR / "v05_roberta_mean_metrics.json"
    if v05_metrics_path.exists() and not config.debug:
        import json

        v05_metrics = json.loads(v05_metrics_path.read_text(encoding="utf-8"))
        grid_rows.append(
            {
                "candidate_version": "v05_roberta_mean",
                "loss_name": "standard_cross_entropy",
                "focal_gamma": np.nan,
                "use_class_weights": False,
                "accuracy": v05_metrics["accuracy"],
                "macro_f1": v05_metrics["macro_f1"],
                "weighted_f1": v05_metrics["weighted_f1"],
                "best_epoch": v05_metrics.get("best_epoch"),
                "is_reference_v05": True,
            }
        )
    for result in results:
        metrics = result["metrics"]
        candidate = result["candidate"]
        grid_rows.append(
            {
                "candidate_version": result["candidate_version"],
                "loss_name": candidate["loss_name"],
                "focal_gamma": candidate["focal_gamma"],
                "use_class_weights": candidate["use_class_weights"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "best_epoch": metrics["best_epoch"],
                "is_reference_v05": False,
            }
        )

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=[False, False, False],
    )
    grid_path = tables_dir / "v06_imbalance_aware_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_non_reference = grid_df[~grid_df["is_reference_v05"]].iloc[0]
    best_result = next(
        result
        for result in results
        if result["candidate_version"] == best_non_reference["candidate_version"]
    )
    best_per_class_path = Path(best_result["artifacts"]["per_class_metrics"])
    recall_path = tables_dir / "v06_per_class_recall_vs_v05.csv"
    recall_comparison(best_per_class_path, recall_path)

    # Also expose the best v06 candidate under the canonical version name.
    best_metrics = dict(best_result["metrics"])
    best_metrics["grid_results_path"] = str(grid_path)
    best_metrics["class_weights_csv"] = str(weights_csv_path)
    best_metrics["class_weights_json"] = str(weights_json_path)
    best_metrics["per_class_recall_comparison"] = str(recall_path)
    canonical_metrics_path = METRICS_DIR / f"{config.version_name}_metrics.json"
    save_json(best_metrics, canonical_metrics_path)
    canonical_submission_path = SUBMISSIONS_DIR / f"{config.version_name}_submission.csv"
    pd.read_csv(best_result["artifacts"]["submission"]).to_csv(canonical_submission_path, index=False)
    canonical_detailed_path = PREDICTIONS_DIR / f"{config.version_name}_leaderboard_detailed.csv"
    pd.read_csv(best_result["artifacts"]["leaderboard_detailed"]).to_csv(canonical_detailed_path, index=False)

    artifact_paths = {
        "grid_results": str(grid_path),
        "class_weights_csv": str(weights_csv_path),
        "class_weights_json": str(weights_json_path),
        "best_candidate_metrics": str(best_result["artifacts"]["metrics"]),
        "canonical_metrics": str(canonical_metrics_path),
        "per_class_recall_comparison": str(recall_path),
        "canonical_submission": str(canonical_submission_path),
        "canonical_leaderboard_detailed": str(canonical_detailed_path),
    }
    summary_path = write_run_summary(config.version_name, asdict(config), best_metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)

    return {
        "version_name": config.version_name,
        "best_candidate": best_result["candidate_version"],
        "metrics": best_metrics,
        "artifacts": artifact_paths,
        "grid": grid_df.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run imbalance-aware RoBERTa mean-pooling variants.")
    parser.add_argument("--debug", action="store_true", help="Run tiny debug jobs.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Token max_length.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument(
        "--focal-gammas",
        default="1,2",
        help="Comma-separated focal-loss gamma values.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = ImbalanceAwareConfig(
        debug=args.debug,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        use_amp=args.amp,
        focal_gammas=tuple(float(value) for value in args.focal_gammas.split(",") if value.strip()),
    )
    result = run(config)
    metrics = result["metrics"]
    print(
        {
            "version_name": result["version_name"],
            "best_candidate": result["best_candidate"],
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "best_epoch": metrics.get("best_epoch"),
            "artifacts": result["artifacts"],
        }
    )


if __name__ == "__main__":
    main()

"""v05: Spanish biomedical-clinical RoBERTa with mean pooling.

Architecture:
1. AutoModel backbone.
2. Mean pooling over non-padding tokens.
3. Dropout(0.1).
4. Linear(hidden_size, num_classes).
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
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    BACKBONE_CHECKPOINT,
    DEFAULT_MAX_LENGTH,
    ICDDatasetConfig,
    ICDLiteralDataset,
    build_label_mappings,
    load_backbone_tokenizer,
)
from src.inference import build_detailed_predictions, build_submission
from src.metrics import compute_full_metrics, per_class_metrics_table, softmax
from src.paths import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SUBMISSIONS_DIR,
    ensure_project_dirs,
)
from src.reporting import write_run_summary
from src.training import ModelRunConfig as SplitConfig
from src.training import load_required_clean_data, save_torch_checkpoint, split_train_validation
from src.utils import get_device, save_json, set_seed


VERSION_NAME = "v05_roberta_mean"


@dataclass
class RobertaMeanConfig:
    """Configuration for the mean-pooling RoBERTa baseline."""

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


class RobertaMeanClassifier(nn.Module):
    """RoBERTa classifier using attention-mask-aware mean pooling."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            features = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(features))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def make_datasets(config: RobertaMeanConfig):
    """Load data and build train/validation/leaderboard datasets."""
    train, leaderboard = load_required_clean_data()
    labels, label2id, id2label = build_label_mappings(train["y_category"])
    split_config = SplitConfig(
        version_name=config.version_name,
        model_family="roberta_mean",
        seed=config.seed,
        validation_size=config.validation_size,
    )
    train_df, val_df = split_train_validation(train, split_config)

    if config.debug:
        train_df = train_df.sample(
            min(config.debug_train_examples, len(train_df)),
            random_state=config.seed,
        ).reset_index(drop=True)
        val_df = val_df.sample(
            min(config.debug_val_examples, len(val_df)),
            random_state=config.seed + 1,
        ).reset_index(drop=True)
        leaderboard = leaderboard.head(config.debug_leaderboard_examples).copy()

    tokenizer = load_backbone_tokenizer(config.backbone)
    dataset_config = ICDDatasetConfig(max_length=config.max_length)
    train_dataset = ICDLiteralDataset(
        train_df,
        tokenizer,
        mode="train",
        label2id=label2id,
        config=dataset_config,
    )
    val_dataset = ICDLiteralDataset(
        val_df,
        tokenizer,
        mode="validation",
        label2id=label2id,
        config=dataset_config,
    )
    leaderboard_dataset = ICDLiteralDataset(
        leaderboard,
        tokenizer,
        mode="leaderboard",
        config=dataset_config,
    )
    return train_df, val_df, leaderboard, train_dataset, val_dataset, leaderboard_dataset, labels, label2id, id2label


def train_epoch(model, dataloader, optimizer, device, use_amp: bool, scaler):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
    return total_loss / max(total_examples, 1)


def evaluate(model, dataloader, device, labels: list[str], id2label: dict[int, str]):
    """Evaluate model and collect logits/predictions."""
    model.eval()
    all_logits = []
    all_label_ids = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            batch_size = int(batch["labels"].shape[0])
            total_examples += batch_size
            total_loss += float(outputs["loss"].detach().cpu()) * batch_size
            all_logits.append(outputs["logits"].detach().cpu().numpy())
            all_label_ids.append(batch["labels"].detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    label_ids = np.concatenate(all_label_ids, axis=0)
    probabilities = softmax(logits)
    pred_ids = probabilities.argmax(axis=1)
    y_true = [id2label[int(idx)] for idx in label_ids]
    y_pred = [id2label[int(idx)] for idx in pred_ids]
    metrics = compute_full_metrics(y_true, y_pred, labels=labels, y_proba=probabilities)
    metrics["val_loss"] = total_loss / max(total_examples, 1)
    return metrics, y_true, y_pred, probabilities


def predict_leaderboard(model, dataloader, device, id2label: dict[int, str]):
    """Predict leaderboard examples and return ids, predictions, probabilities."""
    model.eval()
    ids = []
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch_ids = batch["id"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            ids.extend(batch_ids)
            all_logits.append(outputs["logits"].detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    probabilities = softmax(logits)
    pred_ids = probabilities.argmax(axis=1)
    y_pred = [id2label[int(idx)] for idx in pred_ids]
    return ids, y_pred, probabilities


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: RobertaMeanConfig, metrics: dict[str, Any]):
    """Save model checkpoint with config and metrics."""
    return save_torch_checkpoint(
        path,
        model,
        optimizer=optimizer,
        epoch=epoch,
        config=asdict(config),
        metrics=metrics,
    )


def load_model_state(path: Path, model, device):
    """Load the model state from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run(config: RobertaMeanConfig) -> dict[str, Any]:
    """Train/evaluate/predict the mean-pooling baseline."""
    ensure_project_dirs()
    set_seed(config.seed)
    device = get_device()
    if config.debug:
        config = RobertaMeanConfig(
            **{
                **asdict(config),
                "version_name": f"{VERSION_NAME}_debug",
                "batch_size": min(config.batch_size, 4),
                "max_epochs": min(config.max_epochs, 1),
                "patience": 1,
                "use_amp": False,
            }
        )

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
    ) = make_datasets(config)
    if len(labels) != 36:
        raise ValueError(f"Expected 36 labels, found {len(labels)}.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=config.batch_size, shuffle=False)

    model = RobertaMeanClassifier(
        backbone_name=config.backbone,
        num_classes=len(labels),
        dropout=config.dropout,
    ).to(device)
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
    checkpoint_path = CHECKPOINTS_DIR / f"{config.version_name}.pt"

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config.use_amp, scaler)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, labels, id2label)
        row = {
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
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config, val_metrics)
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

    metrics_path = METRICS_DIR / f"{config.version_name}_metrics.json"
    save_json(metrics, metrics_path)
    per_class_path = METRICS_DIR / f"{config.version_name}_per_class_metrics.csv"
    per_class_metrics_table(metrics).to_csv(per_class_path, index=False)

    history_path = LOGS_DIR / f"{config.version_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    config_path = LOGS_DIR / f"{config.version_name}_config.json"
    save_json(asdict(config), config_path)

    val_detailed = build_detailed_predictions(
        val_df,
        y_pred,
        y_true=y_true,
        probabilities=val_probabilities,
        labels=labels,
        literal_col="Literal_required_clean",
    )
    val_predictions_path = PREDICTIONS_DIR / f"{config.version_name}_val_predictions.csv"
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
    leaderboard_detailed_path = PREDICTIONS_DIR / f"{config.version_name}_leaderboard_detailed.csv"
    leaderboard_detailed.to_csv(leaderboard_detailed_path, index=False)

    submission = build_submission(leaderboard_df, leaderboard_pred)
    submission_path = SUBMISSIONS_DIR / f"{config.version_name}_submission.csv"
    submission.to_csv(submission_path, index=False)

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
    summary_path = write_run_summary(config.version_name, asdict(config), metrics, artifact_paths)
    artifact_paths["run_summary"] = str(summary_path)

    return {
        "version_name": config.version_name,
        "metrics": metrics,
        "artifacts": artifact_paths,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train/evaluate RoBERTa mean-pooling baseline.")
    parser.add_argument("--debug", action="store_true", help="Run a tiny debug training job.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Token max_length.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = RobertaMeanConfig(
        debug=args.debug,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        use_amp=args.amp,
    )
    result = run(config)
    metrics = result["metrics"]
    print(
        {
            "version_name": result["version_name"],
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "best_epoch": metrics.get("best_epoch"),
            "artifacts": result["artifacts"],
        }
    )


if __name__ == "__main__":
    main()

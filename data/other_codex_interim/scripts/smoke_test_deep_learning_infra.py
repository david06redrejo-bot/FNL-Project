"""Tiny smoke test for RoBERTa-style deep learning infrastructure.

This script intentionally does not load or train the full RoBERTa model. It
loads the real project tokenizer, builds `ICDLiteralDataset` objects, and trains
a tiny classifier on a handful of examples to verify batching, metrics,
checkpointing, and submission formatting.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    DEFAULT_MAX_LENGTH,
    ICDDatasetConfig,
    ICDLiteralDataset,
    build_label_mappings,
    load_backbone_tokenizer,
)
from src.inference import build_submission
from src.paths import CHECKPOINTS_DIR, LOGS_DIR
from src.training import (
    evaluate_torch_model,
    load_required_clean_data,
    load_torch_checkpoint,
    save_torch_checkpoint,
    train_one_epoch,
)
from src.utils import get_device, save_json, set_seed


class TinyLiteralClassifier(nn.Module):
    """Minimal classifier used only for infrastructure tests."""

    def __init__(self, vocab_size: int, num_labels: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        if attention_mask is None:
            pooled = embeddings.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def main() -> None:
    """Run the tiny infrastructure smoke test."""
    set_seed(42)
    train_df, leaderboard_df = load_required_clean_data()
    labels, label2id, id2label = build_label_mappings(train_df["y_category"])
    tokenizer = load_backbone_tokenizer()
    config = ICDDatasetConfig(max_length=DEFAULT_MAX_LENGTH)

    train_sample = train_df.sample(16, random_state=42)
    val_sample = train_df.drop(train_sample.index).sample(8, random_state=43)
    train_sample = train_sample.reset_index(drop=True)
    val_sample = val_sample.reset_index(drop=True)
    leaderboard_sample = leaderboard_df.head(8).reset_index(drop=True)

    train_dataset = ICDLiteralDataset(
        train_sample,
        tokenizer,
        mode="train",
        label2id=label2id,
        config=config,
    )
    val_dataset = ICDLiteralDataset(
        val_sample,
        tokenizer,
        mode="validation",
        label2id=label2id,
        config=config,
    )
    leaderboard_dataset = ICDLiteralDataset(
        leaderboard_sample,
        tokenizer,
        mode="leaderboard",
        config=config,
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=4, shuffle=False)

    device = get_device()
    model = TinyLiteralClassifier(
        vocab_size=len(tokenizer),
        num_labels=len(labels),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    eval_metrics = evaluate_torch_model(model, val_loader, device, id2label=id2label)

    checkpoint_path = CHECKPOINTS_DIR / "deep_learning_infra_smoke.pt"
    save_torch_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        epoch=1,
        config={
            "backbone_tokenizer": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
            "max_length": DEFAULT_MAX_LENGTH,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
        },
        metrics={**train_metrics, **eval_metrics},
    )
    load_torch_checkpoint(checkpoint_path, model, optimizer=optimizer, map_location=device)

    # Verify leaderboard batch structure and exact submission contract.
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in leaderboard_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions.extend(
                id2label[int(idx)] for idx in outputs.logits.argmax(dim=1).cpu().tolist()
            )
    submission = build_submission(leaderboard_sample, predictions)
    assert submission.columns.tolist() == ["id", "y_category"]
    assert "labels" not in leaderboard_dataset[0]

    summary = {
        "status": "ok",
        "device": str(device),
        "tokenizer_vocab_size": len(tokenizer),
        "max_length": DEFAULT_MAX_LENGTH,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "leaderboard_examples": len(leaderboard_dataset),
        "train_loss": train_metrics["train_loss"],
        "eval_accuracy": eval_metrics["accuracy"],
        "checkpoint": str(checkpoint_path),
        "submission_columns": submission.columns.tolist(),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(summary, LOGS_DIR / "deep_learning_infra_smoke.json")
    print(summary)


if __name__ == "__main__":
    main()

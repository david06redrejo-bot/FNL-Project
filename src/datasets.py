"""PyTorch datasets for ICD category prediction from clinical literals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer


BACKBONE_CHECKPOINT = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
DEFAULT_MAX_LENGTH = 32


def build_label_mappings(labels: list[str] | pd.Series) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build deterministic category mappings from observed labels."""
    sorted_labels = sorted(pd.Series(labels).astype(str).str.upper().unique().tolist())
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return sorted_labels, label2id, id2label


def add_label_ids(
    df: pd.DataFrame,
    label2id: dict[str, int] | None = None,
    label_col: str = "y_category",
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """Return a copy with deterministic `label_id` values."""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column `{label_col}`.")
    output = df.copy()
    output[label_col] = output[label_col].astype(str).str.upper()
    if label2id is None:
        _, label2id, id2label = build_label_mappings(output[label_col])
    else:
        id2label = {idx: label for label, idx in label2id.items()}
    unknown = sorted(set(output[label_col]) - set(label2id))
    if unknown:
        raise ValueError(f"Labels not present in label2id mapping: {unknown}")
    output["label_id"] = output[label_col].map(label2id).astype(int)
    return output, label2id, id2label


def load_backbone_tokenizer(
    checkpoint: str = BACKBONE_CHECKPOINT,
    local_files_only: bool = False,
):
    """Load the tokenizer used by all RoBERTa experiments."""
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=local_files_only)


@dataclass
class ICDDatasetConfig:
    """Configuration for `ICDLiteralDataset`."""

    text_col: str = "Literal_required_clean"
    label_col: str = "y_category"
    id_col: str = "id"
    max_length: int = DEFAULT_MAX_LENGTH


class ICDLiteralDataset(Dataset):
    """Dataset for train/validation and leaderboard ICD literal inference."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        mode: str,
        label2id: dict[str, int] | None = None,
        config: ICDDatasetConfig | None = None,
    ) -> None:
        if mode not in {"train", "validation", "leaderboard"}:
            raise ValueError("mode must be one of: train, validation, leaderboard")
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.mode = mode
        self.config = config or ICDDatasetConfig()
        if self.config.text_col not in self.df.columns:
            raise ValueError(f"Missing text column `{self.config.text_col}`.")

        self.label2id = label2id
        self.id2label: dict[int, str] | None = None
        if self.mode != "leaderboard":
            self.df, self.label2id, self.id2label = add_label_ids(
                self.df,
                label2id=label2id,
                label_col=self.config.label_col,
            )
        elif self.config.id_col not in self.df.columns:
            raise ValueError(f"Leaderboard mode requires `{self.config.id_col}`.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        encoded = self.tokenizer(
            str(row[self.config.text_col]),
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        }
        if self.mode == "leaderboard":
            item["id"] = row[self.config.id_col]
        else:
            item["labels"] = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return item


def stratified_literal_split(df, test_size: float = 0.2, random_state: int = 42):
    """Return stratified train/validation splits for category prediction."""
    stratify_col = "label_id" if "label_id" in df.columns else "y_category"
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col],
    )

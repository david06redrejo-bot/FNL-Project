"""Tests for RoBERTa-style deep learning infrastructure."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets import (
    DEFAULT_MAX_LENGTH,
    ICDDatasetConfig,
    ICDLiteralDataset,
    build_label_mappings,
    load_backbone_tokenizer,
)
from src.inference import build_submission


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return load_backbone_tokenizer(local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Backbone tokenizer is not available locally: {exc}")


@pytest.fixture()
def labeled_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "Code": ["A00", "B00", "J98"],
            "Literal_required_clean": ["Cólera", "Herpes simple", "Hiperreactividad bronquial"],
            "y_category": ["A", "B", "J"],
        }
    )


@pytest.fixture()
def leaderboard_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "id": [1, 2],
            "Literal_required_clean": ["Cólera", "Broncoespástica"],
        }
    )


def test_label_mapping_is_sorted(labeled_df):
    labels, label2id, id2label = build_label_mappings(labeled_df["y_category"])
    assert labels == ["A", "B", "J"]
    assert label2id == {"A": 0, "B": 1, "J": 2}
    assert id2label == {0: "A", 1: "B", 2: "J"}


def test_dataset_shapes(tokenizer, labeled_df):
    _, label2id, _ = build_label_mappings(labeled_df["y_category"])
    dataset = ICDLiteralDataset(
        labeled_df,
        tokenizer,
        mode="train",
        label2id=label2id,
        config=ICDDatasetConfig(max_length=DEFAULT_MAX_LENGTH),
    )
    item = dataset[0]
    assert item["input_ids"].shape == torch.Size([DEFAULT_MAX_LENGTH])
    assert item["attention_mask"].shape == torch.Size([DEFAULT_MAX_LENGTH])
    assert item["labels"].shape == torch.Size([])


def test_batch_structure(tokenizer, labeled_df):
    _, label2id, _ = build_label_mappings(labeled_df["y_category"])
    dataset = ICDLiteralDataset(labeled_df, tokenizer, mode="validation", label2id=label2id)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["input_ids"].shape == torch.Size([2, DEFAULT_MAX_LENGTH])
    assert batch["attention_mask"].shape == torch.Size([2, DEFAULT_MAX_LENGTH])
    assert batch["labels"].shape == torch.Size([2])


def test_leaderboard_has_no_labels(tokenizer, leaderboard_df):
    dataset = ICDLiteralDataset(leaderboard_df, tokenizer, mode="leaderboard")
    item = dataset[0]
    assert "labels" not in item
    assert set(item) == {"id", "input_ids", "attention_mask"}


def test_submission_format(leaderboard_df):
    submission = build_submission(leaderboard_df, ["A", "J"])
    assert submission.columns.tolist() == ["id", "y_category"]
    assert submission.shape == (2, 2)

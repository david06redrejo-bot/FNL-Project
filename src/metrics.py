"""Metric helpers for single-label ICD category classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute strict accuracy plus supporting F1/precision/recall metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def compute_full_metrics(
    y_true,
    y_pred,
    labels: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the full metric contract for a model-version run."""
    metrics: dict[str, Any] = classification_metrics(y_true, y_pred)
    metrics["labels"] = labels
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels
    ).astype(int).tolist()
    metrics["per_class_metrics"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    if y_proba is not None:
        y_true_array = np.asarray(y_true)
        metrics["log_loss"] = float(log_loss(y_true_array, y_proba, labels=labels))
        for k in [2, 3, 5]:
            if len(labels) >= k:
                metrics[f"top_{k}_accuracy"] = float(
                    top_k_accuracy_score(y_true_array, y_proba, k=k, labels=labels)
                )
    return metrics


def per_class_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn's per-class metric dictionary into a dataframe."""
    labels = metrics.get("labels", [])
    rows = []
    per_class = metrics.get("per_class_metrics", {})
    for label in labels:
        values = per_class.get(label, {})
        rows.append(
            {
                "y_category": label,
                "precision": values.get("precision", 0.0),
                "recall": values.get("recall", 0.0),
                "f1_score": values.get("f1-score", 0.0),
                "support": values.get("support", 0.0),
            }
        )
    return pd.DataFrame(rows)


def top_k_predictions_from_proba(
    proba: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> list[list[tuple[str, float]]]:
    """Return top-k `(label, probability)` pairs for each row."""
    top_k = min(k, len(labels))
    label_array = np.asarray(labels)
    order = np.argsort(proba, axis=1)[:, ::-1][:, :top_k]
    output = []
    for row_idx, label_indices in enumerate(order):
        output.append(
            [
                (str(label_array[label_idx]), float(proba[row_idx, label_idx]))
                for label_idx in label_indices
            ]
        )
    return output


def metrics_from_logits(
    labels,
    logits,
    id2label: dict[int, str],
) -> dict[str, Any]:
    """Compute the full metric contract from integer labels and model logits."""
    logits_array = np.asarray(logits)
    label_ids = np.asarray(labels)
    probabilities = softmax(logits_array)
    pred_ids = probabilities.argmax(axis=1)
    label_names = [id2label[idx] for idx in sorted(id2label)]
    y_true = [id2label[int(idx)] for idx in label_ids]
    y_pred = [id2label[int(idx)] for idx in pred_ids]
    return compute_full_metrics(
        y_true,
        y_pred,
        labels=label_names,
        y_proba=probabilities,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for metric computation."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

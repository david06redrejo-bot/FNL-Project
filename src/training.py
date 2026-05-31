"""Shared model-version interface and lightweight sklearn implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data_loading import add_label_columns
from .inference import build_detailed_predictions, build_submission
from .metrics import compute_full_metrics, per_class_metrics_table
from .paths import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    SUBMISSIONS_DIR,
    ensure_project_dirs,
)
from .preprocessing import normalize_literal
from .reporting import write_run_summary
from .utils import save_json, set_seed


@dataclass
class ModelRunConfig:
    """Configuration for a deliverable model-version run."""

    version_name: str
    model_family: str
    seed: int = 42
    validation_size: float = 0.2
    text_col: str = "Literal_required_clean"
    target_col: str = "y_category"
    label_id_col: str = "label_id"
    debug_sample: int | None = None
    save_model: bool = True
    top_k: int = 5


def make_classical_pipeline(model_family: str, seed: int = 42) -> Pipeline:
    """Create a lightweight sklearn pipeline for interface tests and baselines."""
    if model_family == "majority":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(preprocessor=normalize_literal)),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        )
    if model_family == "logreg":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_literal,
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=1,
                        max_features=200_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=None,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_family == "debug_logreg":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_literal,
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        min_df=1,
                        max_features=5000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_literal,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=2.0, max_iter=10_000, random_state=seed)),
        ]
    )


def load_required_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed required-clean train and leaderboard files."""
    train_path = PROCESSED_DATA_DIR / "train_required_clean.csv"
    leaderboard_path = PROCESSED_DATA_DIR / "leaderboard_required_clean.csv"
    if not train_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError(
            "Required-clean files are missing. Run `python scripts/run_preprocessing.py` first."
        )
    train = pd.read_csv(train_path)
    leaderboard = pd.read_csv(leaderboard_path)
    if "label_id" not in train.columns:
        train, _ = add_label_columns(train)
    return train, leaderboard


def apply_debug_sample(
    train: pd.DataFrame,
    leaderboard: pd.DataFrame,
    config: ModelRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a tiny stratified-ish debug subset for smoke tests."""
    if not config.debug_sample:
        return train, leaderboard

    per_class = max(2, config.debug_sample // train[config.target_col].nunique())
    samples = []
    for _, group in train.groupby(config.target_col):
        samples.append(group.sample(min(len(group), per_class), random_state=config.seed))
    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) > config.debug_sample:
        sampled = sampled.sample(config.debug_sample, random_state=config.seed)
    leaderboard_sample = leaderboard.head(min(len(leaderboard), max(20, config.debug_sample // 2))).copy()
    return sampled.reset_index(drop=True), leaderboard_sample.reset_index(drop=True)


def split_train_validation(
    train: pd.DataFrame,
    config: ModelRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the mandated 80/20 split stratified by `label_id`."""
    stratify = train[config.label_id_col]
    try:
        train_df, val_df = train_test_split(
            train,
            test_size=config.validation_size,
            random_state=config.seed,
            stratify=stratify,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            train,
            test_size=config.validation_size,
            random_state=config.seed,
            stratify=None,
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def predict_proba_if_available(model: Any, texts: pd.Series) -> np.ndarray | None:
    """Return probabilities when the estimator supports them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    return None


def run_model_contract(
    config: ModelRunConfig,
    model: Any | None = None,
    train_model: bool = True,
    evaluate: bool = True,
    predict_leaderboard: bool = True,
    make_submission: bool = True,
    load_model_path: Path | None = None,
) -> dict[str, Any]:
    """Run the strict model-version contract."""
    ensure_project_dirs()
    set_seed(config.seed)

    train, leaderboard = load_required_clean_data()
    train, leaderboard = apply_debug_sample(train, leaderboard, config)
    labels = sorted(train[config.target_col].astype(str).unique().tolist())
    train_df, val_df = split_train_validation(train, config)

    if load_model_path:
        model = joblib.load(load_model_path)
    elif model is None:
        model = make_classical_pipeline(config.model_family, seed=config.seed)

    artifact_paths: dict[str, str] = {}
    config_dict = asdict(config)
    config_path = LOGS_DIR / f"{config.version_name}_config.json"
    save_json(config_dict, config_path)
    artifact_paths["config"] = str(config_path)

    if train_model:
        model.fit(train_df[config.text_col], train_df[config.target_col])

    metrics: dict[str, Any] = {}
    if evaluate:
        val_pred = model.predict(val_df[config.text_col])
        val_proba = predict_proba_if_available(model, val_df[config.text_col])
        metrics = compute_full_metrics(
            val_df[config.target_col],
            val_pred,
            labels=labels,
            y_proba=val_proba,
        )
        metrics_path = METRICS_DIR / f"{config.version_name}_metrics.json"
        save_json(metrics, metrics_path)
        artifact_paths["metrics"] = str(metrics_path)

        val_detailed = build_detailed_predictions(
            val_df,
            val_pred,
            y_true=val_df[config.target_col].values,
            probabilities=val_proba,
            labels=labels,
            literal_col=config.text_col,
        )
        val_path = PREDICTIONS_DIR / f"{config.version_name}_val_predictions.csv"
        val_detailed.to_csv(val_path, index=False)
        artifact_paths["validation_predictions"] = str(val_path)

        per_class_path = METRICS_DIR / f"{config.version_name}_per_class_metrics.csv"
        per_class_metrics_table(metrics).to_csv(per_class_path, index=False)
        artifact_paths["per_class_metrics"] = str(per_class_path)

    if config.save_model:
        checkpoint_path = CHECKPOINTS_DIR / f"{config.version_name}.joblib"
        joblib.dump(model, checkpoint_path)
        artifact_paths["model_artifact"] = str(checkpoint_path)

    if predict_leaderboard or make_submission:
        # Fit on all available labeled data before leaderboard prediction. This
        # does not change validation metrics, which were computed above.
        if train_model:
            model.fit(train[config.text_col], train[config.target_col])
        leaderboard_pred = model.predict(leaderboard[config.text_col])
        leaderboard_proba = predict_proba_if_available(model, leaderboard[config.text_col])
        detailed = build_detailed_predictions(
            leaderboard,
            leaderboard_pred,
            probabilities=leaderboard_proba,
            labels=labels,
            id_col="id",
            literal_col=config.text_col,
        )
        detailed_path = PREDICTIONS_DIR / f"{config.version_name}_leaderboard_detailed.csv"
        detailed.to_csv(detailed_path, index=False)
        artifact_paths["leaderboard_detailed"] = str(detailed_path)

        if make_submission:
            submission = build_submission(leaderboard, leaderboard_pred)
            submission_path = SUBMISSIONS_DIR / f"{config.version_name}_submission.csv"
            submission.to_csv(submission_path, index=False)
            artifact_paths["submission"] = str(submission_path)

    summary_path = write_run_summary(
        config.version_name,
        config_dict,
        metrics,
        artifact_paths,
    )
    artifact_paths["run_summary"] = str(summary_path)
    return {
        "version_name": config.version_name,
        "metrics": metrics,
        "artifacts": artifact_paths,
    }


def run_sklearn_model_version(
    model_id: str,
    model_family: str,
    train: bool = True,
    evaluate: bool = True,
    predict: bool = True,
    make_submission: bool = True,
    dry_run: bool = False,
    debug_sample: int | None = None,
    seed: int = 42,
    load_model_path: Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper used by model entry-point scripts."""
    version_name = f"{model_id}_debug" if dry_run else model_id
    if dry_run:
        debug_sample = debug_sample or 120
        model_family = "debug_logreg" if model_family != "majority" else "majority"
    config = ModelRunConfig(
        version_name=version_name,
        model_family=model_family,
        seed=seed,
        debug_sample=debug_sample,
    )
    return run_model_contract(
        config,
        train_model=train,
        evaluate=evaluate,
        predict_leaderboard=predict,
        make_submission=make_submission,
        load_model_path=load_model_path,
    )


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: Any,
    scaler: Any | None = None,
    use_mixed_precision: bool = False,
) -> dict[str, float]:
    """Train a PyTorch model for one epoch."""
    import torch

    model.train()
    total_loss = 0.0
    total_examples = 0
    autocast_enabled = bool(use_mixed_precision and getattr(device, "type", "") == "cuda")

    for batch in dataloader:
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size

    return {"train_loss": total_loss / max(total_examples, 1)}


def evaluate_torch_model(
    model: Any,
    dataloader: Any,
    device: Any,
    id2label: dict[int, str],
) -> dict[str, Any]:
    """Evaluate a PyTorch classifier and return project metrics."""
    import torch

    from .metrics import metrics_from_logits

    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            batch_size = int(batch["labels"].shape[0])
            total_examples += batch_size
            total_loss += float(outputs.loss.detach().cpu()) * batch_size
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = metrics_from_logits(labels, logits, id2label=id2label)
    metrics["eval_loss"] = total_loss / max(total_examples, 1)
    return metrics


def predict_torch_model(
    model: Any,
    dataloader: Any,
    device: Any,
    id2label: dict[int, str],
) -> pd.DataFrame:
    """Predict leaderboard rows with a PyTorch classifier."""
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["id"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = outputs.logits.argmax(dim=1).detach().cpu().numpy()
            for row_id, pred_id in zip(ids, pred_ids, strict=False):
                rows.append({"id": row_id, "y_category": id2label[int(pred_id)]})
    return pd.DataFrame(rows)[["id", "y_category"]]


def save_torch_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any | None = None,
    epoch: int | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a lightweight PyTorch checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_torch_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any | None = None,
    map_location: str | Any = "cpu",
) -> dict[str, Any]:
    """Load a PyTorch checkpoint into a model and optionally an optimizer."""
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

"""Shared command-line interface for model-version files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import run_sklearn_model_version


def run_version(model_id: str, model_family: str, description: str) -> None:
    """Parse CLI flags and execute a model version."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on validation split.")
    parser.add_argument("--predict", action="store_true", help="Predict leaderboard literals.")
    parser.add_argument(
        "--make-submission", action="store_true", help="Write a submission CSV."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full contract on a tiny debug sample.",
    )
    parser.add_argument("--debug-sample", type=int, default=None, help="Use a tiny labeled sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--load-model-path",
        type=Path,
        default=None,
        help="Load an existing model artifact/checkpoint instead of training.",
    )
    args = parser.parse_args()

    run_all = not any([args.train, args.evaluate, args.predict, args.make_submission])
    train_model = args.train or (run_all and args.load_model_path is None)
    result = run_sklearn_model_version(
        model_id=model_id,
        model_family=model_family,
        train=train_model,
        evaluate=args.evaluate or run_all,
        predict=args.predict or run_all,
        make_submission=args.make_submission or run_all,
        dry_run=args.dry_run,
        debug_sample=args.debug_sample,
        seed=args.seed,
        load_model_path=args.load_model_path,
    )
    metrics = result.get("metrics", {})
    print(
        {
            "version_name": result.get("version_name"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "artifacts": result.get("artifacts", {}),
        }
    )

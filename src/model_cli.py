"""Shared command-line interface for model-version files."""

from __future__ import annotations

import argparse

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
        help="Validate the script wiring without reading data or training.",
    )
    args = parser.parse_args()

    run_all = not any([args.train, args.evaluate, args.predict, args.make_submission])
    result = run_sklearn_model_version(
        model_id=model_id,
        model_family=model_family,
        train=args.train or run_all,
        evaluate=args.evaluate or run_all,
        predict=args.predict or run_all,
        make_submission=args.make_submission or run_all,
        dry_run=args.dry_run,
    )
    print(result)


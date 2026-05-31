"""Lightweight local demo for ICD category prediction.

This optional demo is intentionally small. The final v10 submission is an
ensemble over completed model predictions and component models, not a single
interactive artifact. For typed literals, the demo loads the best available
probabilistic local artifact; by default this is the character TF-IDF logistic
regression component, which can return top-k probabilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_required


DISCLAIMER = (
    "Academic demo only. This tool predicts a broad ICD-10 category prefix from "
    "a short literal. It is not a clinical device, not a coding assistant for "
    "real hospital use, and must not be used for patient care, reimbursement, "
    "or administrative decisions."
)


DEFAULT_CANDIDATES = [
    (
        PROJECT_ROOT / "outputs" / "checkpoints" / "v10_vote_diverse_no_retrieval.joblib",
        "final v10 ensemble artifact",
    ),
    (
        PROJECT_ROOT / "outputs" / "checkpoints" / "v01_tfidf_char_logreg.joblib",
        "v01 character TF-IDF logistic regression component",
    ),
]


def find_model_path(explicit_path: str | None = None) -> tuple[Path, str]:
    """Choose the best available lightweight demo model artifact."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return path, "custom model artifact"

    for path, description in DEFAULT_CANDIDATES:
        if path.exists():
            return path, description

    candidates = "\n".join(f"- {path}" for path, _ in DEFAULT_CANDIDATES)
    raise FileNotFoundError(
        "No compatible local demo model artifact was found. Expected one of:\n"
        f"{candidates}\n\n"
        "Run `python models/v01_tfidf_char_logreg.py` to create the lightweight "
        "probabilistic fallback used by this demo."
    )


def load_model(path: Path):
    """Load a joblib model and verify that it can produce probabilities."""
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from {path} has no predict method.")
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Loaded model from {path} does not provide predict_proba. "
            "The demo needs probabilities to show top-3 categories."
        )
    return model


def predict_literal(model, literal: str, top_k: int = 3) -> tuple[str, list[tuple[str, float]]]:
    """Predict one literal and return the top-k category probabilities."""
    cleaned = clean_required(literal)
    if not cleaned:
        raise ValueError("Please enter a non-empty clinical literal.")

    probabilities = model.predict_proba([cleaned])[0]
    labels = np.asarray(model.classes_, dtype=str)
    order = np.argsort(probabilities)[::-1][:top_k]
    top = [(str(labels[i]), float(probabilities[i])) for i in order]
    return top[0][0], top


def print_prediction(literal: str, predicted: str, top: list[tuple[str, float]]) -> None:
    """Pretty-print one prediction."""
    print("\nInput literal:")
    print(f"  {literal}")
    print(f"\nPredicted y_category: {predicted}")
    print("\nTop-3 categories:")
    for rank, (label, probability) in enumerate(top, start=1):
        print(f"  {rank}. {label}: {probability:.4f}")


def interactive_loop(model) -> None:
    """Run a tiny console interface."""
    print("\nType a clinical literal and press Enter. Use Ctrl-D or an empty line to exit.\n")
    while True:
        try:
            literal = input("literal> ").strip()
        except EOFError:
            print()
            break
        if not literal:
            break
        try:
            predicted, top = predict_literal(model, literal)
            print_prediction(literal, predicted, top)
        except Exception as exc:  # noqa: BLE001 - user-facing demo guard
            print(f"Could not predict this literal: {exc}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional local ICD category demo.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional joblib artifact with predict and predict_proba.",
    )
    parser.add_argument(
        "--literal",
        default=None,
        help="Predict one literal and exit. If omitted, start an interactive prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(DISCLAIMER)

    try:
        model_path, description = find_model_path(args.model_path)
        model = load_model(model_path)
    except Exception as exc:  # noqa: BLE001 - user-facing demo guard
        print(f"\nDemo could not start: {exc}")
        return 1

    print(f"\nLoaded model: {description}")
    print(f"Artifact: {model_path.relative_to(PROJECT_ROOT) if model_path.is_relative_to(PROJECT_ROOT) else model_path}")

    if description != "final v10 ensemble artifact":
        print(
            "\nNote: the final v10 Kaggle system is an ensemble submission recipe. "
            "This lightweight demo uses the available probabilistic component for "
            "interactive top-3 predictions."
        )

    if args.literal is not None:
        predicted, top = predict_literal(model, args.literal)
        print_prediction(args.literal, predicted, top)
        return 0

    interactive_loop(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""v00: majority-category baseline.

This deliberately simple baseline estimates the lower bound for strict
accuracy by always predicting the most frequent category in the training split.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v00_majority_baseline",
        model_family="majority",
        description="Train/evaluate/predict the majority-category baseline.",
    )

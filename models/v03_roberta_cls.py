"""v03: RoBERTa CLS-pooling baseline entry point.

This file provides the traceable model-version interface. The transformer
implementation will replace the current lightweight sklearn-backed runner once
the raw data and compute environment are confirmed.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v03_roberta_cls",
        model_family="svm",
        description="Run the RoBERTa CLS baseline interface.",
    )

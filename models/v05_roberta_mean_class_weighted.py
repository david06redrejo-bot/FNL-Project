"""v05: class-weighted RoBERTa mean-pooling baseline entry point."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v05_roberta_mean_class_weighted",
        model_family="svm",
        description="Run the class-weighted RoBERTa mean-pooling interface.",
    )

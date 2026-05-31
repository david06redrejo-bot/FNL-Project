"""v01: character TF-IDF + logistic regression baseline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v01_tfidf_char_logreg",
        model_family="logreg",
        description="Train/evaluate/predict a char n-gram TF-IDF logistic regression.",
    )

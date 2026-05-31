"""v02: TF-IDF + linear SVM baseline.

The current scaffold uses the shared short-literal SVM configuration. The final
version should compare word, character, and combined feature spaces explicitly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v02_tfidf_word_svm",
        model_family="svm",
        description="Train/evaluate/predict a TF-IDF linear SVM baseline.",
    )

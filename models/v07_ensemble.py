"""v07: ensemble model-version entry point.

The final version should combine calibrated predictions from the strongest
classical and transformer models. The scaffold currently uses the shared runner
so it can already produce traceable metrics and submission files.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_cli import run_version


if __name__ == "__main__":
    run_version(
        model_id="v07_ensemble",
        model_family="svm",
        description="Run the final ensemble interface.",
    )

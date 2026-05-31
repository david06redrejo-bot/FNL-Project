"""Central project paths.

All scripts should import paths from this module instead of hard-coding
relative locations. This keeps notebooks narrative and makes command-line
experiments reproducible from the project root.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPERIMENT_LOG_PATH = PROJECT_ROOT / "EXPERIMENT_LOG.md"
REPORT_NOTES_PATH = PROJECT_ROOT / "REPORT_NOTES.md"


def ensure_project_dirs() -> None:
    """Create the standard project directories if they are missing."""
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EDA_OUTPUT_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        SUBMISSIONS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


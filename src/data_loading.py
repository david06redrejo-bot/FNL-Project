"""Data loading helpers for the competition CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import RAW_DATA_DIR


RAW_FILENAMES = {
    "codification": "codification_data.csv",
    "leaderboard": "leaderboard_data.csv",
    "icd_catalog": "icd_d_p_pairs.csv",
}


def raw_file_path(name: str, data_dir: Path = RAW_DATA_DIR) -> Path:
    """Return the expected path for a raw competition file."""
    if name not in RAW_FILENAMES:
        raise KeyError(f"Unknown raw dataset name: {name}")
    return data_dir / RAW_FILENAMES[name]


def load_csv(name: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load one competition CSV from `data/raw`."""
    path = raw_file_path(name, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Place the Kaggle competition data in data/raw/."
        )
    return pd.read_csv(path)


def load_competition_data(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all expected raw competition files."""
    return {name: load_csv(name, data_dir=data_dir) for name in RAW_FILENAMES}


"""Data loading, inspection, and schema validation for competition CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPECTED_CATEGORIES
from .paths import DATA_DIR, RAW_DATA_DIR


RAW_FILENAMES = {
    "codification": "codification_data.csv",
    "leaderboard": "leaderboard_data.csv",
    "icd_catalog": "icd_d_p_pairs.csv",
}

EXPECTED_SCHEMAS = {
    "codification": {"required": ["Code", "Literal"], "optional": []},
    # Kaggle leaderboard/test files may not include y_category. If present, we
    # validate it; if absent, we warn rather than hiding the mismatch.
    "leaderboard": {"required": ["id", "Literal"], "optional": ["y_category"]},
    "icd_catalog": {"required": ["Code"], "optional": ["D_P", "Description"]},
}


def raw_file_path(name: str, data_dir: Path = RAW_DATA_DIR) -> Path:
    """Return the expected path for a raw competition file."""
    if name not in RAW_FILENAMES:
        raise KeyError(f"Unknown raw dataset name: {name}")
    return data_dir / RAW_FILENAMES[name]


def discover_csv_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Find all CSV files below `data/`, excluding hidden checkpoint files."""
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.rglob("*.csv")
        if not any(part.startswith(".") for part in path.parts)
    )


def infer_dataset_role(path: Path, columns: list[str] | None = None) -> str:
    """Infer the dataset role from filename and, if needed, columns."""
    filename = path.name.lower()
    if filename == RAW_FILENAMES["codification"]:
        return "codification"
    if filename == RAW_FILENAMES["leaderboard"]:
        return "leaderboard"
    if filename == RAW_FILENAMES["icd_catalog"]:
        return "icd_catalog"

    colset = set(columns or [])
    if {"Code", "Literal"}.issubset(colset):
        return "codification_like"
    if {"id", "Literal"}.issubset(colset):
        return "leaderboard_like"
    if "Code" in colset and ("Description" in colset or "D_P" in colset):
        return "icd_catalog_like"
    return "unknown"


def resolve_csv_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Resolve an expected CSV by searching `data/raw` and then all `data/`."""
    if name not in RAW_FILENAMES:
        raise KeyError(f"Unknown raw dataset name: {name}")

    expected_name = RAW_FILENAMES[name]
    preferred = RAW_DATA_DIR / expected_name
    if preferred.exists():
        return preferred

    matches = [path for path in discover_csv_files(data_dir) if path.name == expected_name]
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing {expected_name}. Expected it under data/raw/ or elsewhere in data/."
    )


def load_csv(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one expected competition CSV from `data/`."""
    path = resolve_csv_path(name, data_dir=data_dir)
    return pd.read_csv(path)


def load_csv_path(path: Path) -> pd.DataFrame:
    """Load a CSV path with a consistent error message."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    return pd.read_csv(path)


def load_competition_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all expected raw competition files."""
    return {name: load_csv(name, data_dir=data_dir) for name in RAW_FILENAMES}


def derive_y_category(df: pd.DataFrame, code_col: str = "Code") -> pd.Series:
    """Derive `y_category` from the first character of `Code`."""
    if code_col not in df.columns:
        raise ValueError(f"Cannot derive y_category: missing column `{code_col}`.")
    code = df[code_col].astype(str)
    invalid_mask = code.str.len().eq(0) | code.str.lower().eq("nan")
    if invalid_mask.any():
        raise ValueError(
            f"Cannot derive y_category: {int(invalid_mask.sum())} empty/null codes."
        )
    return code.str[0].str.upper()


def build_label_mapping(labels: pd.Series) -> dict[str, Any]:
    """Build sorted label mapping objects and integer ids."""
    sorted_labels = sorted(labels.dropna().astype(str).str.upper().unique().tolist())
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return {
        "sorted_labels": sorted_labels,
        "label2id": label2id,
        "id2label": id2label,
        "n_classes": len(sorted_labels),
    }


def add_label_columns(df: pd.DataFrame, code_col: str = "Code") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a copy with `y_category` and `label_id` plus mapping metadata."""
    result = df.copy()
    result["y_category"] = derive_y_category(result, code_col=code_col)
    mapping = build_label_mapping(result["y_category"])
    result["label_id"] = result["y_category"].map(mapping["label2id"]).astype("int64")
    return result, mapping


def validate_schema(df: pd.DataFrame, role: str) -> dict[str, Any]:
    """Validate a dataframe schema and return errors/warnings."""
    schema_role = role.replace("_like", "")
    expected = EXPECTED_SCHEMAS.get(schema_role)
    columns = df.columns.tolist()
    if expected is None:
        return {
            "role": role,
            "valid": False,
            "columns": columns,
            "errors": [f"Unknown dataset role `{role}`."],
            "warnings": [],
        }

    missing_required = [col for col in expected["required"] if col not in columns]
    missing_optional = [col for col in expected["optional"] if col not in columns]
    warnings = []
    if schema_role == "leaderboard" and "y_category" not in columns:
        warnings.append(
            "`y_category` is absent from leaderboard data; this is normal for unlabeled "
            "Kaggle test files but differs from the stated expected columns."
        )
    elif missing_optional:
        warnings.append(f"Optional columns missing: {missing_optional}")

    return {
        "role": role,
        "valid": len(missing_required) == 0,
        "columns": columns,
        "required_columns": expected["required"],
        "optional_columns": expected["optional"],
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "errors": [f"Missing required columns: {missing_required}"] if missing_required else [],
        "warnings": warnings,
    }


def inspect_csv_file(path: Path, max_examples: int = 5) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load and summarize one CSV file."""
    df = load_csv_path(path)
    role = infer_dataset_role(path, df.columns.tolist())
    validation = validate_schema(df, role)

    summary = {
        "filename": str(path),
        "basename": path.name,
        "role": role,
        "shape_rows": int(df.shape[0]),
        "shape_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "null_counts": {col: int(value) for col, value in df.isna().sum().items()},
        "duplicate_row_count": int(df.duplicated().sum()),
        "example_literals": (
            df["Literal"].dropna().astype(str).head(max_examples).tolist()
            if "Literal" in df.columns
            else []
        ),
        "example_codes": (
            df["Code"].dropna().astype(str).head(max_examples).tolist()
            if "Code" in df.columns
            else []
        ),
        "first_rows": df.head(max_examples).astype(object).where(pd.notna(df), None).to_dict("records"),
        "validation": validation,
    }

    if "Code" in df.columns:
        labels = derive_y_category(df)
        mapping = build_label_mapping(labels)
        summary["label_mapping"] = mapping
        summary["expected_36_classes"] = mapping["n_classes"] == len(EXPECTED_CATEGORIES)
        summary["missing_expected_categories"] = sorted(
            set(EXPECTED_CATEGORIES) - set(mapping["sorted_labels"])
        )
        summary["unexpected_categories"] = sorted(
            set(mapping["sorted_labels"]) - set(EXPECTED_CATEGORIES)
        )

    return summary, df


def write_schema_validation(payload: dict[str, Any], output_path: Path) -> None:
    """Write schema validation JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

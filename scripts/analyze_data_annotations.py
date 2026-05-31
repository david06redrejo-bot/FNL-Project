"""Analyze the raw CSV files and validate annotation schemas.

This script performs the first project phase only: analyzing the data and the
annotations. It does not train any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPECTED_CATEGORIES
from src.data_loading import (
    add_label_columns,
    discover_csv_files,
    infer_dataset_role,
    inspect_csv_file,
    write_schema_validation,
)
from src.paths import DATA_DIR, EDA_OUTPUT_DIR, REPORTS_DIR, ensure_project_dirs
from src.utils import save_json


def compact_inventory_row(summary: dict) -> dict:
    """Flatten a file summary into one CSV-friendly row."""
    return {
        "filename": summary["filename"],
        "basename": summary["basename"],
        "role": summary["role"],
        "rows": summary["shape_rows"],
        "columns_count": summary["shape_columns"],
        "columns": " | ".join(summary["columns"]),
        "duplicate_row_count": summary["duplicate_row_count"],
        "validation_valid": summary["validation"]["valid"],
        "validation_errors": " | ".join(summary["validation"].get("errors", [])),
        "validation_warnings": " | ".join(summary["validation"].get("warnings", [])),
        "example_literals": " || ".join(summary["example_literals"]),
        "example_codes": " || ".join(summary["example_codes"]),
    }


def build_report_note(summaries: list[dict], validation_payload: dict) -> str:
    """Create the report-note text for phase 1."""
    if not summaries:
        return (
            "No real CSV files were found under `data/` during the first data and "
            "annotation analysis run. The repository currently contains only data "
            "folder placeholders, so row counts, schemas, and the 36-class label "
            "contract cannot yet be verified from the actual Kaggle files.\n\n"
            "Required next action: place the competition CSV files in `data/raw/` "
            "and rerun `python scripts/analyze_data_annotations.py`."
        )

    lines = [
        "We inspected all CSV files under `data/` and validated them against the "
        "competition annotation contract.",
        "",
        "| File | Role | Shape | Schema status |",
        "|---|---|---:|---|",
    ]
    for summary in summaries:
        status = "valid" if summary["validation"]["valid"] else "invalid"
        lines.append(
            f"| `{summary['basename']}` | `{summary['role']}` | "
            f"{summary['shape_rows']} x {summary['shape_columns']} | {status} |"
        )

    training = next(
        (item for item in summaries if item["role"].replace("_like", "") == "codification"),
        None,
    )
    if training and "label_mapping" in training:
        mapping = training["label_mapping"]
        lines.extend(
            [
                "",
                f"The training labels are derived as `y_category = Code.astype(str).str[0]`.",
                f"The observed label set has {mapping['n_classes']} classes: "
                f"{', '.join(mapping['sorted_labels'])}.",
            ]
        )
        if mapping["n_classes"] != len(EXPECTED_CATEGORIES):
            lines.append(
                "This differs from the expected 36 categories, so the mismatch must "
                "be considered before training."
            )

    warnings = validation_payload.get("warnings", [])
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def upsert_report_section(section_title: str, body: str) -> None:
    """Create or replace a report-note section with an exact title."""
    notes_path = PROJECT_ROOT / "REPORT_NOTES.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    if not notes_path.exists():
        notes_path.write_text("# Report Notes\n", encoding="utf-8")
    content = notes_path.read_text(encoding="utf-8")
    heading = f"## {section_title}"
    replacement = f"{heading}\n\n{body.strip()}\n"

    if heading not in content:
        notes_path.write_text(content.rstrip() + "\n\n" + replacement, encoding="utf-8")
        return

    start = content.index(heading)
    next_heading = content.find("\n## ", start + len(heading))
    if next_heading == -1:
        updated = content[:start].rstrip() + "\n\n" + replacement
    else:
        updated = content[:start].rstrip() + "\n\n" + replacement + content[next_heading:]
    notes_path.write_text(updated, encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    (REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(DATA_DIR)
    summaries = []
    validation_payload = {
        "phase": "1. Analyzing the Data and the Annotations",
        "data_dir": str(DATA_DIR),
        "csv_file_count": len(csv_files),
        "files": [],
        "warnings": [],
        "errors": [],
    }

    if not csv_files:
        warning = "No CSV files found under data/. Place competition files in data/raw/."
        validation_payload["warnings"].append(warning)
        inventory_df = pd.DataFrame(
            columns=[
                "filename",
                "basename",
                "role",
                "rows",
                "columns_count",
                "columns",
                "duplicate_row_count",
                "validation_valid",
                "validation_errors",
                "validation_warnings",
                "example_literals",
                "example_codes",
            ]
        )
    else:
        rows = []
        for path in csv_files:
            summary, df = inspect_csv_file(path)
            summaries.append(summary)
            validation_payload["files"].append(summary)
            rows.append(compact_inventory_row(summary))

            print(f"\n=== {path} ===")
            print(f"shape: {df.shape}")
            print(f"columns: {df.columns.tolist()}")
            print("first rows:")
            print(df.head().to_string(index=False))
            print("null counts:")
            print(df.isna().sum().to_string())
            print(f"duplicate rows: {int(df.duplicated().sum())}")
            if "Literal" in df.columns:
                print("example literals:")
                print(df["Literal"].dropna().astype(str).head(5).to_string(index=False))
            if "Code" in df.columns:
                print("example codes:")
                print(df["Code"].dropna().astype(str).head(5).to_string(index=False))

        inventory_df = pd.DataFrame(rows)

        expected_basenames = {"codification_data.csv", "leaderboard_data.csv"}
        found_basenames = {path.name for path in csv_files}
        missing = sorted(expected_basenames - found_basenames)
        if missing:
            validation_payload["warnings"].append(f"Missing expected files: {missing}")

        for summary in summaries:
            validation_payload["warnings"].extend(summary["validation"].get("warnings", []))
            errors = summary["validation"].get("errors", [])
            if summary["role"] == "unknown":
                validation_payload["warnings"].extend(
                    f"Non-competition CSV `{summary['basename']}` has unknown schema: {error}"
                    for error in errors
                )
            else:
                validation_payload["errors"].extend(errors)

        training = next(
            (
                item
                for item in summaries
                if infer_dataset_role(Path(item["basename"]), item["columns"]).replace("_like", "")
                == "codification"
            ),
            None,
        )
        if training and "label_mapping" in training:
            label_path = EDA_OUTPUT_DIR / "label_mapping.json"
            save_json(training["label_mapping"], label_path)
            if training["label_mapping"]["n_classes"] != len(EXPECTED_CATEGORIES):
                validation_payload["warnings"].append(
                    f"Expected 36 classes but found {training['label_mapping']['n_classes']}."
                )

            # Save an example processed training table with y_category/label_id.
            training_path = next(path for path in csv_files if path.name == training["basename"])
            training_df = pd.read_csv(training_path)
            labeled_df, _ = add_label_columns(training_df)
            labeled_df.head(50).to_csv(EDA_OUTPUT_DIR / "training_label_preview.csv", index=False)

    inventory_path = EDA_OUTPUT_DIR / "data_file_inventory.csv"
    schema_path = EDA_OUTPUT_DIR / "schema_validation.json"
    table_path = REPORTS_DIR / "tables" / "data_schema_summary.csv"

    inventory_df.to_csv(inventory_path, index=False)
    inventory_df.to_csv(table_path, index=False)
    write_schema_validation(validation_payload, schema_path)

    upsert_report_section(
        "1. Analyzing the Data and the Annotations",
        build_report_note(summaries, validation_payload),
    )

    print(f"\nWrote {inventory_path}")
    print(f"Wrote {schema_path}")
    print(f"Wrote {table_path}")
    if validation_payload["warnings"]:
        print("\nWarnings:")
        for warning in validation_payload["warnings"]:
            print(f"- {warning}")
    if validation_payload["errors"]:
        print("\nErrors:")
        for error in validation_payload["errors"]:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

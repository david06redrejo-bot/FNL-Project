# Reproducibility Guide

This document explains how to reproduce the Team 10 UAB-ASHO AI Codification
project from a fresh checkout.

## 1. Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Download the Kaggle competition files and place them in `data/raw/`:

```text
data/raw/codification_data.csv
data/raw/leaderboard_data.csv
data/raw/icd_d_p_pairs.csv
```

Raw data is intentionally ignored by Git. The repository keeps the directory
structure but not the competition files.

## 3. Smoke Checks

Run a dry-run model command to verify that imports, paths, and logging work:

```bash
python models/v00_majority_baseline.py --dry-run
```

Run syntax checks:

```bash
python -m compileall src models
```

## 3.1 Data and Annotation Validation

Before training any model, run the first project phase:

```bash
python scripts/analyze_data_annotations.py
```

This command inspects all CSV files under `data/`, validates schemas, derives
`y_category = Code.astype(str).str[0]` when `Code` is available, builds the
label mapping, and writes:

- `outputs/eda/data_file_inventory.csv`
- `outputs/eda/schema_validation.json`
- `reports/tables/data_schema_summary.csv`
- an entry in `REPORT_NOTES.md`

## 4. Reproduce Baselines

After placing data in `data/raw/`, run:

```bash
python models/v00_majority_baseline.py
python models/v01_tfidf_char_logreg.py
python models/v02_tfidf_word_svm.py
```

Each script writes:

- validation metrics to `outputs/metrics/`
- validation and leaderboard predictions to `outputs/predictions/`
- submission files to `submissions/`
- experiment notes to `EXPERIMENT_LOG.md`
- report observations to `REPORT_NOTES.md`

## 5. Notebooks

The notebooks are narrative companions to the scripts. They should explain the
reasoning and show selected outputs, but reusable logic should stay in `src/`.

Recommended order:

1. `notebooks/00_task_formulation_and_eda.ipynb`
2. `notebooks/01_data_preprocessing_and_annotation_design.ipynb`
3. `notebooks/02_reference_methods_from_survey.ipynb`
4. `notebooks/03_classical_baselines.ipynb`
5. `notebooks/04_roberta_backbone_baseline.ipynb`
6. `notebooks/05_advanced_model_experiments.ipynb`
7. `notebooks/06_hyperparameter_and_ablation_studies.ipynb`
8. `notebooks/07_evaluation_error_analysis_and_interpretability.ipynb`
9. `notebooks/08_submission_and_final_story.ipynb`

## 6. Final Report

The LaTeX source starts at:

```text
reports/final_report.tex
```

Compiled PDFs should be written to `reports/compiled/` and are treated as
generated artifacts unless the team decides to commit the final version.

## 7. Current Limitation

At takeover time, the raw data files were not present in this checkout. Model
metrics from older notebooks must therefore be rerun before they are reported as
final results.

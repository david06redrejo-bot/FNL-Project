# Experiment Log

This file tracks every model attempt, including failed and inconclusive attempts. Historical results from existing notebooks are marked as not yet revalidated until rerun from the current data.

## 2026-05-31 — Repository Takeover / Historical Experiments Inventory

**Goal:** Identify existing experiments before rebuilding the project.

**Data status:** The expected `data/` directory is not present in this checkout, so no model metrics were rerun during takeover.

**Historical experiments found:**

| ID | Source | Method | Reported result | Current status |
|---|---|---|---|---|
| HIST-EDA-01 | `notebooks/01_eda.ipynb`, `notebooks/reports/01_eda_report.md` | EDA, overlap checks, ambiguity analysis, label distribution | No model score; reports short texts, imbalance, and normalization benefit | Useful, must rerun |
| HIST-SVM-01 | `notebooks/02_baseline_models.ipynb`, `submissions/svm_baseline.csv` | Character TF-IDF `(3,6)` + `LinearSVC`, balanced class weights | Reported validation accuracy 0.5693 | Historical baseline, must rerun |
| HIST-SVM-02 | `notebooks/03_improved_training_accuracy.ipynb`, `submissions/svm_improved_training_accuracy.csv` | Character TF-IDF `(2,5)` + `LinearSVC(C=2)`, no balanced weights | Reported validation accuracy 0.6034; final training accuracy 0.8596 | Candidate baseline, must rerun |
| HIST-AUG-01 | `notebooks/03_improved_training_accuracy.ipynb` | SVM with ICD-description augmentation | Reported validation accuracy around 0.619 | Promising but optional and not final until cross-validated |
| HIST-ROBERTA-01 | `notebooks/04_dl_baseline_roberta.ipynb` | Spanish biomedical clinical RoBERTa backbone | No trustworthy final score found; notebook contains an error output | Incomplete exploratory attempt |

**Next experiment to run once data is available:** a clean data and annotation audit that confirms row counts, category set, duplicate literals, ambiguous mappings, and exact submission format.

## 2026-05-31 — Skeleton Smoke-Test Plan

**Goal:** Prepare runnable model-version entry points before rerunning metrics.

**Status:** Scaffold created. The next executable checks are:

- `python -m compileall src models`
- `python models/v00_majority_baseline.py --dry-run`
- `python models/v02_tfidf_word_svm.py --dry-run`

Full training still requires restoring the raw Kaggle CSV files into
`data/raw/`.

## No model yet; EDA conclusions only

Generated the visual EDA figures and tables. No model training, validation, or prediction was run.

## No model yet; preprocessing conclusions only

Created required-clean processed datasets and preprocessing ablation summaries. No model training, validation, or prediction was run.

## No model yet; tokenizer analysis only

Analyzed RoBERTa tokenizer lengths with `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`. Recommended max_length=32. No model training, validation, or prediction was run.

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

## Model Version Interface Defined

Defined the strict model-version contract before implementing additional
models. The interface standardizes config saving, 80/20 stratified validation,
metrics, detailed predictions, exact Kaggle submissions, checkpoints, run
summaries, and experiment-log updates. A tiny debug sample can be used for smoke
tests without expensive training.

## Run v00_majority_baseline

**Version:** `v00_majority_baseline`

Metrics:
- `accuracy`: 0.125182
- `macro_f1`: 0.006181
- `weighted_f1`: 0.027854

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v00_majority_baseline_run.md`

Interpretation:
This is the sanity baseline for class imbalance. It predicts only the majority
category `Z`, so the 12.5% validation accuracy comes from the label prior rather
than from clinical text understanding. All later models must beat this result,
especially in macro F1.

## Run v01_tfidf_char_logreg_debug

**Version:** `v01_tfidf_char_logreg_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v01_tfidf_char_logreg_debug_run.md`

## Run v02_tfidf_word_svm_debug

**Version:** `v02_tfidf_word_svm_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v02_tfidf_word_svm_debug_run.md`

## Run v03_roberta_cls_debug

**Version:** `v03_roberta_cls_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v03_roberta_cls_debug_run.md`

## Run v04_roberta_mean_debug

**Version:** `v04_roberta_mean_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_mean_debug_run.md`

## Run v05_roberta_mean_class_weighted_debug

**Version:** `v05_roberta_mean_class_weighted_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_class_weighted_debug_run.md`

## Run v06_roberta_mean_augmented_debug

**Version:** `v06_roberta_mean_augmented_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_augmented_debug_run.md`

## Run v07_ensemble_debug

**Version:** `v07_ensemble_debug`

Metrics:
- `accuracy`: 0.045455
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v07_ensemble_debug_run.md`

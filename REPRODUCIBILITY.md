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
python models/v00_majority_baseline.py --dry-run --load-model-path outputs/checkpoints/v00_majority_baseline_debug.joblib
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

Continue the non-modeling EDA with:

```bash
python scripts/run_visual_eda.py
```

This command generates the first report figures and EDA tables, including class
distribution, long-tail analysis, literal lengths, duplicate/ambiguous literals,
text-pattern prevalence, and train/leaderboard shift checks.

Create the required-clean preprocessing files with:

```bash
python scripts/run_preprocessing.py
```

This command writes:

- `data/processed/train_required_clean.csv`
- `data/processed/leaderboard_required_clean.csv`
- `data/interim/preprocessing_ablation/preprocessing_ablation_summary.csv`
- `data/interim/preprocessing_ablation/preprocessing_ablation_examples.csv`

It does not train any model.

Analyze RoBERTa tokenization before training with:

```bash
python scripts/analyze_tokenization.py
```

This command loads
`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, tokenizes the required-clean
train and leaderboard literals, and writes:

- `reports/tables/token_length_summary.csv`
- `reports/tables/truncation_by_max_length.csv`
- `reports/figures/fig_07_token_length_distribution.png`
- `reports/figures/fig_08_truncation_rate_by_max_length.png`

If the tokenizer is not cached locally, this command needs network access to
download it from Hugging Face.

## 4. Reproduce Baselines

After placing data in `data/raw/`, run:

```bash
python models/v00_majority_baseline.py
python models/v01_tfidf_char_logreg.py
python models/v02_tfidf_word_svm.py
python models/v03_similarity_retrieval_baseline.py
```

`v01_tfidf_char_logreg.py` runs the internal character n-gram grid and writes:

- `reports/tables/v01_tfidf_char_grid.csv`
- `reports/tables/v01_tfidf_char_top_ngrams.csv`
- `outputs/checkpoints/v01_tfidf_char_logreg.joblib`
- `submissions/v01_tfidf_char_logreg_submission.csv`

`v02_tfidf_word_svm.py` runs the internal word n-gram grid and writes:

- `reports/tables/v02_tfidf_word_grid.csv`
- `reports/tables/classical_baseline_comparison.csv`
- `reports/figures/fig_10_classical_baseline_comparison.png`
- `outputs/checkpoints/v02_tfidf_word_svm.joblib`
- `submissions/v02_tfidf_word_svm_submission.csv`

`v03_similarity_retrieval_baseline.py` runs the optional survey-inspired
retrieval baseline and writes:

- `reports/tables/v03_similarity_retrieval_grid.csv`
- `reports/tables/v03_similarity_correct_neighbors.csv`
- `reports/tables/v03_similarity_wrong_neighbors.csv`
- `outputs/checkpoints/v03_similarity_retrieval_baseline.joblib`
- `submissions/v03_similarity_retrieval_baseline_submission.csv`

Before expensive model work, smoke-test the model-version interface:

```bash
python models/v00_majority_baseline.py --dry-run
```

This debug run uses a tiny sample but still exercises the full contract:
configuration, 80/20 split, validation metrics, detailed predictions, exact
`id,y_category` submission, checkpoint, run summary, and experiment-log update.

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

Raw data is expected under `data/raw/` and is intentionally ignored by git. The
metrics reported in the current notebooks should always come from rerunning the
scripts in this document, not from older exploratory notebooks.

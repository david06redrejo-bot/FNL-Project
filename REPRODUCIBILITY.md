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

## 4.1 Deep Learning Infrastructure Smoke Test

Before training the full RoBERTa model, verify the PyTorch data and training
contract with a tiny run:

```bash
python scripts/smoke_test_deep_learning_infra.py
```

This command loads the tokenizer from
`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, builds tiny train,
validation, and leaderboard datasets, trains a toy classifier on 16 examples,
evaluates on 8 examples, saves a checkpoint, and validates the exact
`id,y_category` submission format. It does **not** train RoBERTa.

Run the infrastructure tests with:

```bash
pytest -q tests/test_deep_learning_infrastructure.py tests/test_preprocessing.py
```

## 4.2 RoBERTa CLS Baseline

Run a tiny debug job first:

```bash
python models/v04_roberta_cls.py --debug
```

Run the full CLS-pooling baseline:

```bash
python models/v04_roberta_cls.py
```

Default full-training configuration:

- backbone: `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
- pooling: CLS token, `hidden_states[:, 0, :]`
- learning rate: `2e-5`
- weight decay: `0.01`
- batch size: `128`
- max epochs: `50`
- patience: `10`
- default max length: `32`

The run writes:

- `outputs/checkpoints/v04_roberta_cls.pt`
- `outputs/metrics/v04_roberta_cls_metrics.json`
- `outputs/logs/v04_roberta_cls_history.csv`
- `outputs/predictions/v04_roberta_cls_val_predictions.csv`
- `outputs/predictions/v04_roberta_cls_leaderboard_detailed.csv`
- `submissions/v04_roberta_cls_submission.csv`

## 4.3 RoBERTa Mean-Pooling Baseline

Run a tiny debug job first:

```bash
python models/v05_roberta_mean.py --debug
```

Run the full mean-pooling baseline:

```bash
python models/v05_roberta_mean.py
```

The run uses the same split and hyperparameters as the CLS baseline, but pools
the last hidden states by averaging over non-padding tokens. It writes:

- `outputs/checkpoints/v05_roberta_mean.pt`
- `outputs/metrics/v05_roberta_mean_metrics.json`
- `outputs/logs/v05_roberta_mean_history.csv`
- `outputs/predictions/v05_roberta_mean_val_predictions.csv`
- `outputs/predictions/v05_roberta_mean_leaderboard_detailed.csv`
- `submissions/v05_roberta_mean_submission.csv`
- `reports/tables/roberta_pooling_comparison.csv`

## 4.4 Imbalance-Aware RoBERTa Mean-Pooling Variants

Run a tiny debug grid first:

```bash
python models/v06_roberta_mean_imbalance_aware.py --debug
```

Run the full imbalance-aware grid:

```bash
python models/v06_roberta_mean_imbalance_aware.py
```

By default this compares class-weighted CrossEntropyLoss and focal loss with
gamma values 1 and 2. Focal gammas are configurable:

```bash
python models/v06_roberta_mean_imbalance_aware.py --focal-gammas 1,2
```

The run writes:

- `reports/tables/v06_imbalance_aware_grid.csv`
- `reports/tables/v06_roberta_mean_imbalance_aware_class_weights.csv`
- `reports/tables/v06_per_class_recall_vs_v05.csv`
- `outputs/metrics/v06_roberta_mean_imbalance_aware_metrics.json`
- `submissions/v06_roberta_mean_imbalance_aware_submission.csv`

## 4.5 Controlled RoBERTa Mean-Pooling Tuning

Run Stage A quick checks:

```bash
python models/v07_roberta_mean_tuning.py --stage stage_a
```

Run Stage B medium subset checks:

```bash
python models/v07_roberta_mean_tuning.py --stage stage_b
```

Run Stage C final full run:

```bash
python models/v07_roberta_mean_tuning.py --stage stage_c
```

The tuning script writes:

- `reports/tables/v07_tuning_results.csv`
- run configs under `outputs/logs/`
- training curves under `reports/figures/`
- `outputs/metrics/v07_roberta_mean_tuning_metrics.json`
- `submissions/v07_roberta_mean_tuning_submission.csv`

## 4.6 Safe Data-Strategy Experiments

Run the safe clinical data-strategy experiments:

```bash
python models/v08_roberta_mean_augmented.py
```

For a tiny smoke test:

```bash
python models/v08_roberta_mean_augmented.py --debug
```

The script writes:

- `reports/tables/v08_data_strategy_results.csv`
- `reports/tables/v08_roberta_mean_augmented_duplicate_report.csv`
- `reports/safe_augmentation_note.md`
- `outputs/metrics/v08_roberta_mean_augmented_metrics.json`
- `submissions/v08_roberta_mean_augmented_submission.csv`

## 4.7 Ensemble

Run the final validation-driven ensemble over completed model predictions:

```bash
python models/v09_ensemble.py
```

Validate that all ensemble inputs are aligned without writing final artifacts:

```bash
python models/v09_ensemble.py --dry-run
```

The script writes:

- `reports/tables/v09_ensemble_comparison.csv`
- `outputs/logs/v09_ensemble_recipe.md`
- `outputs/metrics/v09_ensemble_metrics.json`
- `outputs/predictions/v09_ensemble_val_predictions.csv`
- `outputs/predictions/v09_ensemble_leaderboard_detailed.csv`
- `submissions/v09_ensemble_submission.csv`

## 4.8 Final Evaluation and Error Analysis

Generate final comparison tables and figures:

```bash
python -m src.evaluation
```

This command writes:

- `reports/tables/final_experiment_comparison.csv`
- `reports/tables/final_per_class_metrics.csv`
- `reports/tables/final_error_examples.csv`
- `reports/tables/final_top_confusions.csv`
- `reports/figures/fig_10_model_comparison.png`
- `reports/figures/fig_11_final_confusion_matrix.png`
- `reports/figures/fig_12_per_class_recall.png`
- `reports/figures/fig_13_confidence_correct_vs_wrong.png`
- `reports/figures/fig_14_training_curves.png`

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

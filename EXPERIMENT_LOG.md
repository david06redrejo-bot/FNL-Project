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

## Run v01_tfidf_char_logreg

**Version:** `v01_tfidf_char_logreg`

Metrics:
- `accuracy`: 0.522628
- `macro_f1`: 0.402554
- `weighted_f1`: 0.494943

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v01_tfidf_char_logreg_run.md`

Best validation configuration:
- preprocessing: required-clean
- TF-IDF analyzer: `char_wb`
- n-gram range: `(3, 5)`
- classifier: LogisticRegression
- class weight: none

Grid results were saved to `reports/tables/v01_tfidf_char_grid.csv`.
Feature interpretation was saved to
`reports/tables/v01_tfidf_char_top_ngrams.csv`.

Interpretation:
This is the first model that clearly learns from the literal text instead of
only from the label prior. It improves accuracy from 0.1252 to 0.5226 and macro
F1 from 0.0062 to 0.4026. Lowercasing tied the required-clean variant but did
not improve it, so the conservative preprocessing decision remains unchanged.

## Run v02_tfidf_word_svm

**Version:** `v02_tfidf_word_svm`

Metrics:
- `accuracy`: 0.520073
- `macro_f1`: 0.474196
- `weighted_f1`: 0.514018

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v02_tfidf_word_svm_run.md`

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

## Run v05_roberta_mean

**Version:** `v05_roberta_mean`

Metrics:
- `accuracy`: 0.564599
- `macro_f1`: 0.496567
- `weighted_f1`: 0.549541

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_run.md`

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

## Run v03_similarity_retrieval_baseline

**Version:** `v03_similarity_retrieval_baseline`

Metrics:
- `accuracy`: 0.497445
- `macro_f1`: 0.462789
- `weighted_f1`: 0.496120

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v03_similarity_retrieval_baseline_run.md`

Best validation configuration:
- retrieval index: training literals
- representation: TF-IDF `char_wb`
- n-gram range: `(3, 5)`
- k: 1

Grid results were saved to `reports/tables/v03_similarity_retrieval_grid.csv`.
Correct and wrong nearest-neighbor examples were saved to
`reports/tables/v03_similarity_correct_neighbors.csv` and
`reports/tables/v03_similarity_wrong_neighbors.csv`.

Interpretation:
The retrieval baseline is useful but not the strongest classical method. It is
below `v01_tfidf_char_logreg` in accuracy and below `v02_tfidf_word_svm` in macro
F1. The ICD-description retrieval variants performed worse than training-literal
retrieval, probably because the short clinical literals do not always match the
formal ICD description wording. We keep this method as a survey-inspired
ablation, not as the main modeling direction.

## Run v04_roberta_cls

**Version:** `v04_roberta_cls`

Metrics:
- `accuracy`: 0.569343
- `macro_f1`: 0.494329
- `weighted_f1`: 0.554347

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_cls_run.md`

Configuration:
- backbone: `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
- pooling: CLS token, `hidden_states[:, 0, :]`
- hidden size: 768
- number of classes: 36
- dropout: 0.1
- optimizer: AdamW
- learning rate: 2e-5
- weight decay: 0.01
- batch size: 128
- max epochs: 50
- patience: 10
- best epoch: 10

Artifacts:
- history: `outputs/logs/v04_roberta_cls_history.csv`
- training curves: `reports/figures/fig_11_roberta_cls_training_curves.png`
- submission: `submissions/v04_roberta_cls_submission.csv`

Interpretation:
This is the first full Transformer baseline. It improves validation accuracy
over the strongest classical baseline (`0.5693` vs `0.5226`) and improves
weighted F1 over the word TF-IDF SVM (`0.5543` vs `0.5140`). Macro F1 is similar
to the best classical baselines, which means minority-category behavior remains
an important target for the next models.

## Run v05_roberta_mean

**Version:** `v05_roberta_mean`

Metrics:
- `accuracy`: 0.564599
- `macro_f1`: 0.496567
- `weighted_f1`: 0.549541

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_run.md`

Configuration:
- backbone: `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
- pooling: attention-mask-aware mean pooling over non-padding tokens
- hidden size: 768
- number of classes: 36
- dropout: 0.1
- optimizer: AdamW
- learning rate: 2e-5
- weight decay: 0.01
- batch size: 128
- max epochs: 50
- patience: 10
- best epoch: 10

Artifacts:
- history: `outputs/logs/v05_roberta_mean_history.csv`
- training curves: `reports/figures/fig_12_roberta_mean_training_curves.png`
- submission: `submissions/v05_roberta_mean_submission.csv`
- pooling comparison: `reports/tables/roberta_pooling_comparison.csv`

Interpretation:
Mean pooling did not beat CLS in validation accuracy (`0.5646` vs `0.5693`), so
CLS remains the current candidate final model by the competition-oriented
selection criterion. Mean pooling did slightly improve macro F1 (`0.4966` vs
`0.4943`), which suggests that averaging token evidence may help some
minority-category behavior on short literals.

## Advanced Experiment Roadmap

No new advanced model was trained in this step. We created a structured roadmap
before adding more model variants.

Inputs used:
- EDA: imbalance, duplicates, ambiguous literals, possible distribution shift.
- Preprocessing: conservative required-clean text remains the default.
- Baselines: TF-IDF and retrieval define strong non-neural references.
- RoBERTa CLS: current best validation accuracy, `0.569343`.
- RoBERTa mean: slightly better macro F1, `0.496567`, but lower accuracy.

Decisions:
- implement next: class-weighted loss, learning-rate tuning, warmup scheduler,
  dropout tuning, ensembling, calibration/confidence analysis.
- maybe: focal loss, max-length tuning, freezing/unfreezing, label smoothing,
  safe data augmentation.
- future work: layer-wise learning-rate decay, pseudo-labeling.

Roadmap artifacts:
- `configs/experiments.yaml`
- `reports/tables/advanced_experiment_roadmap.csv`
- `notebooks/05_advanced_model_experiments.ipynb`

## Run v06_roberta_mean_imbalance_aware

**Version:** `v06_roberta_mean_imbalance_aware`

Grid:
- mean + class-weighted CrossEntropyLoss
- mean + focal loss gamma 1
- mean + focal loss gamma 2
- v05 mean standard CrossEntropyLoss included as reference

Best v06 candidate by validation accuracy:
- `v06_roberta_mean_imbalance_aware_focal_gamma1`

Metrics:
- `accuracy`: 0.557299
- `macro_f1`: 0.480394
- `weighted_f1`: 0.539776
- `best_epoch`: 9

Important comparison:
- v05 standard mean: accuracy 0.564599, macro F1 0.496567
- v06 focal gamma 1: accuracy 0.557299, macro F1 0.480394
- v06 class-weighted CE: accuracy 0.544526, macro F1 0.518270

Interpretation:
The imbalance-aware losses did not become the final model by validation
accuracy. Focal gamma 1 is the best v06 candidate by accuracy but remains below
v05 mean and v04 CLS. Class-weighted CE is the most useful imbalance ablation:
it improves macro F1 substantially, but the accuracy drop is too large for the
competition-oriented final criterion.

Artifacts:
- grid: `reports/tables/v06_imbalance_aware_grid.csv`
- class weights: `reports/tables/v06_roberta_mean_imbalance_aware_class_weights.csv`
- recall comparison vs v05: `reports/tables/v06_per_class_recall_vs_v05.csv`
- canonical submission: `submissions/v06_roberta_mean_imbalance_aware_submission.csv`

## Run v07_roberta_mean_tuning

**Version:** `v07_roberta_mean_tuning`

Staged search:
- Stage A: quick subset sanity checks for max_length, dropout, warmup, weight
  decay, and AMP.
- Stage B: medium subset runs for promising learning-rate/dropout settings.
- Stage C: one final full run with the recommended controlled configuration.

Best candidate:
- `c_recommended_32_lr2e5_warmup006_clip`

Recommended mean-pooling training configuration:
- max_length: 32
- learning_rate: 2e-5
- batch_size: 128
- dropout: 0.1
- weight_decay: 0.01
- scheduler: linear warmup
- warmup_ratio: 0.06
- gradient_clip: 1.0
- AMP: off for the final run

Metrics:
- `accuracy`: 0.564599
- `macro_f1`: 0.494151
- `weighted_f1`: 0.549054
- `best_epoch`: 13

Interpretation:
The tuned mean-pooling configuration matched the standard v05 mean-pooling
accuracy but did not improve over it, and it remains below the CLS model by
validation accuracy. The experiment still gives a useful recommended training
configuration for future mean-pooling runs: keep max_length 32 and lr 2e-5,
use batch size 128 when GPU memory allows, and use gradient clipping plus a
small warmup schedule. We did not expand into a large grid because the first
controlled runs did not show a clear gain.

Artifacts:
- tuning table: `reports/tables/v07_tuning_results.csv`
- top-run curves:
  `reports/figures/v07_roberta_mean_tuning_c_recommended_32_lr2e5_warmup006_clip_training_curves.png`
- canonical submission: `submissions/v07_roberta_mean_tuning_submission.csv`

## Run v05_roberta_mean_debug

**Version:** `v05_roberta_mean_debug`

Metrics:
- `accuracy`: 0.062500
- `macro_f1`: 0.016667
- `weighted_f1`: 0.015625

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_debug_run.md`

## Run v06_roberta_mean_imbalance_aware

**Version:** `v06_roberta_mean_imbalance_aware`

Metrics:
- `accuracy`: 0.557299
- `macro_f1`: 0.480394
- `weighted_f1`: 0.539776

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_run.md`

## Run v06_roberta_mean_imbalance_aware_debug_focal_gamma1

**Version:** `v06_roberta_mean_imbalance_aware_debug_focal_gamma1`

Metrics:
- `accuracy`: 0.000000
- `macro_f1`: 0.000000
- `weighted_f1`: 0.000000

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_run.md`

## Run v06_roberta_mean_imbalance_aware_debug_focal_gamma2

**Version:** `v06_roberta_mean_imbalance_aware_debug_focal_gamma2`

Metrics:
- `accuracy`: 0.062500
- `macro_f1`: 0.012987
- `weighted_f1`: 0.011364

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_focal_gamma2_run.md`

## Run v06_roberta_mean_imbalance_aware_class_weight_balanced

**Version:** `v06_roberta_mean_imbalance_aware_class_weight_balanced`

Metrics:
- `accuracy`: 0.544526
- `macro_f1`: 0.518270
- `weighted_f1`: 0.537546

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_class_weight_balanced_run.md`

## Run v06_roberta_mean_imbalance_aware_focal_gamma1

**Version:** `v06_roberta_mean_imbalance_aware_focal_gamma1`

Metrics:
- `accuracy`: 0.557299
- `macro_f1`: 0.480394
- `weighted_f1`: 0.539776

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma1_run.md`

## Run v06_roberta_mean_imbalance_aware_focal_gamma2

**Version:** `v06_roberta_mean_imbalance_aware_focal_gamma2`

Metrics:
- `accuracy`: 0.555474
- `macro_f1`: 0.492846
- `weighted_f1`: 0.542396

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma2_run.md`

## Run v07_roberta_mean_tuning

**Version:** `v07_roberta_mean_tuning`

Metrics:
- `accuracy`: 0.564599
- `macro_f1`: 0.494151
- `weighted_f1`: 0.549054

Run summary: `/home/iadlG010/FNL-Project/outputs/logs/v07_roberta_mean_tuning_run.md`

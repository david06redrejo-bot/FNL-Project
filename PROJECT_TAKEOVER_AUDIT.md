# Project Takeover Audit

Date: 2026-05-31  
Project: Fundamentals of Natural Language / NLP-I, Team 10  
Team: Phoebe Iglesias (1713459), David Redrejo (1790336), Pau Rossell (1750424)  
Supervisors: Ernest Valveny and Lei Kang  
Competition: Kaggle UAB-ASHO AI Codification (`uab-asho-ai-codification`)  
Task: predict exactly one ICD category prefix (`y_category`) for each clinical literal.

## 1. Takeover Summary

This repository already contains useful exploratory work, baseline notebooks, a literature review, plots, presentation PDFs, and two SVM submission CSV files. However, it is not yet a clean final project repository. The most important takeover finding is that the expected `data/` folder is not present in this checkout, even though the README and notebooks depend on it. There is also no committed `models/` folder, no final LaTeX report, no reproducible experiment runner, and no verified final pipeline.

The existing work should be treated as a partial student project history: useful for narrative and hypotheses, but not automatically trusted until rerun from the raw dataset.

## 2. Required Command Output

Command run:

```bash
find . -maxdepth 3 -type f | sort
```

Output found:

```text
./.git/FETCH_HEAD
./.git/HEAD
./.git/config
./.git/description
./.git/hooks/applypatch-msg.sample
./.git/hooks/commit-msg.sample
./.git/hooks/fsmonitor-watchman.sample
./.git/hooks/post-update.sample
./.git/hooks/pre-applypatch.sample
./.git/hooks/pre-commit.sample
./.git/hooks/pre-merge-commit.sample
./.git/hooks/pre-push.sample
./.git/hooks/pre-rebase.sample
./.git/hooks/pre-receive.sample
./.git/hooks/prepare-commit-msg.sample
./.git/hooks/push-to-checkout.sample
./.git/hooks/sendemail-validate.sample
./.git/hooks/update.sample
./.git/index
./.git/info/exclude
./.git/logs/HEAD
./.git/packed-refs
./.gitignore
./README.md
./docs/info/Project - Presentation.pdf
./docs/info/Project - Presentation2.pdf
./docs/info/survey_icd_coding.pdf
./docs/literature_review.md
./docs/plots/baseline_comparison.png
./docs/plots/label_distribution.png
./docs/plots/model_comparison.png
./docs/plots/pipeline_optimization_comparison.png
./docs/plots/text_lengths.png
./docs/presentation/group_10_FNL_Follow-up.pdf
./docs/presentation/presentation_summary.md
./docs/reports/01_data_exploration_report.md
./notebooks/01_eda.ipynb
./notebooks/02_baseline_models.ipynb
./notebooks/03_improved_training_accuracy.ipynb
./notebooks/04_dl_baseline_roberta.ipynb
./notebooks/reports/01_eda_report.md
./notebooks/reports/02_baseline_models_report.md
./notebooks/reports/03_improved_training_accuracy_report.md
./requirements.txt
./src/data_processing.py
./src/evaluation.py
./submissions/svm_baseline.csv
./submissions/svm_improved_training_accuracy.csv
```

## 3. Current Repository Tree

Ignoring `.git`, the visible project tree is:

```text
.
├── .agents/
├── .codex/
├── .gitignore
├── README.md
├── docs/
│   ├── info/
│   │   ├── Project - Presentation.pdf
│   │   ├── Project - Presentation2.pdf
│   │   └── survey_icd_coding.pdf
│   ├── literature_review.md
│   ├── plots/
│   │   ├── baseline_comparison.png
│   │   ├── label_distribution.png
│   │   ├── model_comparison.png
│   │   ├── pipeline_optimization_comparison.png
│   │   └── text_lengths.png
│   ├── presentation/
│   │   ├── group_10_FNL_Follow-up.pdf
│   │   └── presentation_summary.md
│   └── reports/
│       └── 01_data_exploration_report.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_improved_training_accuracy.ipynb
│   ├── 04_dl_baseline_roberta.ipynb
│   └── reports/
│       ├── 01_eda_report.md
│       ├── 02_baseline_models_report.md
│       └── 03_improved_training_accuracy_report.md
├── requirements.txt
├── src/
│   ├── data_processing.py
│   └── evaluation.py
└── submissions/
    ├── svm_baseline.csv
    └── svm_improved_training_accuracy.csv
```

Directory check:

- Present: `docs/`, `notebooks/`, `src/`, `submissions/`, `.agents/`, `.codex/`, `.git/`.
- Missing from checkout: `data/`, `models/`, `configs/`, `experiments/`, `reports/latex/` or equivalent final report folder.
- `.agents/` and `.codex/` are visible directories but contain no files at max depth 2.

## 4. Existing Assets

### Documentation and narrative

- `README.md`: short project overview, setup notes, and folder descriptions. It still describes the broader problem as hierarchical ICD coding and multi-label clinical data, while the current competition target is exactly one category prefix per literal.
- `docs/literature_review.md`: substantial survey-based literature review. Useful as a reference-methods section, especially for TF-IDF/SVM, n-grams, Naive Bayes, and the transition to neural methods.
- `docs/reports/01_data_exploration_report.md`: older EDA-style report with dataset statistics and task observations.
- `notebooks/reports/01_eda_report.md`: stronger narrative summary of notebook 01, including overlap checks, label imbalance, short-text properties, ambiguity, and recommended modeling strategy.
- `notebooks/reports/02_baseline_models_report.md`: detailed TF-IDF + Linear SVM baseline report. Reported validation accuracy: 0.5693.
- `notebooks/reports/03_improved_training_accuracy_report.md`: detailed improved SVM report. Reported best validation accuracy: 0.6034 for char `(2,5)`, `C=2`, no balanced class weights. Also reports optional ICD-description augmentation around 0.619 validation accuracy, but this is flagged as optional and not adopted.
- `docs/presentation/presentation_summary.md`: concise proposed solution summary.

### Notebooks

- `notebooks/01_eda.ipynb`: EDA notebook. Inspection found 31 cells: 17 markdown, 14 code, 13 executed code cells, and 1 code cell with error output.
- `notebooks/02_baseline_models.ipynb`: baseline TF-IDF + SVM notebook. Inspection found 23 cells: 10 markdown, 13 code, all 13 code cells executed, no error outputs.
- `notebooks/03_improved_training_accuracy.ipynb`: improved TF-IDF + SVM notebook. Inspection found 27 cells: 15 markdown, 12 code, all 12 code cells executed, no error outputs.
- `notebooks/04_dl_baseline_roberta.ipynb`: RoBERTa baseline notebook using `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`. Inspection found 14 cells: 7 markdown, 7 code, 6 executed code cells, and 1 code cell with error output. It appears to have run on CPU and includes a failed or interrupted training section. It creates `../models` in notebook code, but no `models/` directory is present now.

### Source code

- `src/data_processing.py`: contains text cleaning, accent-stripping normalization, ICD category extraction, category dataset preparation, and stratified splitting.
- `src/evaluation.py`: contains accuracy/F1/precision/recall metrics, classification report printing, comparison plotting, and submission generation.

### Submissions

- `submissions/svm_baseline.csv`: 6,668 lines including header, so 6,667 predictions. Columns: `id`, `Literal`, `y_category`.
- `submissions/svm_improved_training_accuracy.csv`: 6,668 lines including header, so 6,667 predictions. Columns: `id`, `Literal`, `y_category`.
- First rows of both current submissions are identical:

```text
id,Literal,y_category
1,AMNIODRENAJE,1
2,Hiperparatiroidismo primario,E
3,MIGRANYA parto,G
4,VHC,B
```

### Plots and PDFs

- `docs/plots/`: five PNG plots for baseline/model comparison, label distribution, pipeline optimization, and text lengths.
- `docs/info/`: course/project PDFs and `survey_icd_coding.pdf`.
- `docs/presentation/group_10_FNL_Follow-up.pdf`: existing follow-up presentation.

## 5. Suspected Problems

1. `data/` is absent from the current checkout. The project cannot be rerun as-is, even though the user context says dataset files should be in `data/`.
2. `models/` is absent. The RoBERTa notebook refers to `../models/best_roberta_baseline.pt`, but no saved model artifact exists.
3. The README is too generic and partly inaccurate for the final Kaggle task. It mentions hierarchical ICD codes and multi-label data, while the target is single-label category prefix prediction.
4. Existing reported results are not currently reproducible because the raw data files are missing.
5. Some reports may duplicate or conflict with each other (`docs/reports/01_data_exploration_report.md` and `notebooks/reports/01_eda_report.md`).
6. Notebook 04 contains error output and appears incomplete. It should be considered an exploratory attempt, not a final deep learning result.
7. The submission generator in `src/evaluation.py` fills empty predictions with `"null"`. For this competition, valid outputs should be one of 36 categories unless the rules explicitly permit `"null"`.
8. `generate_submission()` writes `id`, `Literal`, `y_category`, while notebook 04 writes only `id`, `y_category`. The required Kaggle format must be confirmed and standardized.
9. `prepare_category_dataset()` collapses duplicate literals by majority vote. This is practical for single-label training but hides real ambiguity; the final report should explain it clearly.
10. There is no automated script to reproduce EDA, training, evaluation, and submission outside notebooks.
11. There is no environment lockfile or pinned dependency versions.
12. There is no final LaTeX report source.
13. There is no final presentation source, only PDF and a short markdown summary.
14. There is no formal experiment log, decision log, or report notes file before this takeover.

## 6. Usable / Refactor / Archive / Rebuild

### Usable

- EDA observations from `notebooks/reports/01_eda_report.md`, after rerunning against the available dataset.
- Baseline and improved SVM modeling ideas from notebooks 02 and 03.
- `src/data_processing.py` as a starting point for shared preprocessing.
- `src/evaluation.py` as a starting point for metric and submission utilities.
- `docs/literature_review.md` as source material for the related-work section.
- Existing plots, if regenerated or matched to reproducible notebooks.
- Existing submission CSVs as historical artifacts, not final outputs yet.

### Refactor

- Convert notebook logic into scripts under a cleaner `src/` structure.
- Standardize submission format in one function.
- Clarify normalization variants: one for TF-IDF and one for transformer models.
- Replace print-heavy data preparation with functions that return structured summaries.
- Move repeated notebook reports into a single coherent report narrative.
- Update README to match the exact competition target and reproducibility flow.

### Archive

- Keep existing PDFs and submissions for project history.
- Keep the incomplete RoBERTa notebook as an exploratory attempt unless it is rebuilt into a clean, reproducible experiment.
- Keep duplicated reports until a final report is written; then mark them as intermediate notes rather than canonical outputs.

### Rebuild

- Rebuild the data audit from raw `data/` once available.
- Rebuild all model results from a reproducible script or notebook run.
- Rebuild RoBERTa training cleanly, preferably with explicit CPU/GPU assumptions and saved metrics.
- Rebuild final report in LaTeX.
- Rebuild final presentation support from the final results.

## 7. Missing Deliverables

- Verified `data/` directory in this checkout.
- Reproducible command-line training/evaluation scripts.
- Clean `configs/` for model settings.
- Saved model artifacts or explicit instructions to regenerate them.
- Final selected submission with provenance.
- Final LaTeX report source.
- Final presentation source/materials.
- Reproducibility documentation with exact commands and expected outputs.
- Experiment tracking file.
- Decision log.
- Coherent README aligned with the actual Kaggle task.

## 8. Risk List

| Risk | Impact | Mitigation |
|---|---|---|
| Missing dataset files | Cannot reproduce results | Restore `data/` locally or document expected private data placement |
| Existing results unverified | False claims in report | Rerun every reported metric before final writing |
| Ambiguous literals with multiple categories | Label noise and inconsistent training target | Quantify ambiguity; justify majority vote or alternative strategy |
| Severe class imbalance | Rare categories may be ignored | Report per-class performance, macro-F1, and confusion patterns |
| Submission format mismatch | Kaggle rejection or invalid score | Confirm sample submission format and enforce it centrally |
| RoBERTa notebook incomplete | Misleading deep learning story | Treat as exploratory until rerun successfully |
| Overfitting to validation split | Inflated model confidence | Use cross-validation or multiple seeds for final model selection |
| Missing pinned dependencies | Reproducibility failures | Add pinned requirements or environment file |
| Duplicated reports | Confusing final repository | Consolidate into final LaTeX report and README |

## 9. Proposed Final Architecture

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── svm_char.yaml
│   ├── svm_augmented.yaml
│   └── roberta.yaml
├── data/
│   ├── README.md
│   ├── codification_data.csv
│   ├── icd_d_p_pairs.csv
│   └── leaderboard_data.csv
├── docs/
│   ├── literature_review.md
│   ├── plots/
│   ├── presentation/
│   └── reports/
├── notebooks/
│   ├── 01_data_and_annotations.ipynb
│   ├── 02_task_challenges.ipynb
│   ├── 03_reference_methods_and_baselines.ipynb
│   ├── 04_roberta_backbone.ipynb
│   └── 05_final_evaluation_and_submission.ipynb
├── report/
│   ├── main.tex
│   ├── sections/
│   └── figures/
├── src/
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models/
│   │   ├── exact_match.py
│   │   ├── svm.py
│   │   └── roberta.py
│   └── submission.py
├── scripts/
│   ├── run_eda.py
│   ├── train_svm.py
│   ├── train_roberta.py
│   ├── evaluate.py
│   └── make_submission.py
├── submissions/
├── PROJECT_TAKEOVER_AUDIT.md
├── DECISIONS.md
├── REPORT_NOTES.md
└── EXPERIMENT_LOG.md
```

The `data/`, `models/`, and possibly large `experiments/` artifacts should remain untracked if they contain private, generated, or large files. The repository should still document exactly how to recreate them.

## 10. Recommended Order of Work

The first two real technical phases must be:

1. **Analyzing the data and the annotations**
   - Restore or verify `data/`.
   - Confirm row counts, columns, missing values, duplicates, code prefixes, 36 expected categories, and leaderboard format.
   - Quantify literal-code ambiguity and category imbalance.
   - Confirm exact Kaggle submission format.

2. **Understanding the main challenges of the task**
   - Summarize short-text behavior, Spanish/Catalan mixing, abbreviations, accents, digits, noisy casing, and long-tail labels.
   - Compare clinical literals with official ICD descriptions.
   - Identify what exact matching can solve and where generalization is needed.

Then continue with:

3. Reference methods from the survey and course framing.
4. Reproducible exact-match and TF-IDF baselines.
5. Improved SVM models, including controlled feature and hyperparameter changes.
6. Optional ICD-description augmentation with cross-validation.
7. RoBERTa backbone experiment, rebuilt cleanly and compared honestly.
8. Final evaluation, error analysis, and selected submission.
9. Final LaTeX report and presentation.
10. README and reproducibility cleanup.

## 11. Initial Takeover Decision

No project files were deleted. The existing repository should be preserved as project history while the next work moves toward reproducible scripts, cleaned notebooks, and a coherent final report narrative.

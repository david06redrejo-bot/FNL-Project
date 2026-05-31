# Decision Log

This file records project decisions for Team 10's UAB-ASHO AI Codification project. Decisions should be short, dated, and tied to evidence when possible.

## 2026-05-31 — Start Takeover With Audit-First Workflow

**Decision:** Before changing models or deleting files, inspect the repository and create a written takeover audit.

**Reasoning:** The repository already contains notebooks, reports, submissions, and plots, but the current state is not guaranteed to be correct or reproducible. The dataset folder expected by the notebooks is not visible in this checkout, and some existing results are only reported in notebooks/reports.

**Implications:**

- Existing work will be treated as useful project history, not verified final evidence.
- No non-cache files should be deleted during takeover.
- The first real technical phase will be **Analyzing the data and the annotations**.
- The second real technical phase will be **Understanding the main challenges of the task**.
- Final work should tell the story: EDA -> preprocessing -> reference methods from the survey -> baselines -> RoBERTa backbone -> improved models -> evaluation -> submission -> report.

**Status:** Active.

## 2026-05-31 — Decision: y_category Is Derived From the First Character of Code

**Decision:** The supervised target for this project is `y_category`, computed as
`Code.astype(str).str[0]` and normalized to uppercase for label mapping.

**Reasoning:** The Kaggle task asks for exactly one ICD category prefix per
clinical literal. This keeps the project focused on category-level codification
rather than full ICD code prediction.

**Implications:**

- Training files must contain `Code` and `Literal`.
- Derived labels must be checked against the expected 36 categories: digits
  `0`-`9` and letters `A`-`Z`.
- Every model should train and evaluate on `y_category`, not the full `Code`.
- Any mismatch in observed classes must be written to the EDA outputs and
  report notes before modeling starts.

**Status:** Active.

## 2026-05-31 — Adopt Final Repository Skeleton

**Decision:** Organize the project around a professional final skeleton with
`src/` for reusable logic, `models/` for runnable versioned experiments,
`outputs/` for generated artifacts, `reports/` for LaTeX, and `presentations/`
for final communication material.

**Reasoning:** The final repository must be easy for professors and teammates to
evaluate. Notebooks should tell the story, but the core implementation must live
in importable Python modules and runnable model files.

**Implications:**

- Raw data and large generated artifacts stay out of Git.
- Every model version should write metrics, predictions, submissions, and notes.
- Historical notebooks remain as project history, while the new numbered
  notebooks define the final narrative.

**Status:** Active.

## EDA Before Modeling

Decision: continue with EDA conclusions only before training. The observed imbalance, duplicate literals, and ambiguous literal-category mappings must shape the baseline design and evaluation plan.

## Final Preprocessing Decision

Decision: use `clean_required` for the final RoBERTa pipeline. This preserves case, accents, punctuation, digits, and abbreviations while stripping leading/trailing spaces and collapsing repeated whitespace. Stronger normalization is reserved for classical-baseline ablations only.

## RoBERTa Tokenizer max_length Decision

Decision: use `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` as the tokenizer for RoBERTa experiments and set
the initial default `max_length` to **32**. This value is based
on the observed token length percentiles of required-clean train and leaderboard
literals, not on an arbitrary default. Future ablations may compare nearby
values, but this is the default starting point.

## Deep Learning Infrastructure Decision

Decision: build the RoBERTa phase around a reusable PyTorch dataset and training
utility layer before training the full model. `ICDLiteralDataset` uses the
required-clean literal text, the tokenizer from
`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, padding/truncation, and the
default `max_length=32`.

Reasoning: the classical baselines are now strong enough that the Transformer
stage must be reproducible and controlled. We need deterministic label mapping,
consistent batch structure, checkpointing, metric computation, and submission
format validation before running an expensive model.

Implications:

- Train/validation datasets return `input_ids`, `attention_mask`, and `labels`.
- Leaderboard datasets return `id`, `input_ids`, and `attention_mask` only.
- Seeds are set for Python, NumPy, PyTorch, and CUDA where available.
- Full RoBERTa training should reuse the shared dataset/training utilities
  instead of notebook-only code.

## Safe Clinical Data-Augmentation Decision

Decision: do not use aggressive augmentation for clinical literals. We do not
randomly delete medical words, replace clinical terms with unverified synonyms,
alter negation, or back-translate literals as if the label were guaranteed to
stay unchanged.

Reasoning: the literals are short and semantically dense. A small change in
punctuation, abbreviation, negation, or terminology can change the correct ICD
category. The dataset also contains many duplicate literals with conflicting
categories, so blindly deduplicating by literal would remove real ambiguity from
the task.

Implementation decision for `v08`: test only safe data strategies that do not
invent new clinical text:

- keep the original mean-pooling model as the reference;
- drop duplicate training literals only when all copies share the same
  `y_category`;
- keep all rows but use `WeightedRandomSampler` to expose rare categories more
  often during training.

Result: conservative deduplication improved mean-pooling accuracy to `0.568613`
but did not beat the CLS model (`0.569343`). Weighted sampling improved macro F1
to `0.522584`, but reduced accuracy to `0.542336`, so it remains an ablation
rather than the final accuracy candidate.

## Ensemble Decision

Decision: implement ensembling only after individual model versions were
available and validated. `v09_ensemble` uses saved predictions from completed
runs and selects among predefined recipes by validation accuracy.

Reasoning: ensembling is a legitimate Machine Learning strategy when base
models make partially different errors, but it can also hide weak methodology if
it is used before understanding the models. The project story should show EDA,
preprocessing, baselines, RoBERTa variants, and then ensemble as a final
combination step.

Constraints:

- no leaderboard labels are used;
- no public leaderboard feedback is used for recipe selection;
- the selected recipe must beat individual models on the internal validation
  split;
- if validation contradicts leaderboard intuition, validation wins.

Result: the selected recipe is majority vote over CLS, mean pooling,
safe-deduplicated mean pooling, weighted-sampler mean pooling, and TF-IDF
character logistic regression, with average-probability tie-breaking. It reached
validation accuracy `0.576642`, macro F1 `0.506277`, and weighted F1 `0.561544`.

## Final Model Candidate Decision

Decision: use `v10_vote_diverse_no_retrieval` as the final Kaggle public
submission candidate, while reporting the validation trade-off explicitly.

Reasoning: after the first ensemble pass, `v09_ensemble` improved over the best
individual model (`v04_roberta_cls`) on accuracy, macro F1, and weighted F1.
The later `v10` diverse-ensemble search improved validation accuracy and
produced the best verified public score:

```text
v04_roberta_cls                           accuracy 0.569343  macro_f1 0.494329  weighted_f1 0.554347  public 0.573
v09_ensemble                              accuracy 0.576642  macro_f1 0.506277  weighted_f1 0.561544  public 0.573
v10_vote_diverse_no_retrieval_val_weighted accuracy 0.583577  macro_f1 0.501074  weighted_f1 0.564988  public 0.586
v10_vote_diverse_no_retrieval             accuracy 0.579562  macro_f1 0.496677  weighted_f1 0.561294  public 0.587
```

Implications:

- the final report should present `v10_vote_diverse_no_retrieval` as the final
  public-leaderboard submission candidate;
- `v10_vote_diverse_no_retrieval_val_weighted` should be discussed as the best
  validation-accuracy recipe from the v10 search;
- `v09_ensemble` should remain in the analysis as the stronger macro-F1
  candidate among the main completed models;
- individual RoBERTa and TF-IDF models remain important for ablation and
  explanation;
- the project should state that no leaderboard labels were used, and that
  public leaderboard feedback is reported separately from validation metrics;
- remaining limitations are short-literal ambiguity, rare categories,
  abbreviations, and broad ICD-prefix confusions.

## Final Submission Declaration

Decision: after submitting individual models and broader diverse ensembles to
Kaggle, use `v10_vote_diverse_no_retrieval` as the final public-leaderboard
candidate and copy its Kaggle-format file to `submissions/final_submission.csv`.

Reasoning: the repository should expose one unambiguous final submission file
instead of forcing reviewers to choose among many experiment submissions.

Reasoning update: `v09_ensemble` was replaced because broader diverse ensembles
performed better. Kaggle public leaderboard scores showed:

```text
v10_vote_diverse_no_retrieval               public score 0.587
v10_vote_diverse_no_retrieval_val_weighted  public score 0.586
v08_safe_dedupe                             public score 0.583
v04_roberta_cls                             public score 0.573
v09_ensemble                                public score 0.573
```

Therefore the final upload candidate should follow the best verified public
score, while the report should clearly explain the validation/public difference.

Files:

- `submissions/final_submission.csv`
- `submissions/final_submission_two_column_internal.csv`
- `outputs/predictions/final_leaderboard_detailed.csv`
- `FINAL_MODEL_CARD.md`
- `SUBMISSION.md`
- `notebooks/08_submission_and_final_story.ipynb`

Competition result note: local files verify the public score table, but not the
private/final ranking. A final-placement claim must be supported by private
leaderboard evidence under `reports/tables/` or `reports/figures/`.

## Evaluation Notebook Decision

Decision: the main evaluation notebook uses `v10_vote_diverse_no_retrieval` as
the final candidate because it has the best verified public score and improves
validation accuracy over the previous `v09` ensemble.

Trade-off: `v09_ensemble` remains slightly stronger on macro F1, so the report
must not hide the imbalance perspective. The final recommendation is therefore:
use `v10` for the competition submission, and discuss `v09` as the stronger
macro-F1-oriented ensemble.

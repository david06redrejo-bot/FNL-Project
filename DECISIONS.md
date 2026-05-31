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

# ICD-10 Code Prediction from Short Clinical Literals

<p align="center">
  <strong>Fundamentals of Natural Language Processing · Universitat Autonoma de Barcelona · 2025-2026</strong><br/>
  Group 10 · Phoebe Iglesias · David Redrejo · Pau Rossell
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-2563EB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="NLP" src="https://img.shields.io/badge/NLP-Clinical%20Coding-14B8A6?style=for-the-badge"/>
  <img alt="Model" src="https://img.shields.io/badge/Best%20Model-Char%20TF--IDF%20%2B%20SVM-22C55E?style=for-the-badge"/>
  <img alt="GPU" src="https://img.shields.io/badge/GPU-RoBERTa%20Ready-A855F7?style=for-the-badge"/>
</p>

---

## The hook

Most ICD coding projects imagine long clinical notes. This one is sharper and stranger:

> Given a tiny clinical literal such as `HTA irc 6`, `VHC`, or `breast prosthesis`, predict the first character of its ICD code.

That means the model has almost no context to hide behind. It must learn from fragments, abbreviations, spelling variants, accents, numbers, and the quiet structure of the ICD code system itself.

The final target is:

```math
y_{category} = first(Code)
```

The best validated model in this repository is not the largest one. It is the model that understood the shape of the data best:

```text
normalized literal
  -> character n-grams
  -> TF-IDF weighting
  -> Linear SVM
  -> ICD category
  -> Kaggle-ready submission
```

<p align="center">
  <img src="docs/plots/readme_scorecard.svg" alt="ICD coding results snapshot" width="980"/>
</p>

---

## Kaggle standing

This repository contains the Kaggle-ready prediction files, but it does **not** currently contain an official Kaggle leaderboard screenshot, public score, or rank export. I therefore do not fabricate a position.

| Kaggle item | Current repository evidence |
|---|---|
| Recommended submission | `submissions/svm_improved_training_accuracy.csv` |
| Rows predicted | `6,667` leaderboard literals |
| Output columns | `id`, `Literal`, `y_category` |
| Categories covered | `36` ICD categories |
| Official Kaggle rank | Not stored in this repository snapshot |
| Official Kaggle score | Not stored in this repository snapshot |

When the official leaderboard position is available, update this block:

```text
Kaggle team:
Public score:
Private score:
Leaderboard rank:
Snapshot date:
```

The model we would submit first is:

```text
submissions/svm_improved_training_accuracy.csv
```

---

## Results at a glance

| Stage | Model / experiment | Train accuracy | Validation accuracy | What it taught us |
|---|---:|---:|---:|---|
| Baseline | Char TF-IDF `(3,6)` + LinearSVC, balanced weights | `0.8286` | `0.5701` | Strong first lexical baseline, but too cautious for short literals |
| Improved | Char TF-IDF `(2,5)` + LinearSVC, `C=2`, no class weights | `0.8680` | `0.6034` | Best validated model: higher train and validation performance |
| Higher capacity | Char TF-IDF `(2,6)` + LinearSVC, `C=5` | `0.8814` | `0.5982` | More capacity, slightly worse generalization |
| Word + char | FeatureUnion word/char TF-IDF + LinearSVC | `0.9162` | `0.5883` | Memorizes harder, generalizes worse |
| ICD descriptions | Optional augmentation with official ICD descriptions | lower on original literals | `~0.619` | Promising, but needs cross-validation before being the official model |

The central result is a simple one:

```text
The best model is not the one that memorizes most.
It is the one that improves validation while staying interpretable.
```

<p align="center">
  <img src="Report/improved_confusion_matrix.png" alt="Improved model confusion matrix" width="780"/>
</p>

The off-diagonal blocks are where short literals, numeric procedure families, and ambiguous clinical phrasing still collide.

---

## Repository layout

```text
configs/
  base.yaml
  experiments.yaml
  model_registry.yaml

data/
  raw/
  interim/
  processed/

docs/
  info/
  plots/
  presentation/

models/
  v00_majority_baseline.py
  ...
  v10_diverse_ensemble_search.py

notebooks/
  00_task_formulation_and_eda.ipynb
  01_data_preprocessing_and_annotation_design.ipynb
  02_baseline_models.ipynb
  02_reference_methods_from_survey.ipynb
  03_improved_training_accuracy.ipynb
  04_roberta_backbone_baseline.ipynb
  05_advanced_model_experiments.ipynb
  06_hyperparameter_and_ablation_studies.ipynb
  07_evaluation_error_analysis_and_interpretability.ipynb
  08_submission_and_final_story.ipynb

presentations/
  final_presentation/
  short_presentation/

results/
  compiled/
  figures/
  tables/
  final_report.tex
  references.bib
  safe_augmentation_note.md

submissions/
  final_submission.csv
  svm_baseline.csv
  svm_improved_training_accuracy.csv
  v00_majority_baseline_submission.csv
  ...
  v10_vote_validation_top_plus_tfidf_val_weighted_kaggle.csv

group_10_report_fnl_project.pdf
requirements.txt
```

---

## Recommended reading path

| Notebook | Main question | Why it matters |
|---|---|---|
| `01_eda.ipynb` | What kind of problem is this really? | Shows that this is short-literal coding, not long-document ICD coding |
| `02_baseline_models.ipynb` | How strong is a clean classical baseline? | Builds the first TF-IDF + SVM pipeline and submission |
| `03_improved_training_accuracy.ipynb` | Which baseline choices actually help? | Tunes n-grams, class weights, feature filtering, and capacity |
| `04_dl_baseline_roberta.ipynb` | Can a GPU transformer beat the lexical model? | Prepares the RoBERTa experiment for CUDA execution |

The notebooks are written as project chapters. They explain the decision, run the code, read the result, and then move to the next step.

---

## Installation

Recommended Python version: **Python 3.10+**.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---

## Credits

Project developed for **Fundamentals of Natural Language Processing**, Universitat Autonoma de Barcelona, 2025-2026.

Team:

```text
Phoebe Iglesias
David Redrejo
Pau Rossell
```

---

<p align="center">
  <strong>From tiny clinical literals to ICD categories.</strong><br/>
  Interpretable NLP first. GPU models only when we have something real to prove.
</p>

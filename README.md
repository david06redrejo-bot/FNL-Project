# UAB-ASHO AI Codification: Clinical Literal to ICD Category

Short clinical phrases carry administrative and medical weight. A literal such
as a diagnosis, procedure, allergy, or condition can determine how a case is
indexed in hospital records, counted in public-health statistics, grouped into
DRGs, audited, and reimbursed. Manual ICD coding is expert work: it is slow,
inconsistent under pressure, and difficult when clinical language is abbreviated,
ambiguous, multilingual, or incomplete.

This project explores how Natural Language Processing can help: given a clinical
literal, predict the ICD-10 category prefix that best codifies it.

## Project Identity

**Team 10**

- Phoebe Iglesias (1713459)
- David Redrejo (1790336)
- Pau Rossell (1750424)

**Course:** Fundamentals of Natural Language / NLP-I  
**University:** Universitat Autonoma de Barcelona  
**Academic year:** 2025-2026  
**Supervisors/professors:** Ernest Valveny and Lei Kang

## Competition

**Kaggle:** UAB-ASHO AI Codification (`uab-asho-ai-codification`)  
**Competition link:** <https://www.kaggle.com/competitions/uab-asho-ai-codification>

Kaggle competitions usually expose a public leaderboard during development and a
private leaderboard for final scoring. This repository selects the final model
using internal validation, not leaderboard labels or public-leaderboard tuning.

**Final competition ranking:** not verified in repository files yet. Evidence
should be added before making a formal ranking claim:

```text
reports/tables/kaggle_submission_scores.csv
reports/figures/kaggle_final_ranking.png
```

During the project, after the initial baseline, short presentation, and
follow-up with Lei Kang, the team was on the right track and at one point was
second in the competition. This is kept as project context and should be backed
by evidence if used in the final report.

## What The Model Does

Input:

```text
"Hiperparatiroidismo primario"
```

Output:

```text
y_category = "E"
```

The target is `y_category`, the first character of the ICD code. The task is a
single-label, 36-class multiclass classification problem: digits `0`-`9` and
letters `A`-`Z`.

## Architecture

![Final architecture diagram](reports/figures/final_architecture_diagram.png)

The final candidate is `v09_ensemble`: a validation-selected majority-vote
ensemble over complementary models. It combines Spanish biomedical RoBERTa
variants with a character-level TF-IDF baseline, because clinical literals can
contain both contextual meaning and important surface patterns such as
abbreviations, digits, punctuation, and fragments.

## Repository Structure

```text
.
├── data/                  # Raw, interim, and processed data
├── notebooks/             # Narrative notebooks from EDA to final submission
├── src/                   # Reusable loading, preprocessing, training, evaluation code
├── models/                # Runnable versioned model scripts
├── configs/               # Experiment plan and model registry
├── outputs/               # Metrics, predictions, logs, checkpoints
├── submissions/           # Kaggle-ready CSV files
├── reports/               # LaTeX report, figures, tables, references
├── presentations/         # Short and final presentation material
└── tests/                 # Lightweight correctness checks
```

## Main Results

Validation results on the shared internal split:

| Version | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| `v00` | Majority baseline | 0.125182 | 0.006181 | 0.027854 |
| `v01` | TF-IDF char n-grams + logistic regression | 0.522628 | 0.402554 | 0.494943 |
| `v02` | TF-IDF word n-grams + linear SVM | 0.520073 | 0.474196 | 0.514018 |
| `v03` | Similarity / retrieval baseline | 0.497445 | 0.462789 | 0.496120 |
| `v04` | RoBERTa CLS pooling | 0.569343 | 0.494329 | 0.554347 |
| `v05` | RoBERTa mean pooling | 0.564599 | 0.496567 | 0.549541 |
| `v06` | Imbalance-aware RoBERTa variants | 0.557299 | 0.480394 | 0.539776 |
| `v07` | RoBERTa mean-pooling tuning | 0.564599 | 0.494151 | 0.549054 |
| `v08` | Safe data-strategy RoBERTa | 0.568613 | 0.481648 | 0.549150 |
| `v09` | Final ensemble | **0.576642** | **0.506277** | **0.561544** |

Final submission:

```text
submissions/final_submission.csv
```

Detailed final leaderboard predictions:

```text
outputs/predictions/final_leaderboard_detailed.csv
```

## Model Evolution

We built the project progressively, because a strong final model is easier to
trust when every step has a reason.

1. **v00 majority baseline:** how much can imbalance alone explain?
2. **v01 char TF-IDF:** can morphology, abbreviations, digits, and fragments
   carry the signal?
3. **v02 word TF-IDF:** how far do word-level lexical features go?
4. **v03 retrieval baseline:** what happens if ICD coding is treated as nearest
   clinical-literal matching?
5. **v04 RoBERTa CLS:** first biomedical Spanish Transformer baseline.
6. **v05 RoBERTa mean:** alternative pooling for short literals.
7. **v06 imbalance-aware losses:** class weighting and focal loss.
8. **v07 controlled tuning:** max length, learning rate, warmup, dropout, weight
   decay, gradient clipping.
9. **v08 safe data strategies:** conservative duplicate handling and weighted
   sampling without unsafe clinical augmentation.
10. **v09 ensemble:** final validation-selected model combining complementary
    RoBERTa and TF-IDF signals.

## How To Reproduce

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the competition files in:

```text
data/raw/codification_data.csv
data/raw/leaderboard_data.csv
data/raw/icd_d_p_pairs.csv
```

Run the project phases:

```bash
# Data inventory and schema validation
python -m src.data_loading

# Preprocessing
python -m src.preprocessing

# Tokenization analysis
python scripts/analyze_tokenization.py

# Classical baselines
python models/v00_majority_baseline.py
python models/v01_tfidf_char_logreg.py
python models/v02_tfidf_word_svm.py
python models/v03_similarity_retrieval_baseline.py

# RoBERTa models and ablations
python models/v04_roberta_cls.py
python models/v05_roberta_mean.py
python models/v06_roberta_mean_imbalance_aware.py
python models/v07_roberta_mean_tuning.py
python models/v08_roberta_mean_augmented.py

# Final ensemble and evaluation
python models/v09_ensemble.py
python -m src.evaluation
```

Create the final submission copy:

```bash
python - <<'PY'
import pandas as pd

sub = pd.read_csv("submissions/v09_ensemble_submission.csv")
assert list(sub.columns) == ["id", "y_category"]
assert len(sub) == 6667
sub.to_csv("submissions/final_submission.csv", index=False)

detailed = pd.read_csv("outputs/predictions/v09_ensemble_leaderboard_detailed.csv")
detailed.to_csv("outputs/predictions/final_leaderboard_detailed.csv", index=False)
PY
```

For full commands and artifact descriptions, see `REPRODUCIBILITY.md`.

## Notebooks

The notebooks are the narrative version of the project:

| Notebook | Purpose |
|---|---|
| `00_task_formulation_and_eda.ipynb` | Task formulation, data and annotation analysis |
| `01_data_preprocessing_and_annotation_design.ipynb` | Preprocessing decisions and tokenizer analysis |
| `02_reference_methods_from_survey.ipynb` | ICD coding survey mapped to our project |
| `03_classical_baselines.ipynb` | Majority, TF-IDF, and retrieval baselines |
| `04_roberta_backbone_baseline.ipynb` | RoBERTa CLS and mean-pooling baselines |
| `05_advanced_model_experiments.ipynb` | Improvement roadmap, imbalance, ensemble strategy |
| `06_hyperparameter_and_ablation_studies.ipynb` | Tuning and safe data-strategy ablations |
| `07_evaluation_error_analysis_and_interpretability.ipynb` | Final evaluation, errors, confidence, interpretability |
| `08_submission_and_final_story.ipynb` | Final submission and project story |

## Reports And Presentations

- Final report source: `reports/final_report.tex`
- References: `reports/references.bib`
- Report tables: `reports/tables/`
- Report figures: `reports/figures/`
- Presentations: `presentations/`
- Final model card: `FINAL_MODEL_CARD.md`
- Submission instructions: `SUBMISSION.md`

## Ethical And Clinical Caution

This repository is an academic NLP competition project. It is **not** a medical
device, not a coding assistant ready for deployment, and not suitable for direct
clinical or billing use.

The model predicts broad ICD category prefixes from short literals, not full ICD
codes from complete medical records. Errors remain on ambiguous literals,
abbreviations, rare categories, and cases that require clinical context.
Clinical deployment would require expert validation, governance, privacy review,
monitoring, and clear accountability.

## Acknowledgements And References

We thank Ernest Valveny and Lei Kang for supervising the project in the
Fundamentals of Natural Language / NLP-I course at UAB.

This work was guided by:

- the Kaggle competition **UAB-ASHO AI Codification**;
- the ICD coding survey by Yan et al. (2022), used to frame baselines,
  challenges, and future work;
- the Hugging Face backbone
  `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`;
- course material on corpora, text processing, vector-space models,
  Transformers, evaluation, and responsible NLP.

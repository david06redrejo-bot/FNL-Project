# UAB-ASHO AI Codification — Team 10

ICD coding matters because a short clinical phrase can determine how a case is
indexed, audited, studied, and reimbursed. In this project we work on a compact
version of that problem: given a clinical literal, predict exactly one ICD-10
category prefix, `y_category`, defined as the first character of `Code`.

This repository is for the Fundamentals of Natural Language / NLP-I project at
Universitat Autonoma de Barcelona, academic year 2025-2026.

**Team 10**

- Phoebe Iglesias (1713459)
- David Redrejo (1790336)
- Pau Rossell (1750424)

**Supervisors/professors:** Ernest Valveny and Lei Kang  
**Competition:** Kaggle UAB-ASHO AI Codification (`uab-asho-ai-codification`)

## Project Story

The repository is organized around a reproducible project arc:

```text
EDA -> preprocessing -> reference methods -> baselines -> RoBERTa backbone
-> improved models -> evaluation -> submission -> report
```

The first real technical phase is **Analyzing the data and the annotations**.
The second phase is **Understanding the main challenges of the task**. From
there, the project moves into reference methods from the ICD coding survey,
classical baselines, RoBERTa experiments, ablations, and final evaluation.

## Repository Structure

```text
.
├── data/                  # Raw, interim, and processed data; raw CSVs are private
├── notebooks/             # Narrative notebooks, not the only source of logic
├── src/                   # Reusable project code
├── models/                # Runnable model-version scripts
├── configs/               # Experiment and model registry configs
├── outputs/               # Generated metrics, predictions, logs, checkpoints
├── submissions/           # Submission CSVs
├── reports/               # Final LaTeX report and report assets
├── presentations/         # Short and final presentation materials
└── tests/                 # Lightweight checks
```

## Final Architecture

Placeholder for final architecture image:

```text
reports/figures/final_architecture.png
```

## Main Results

Placeholder for the final result table. Historical notebook metrics must be
rerun before being reported as final.

| Model | Validation accuracy | Notes |
|---|---:|---|
| Majority baseline | TBD | Lower bound |
| TF-IDF + Logistic Regression | TBD | Classical baseline |
| TF-IDF + SVM | TBD | Survey-inspired baseline |
| RoBERTa backbone | TBD | Transformer baseline |
| Final selected model | TBD | Official submission candidate |

## Data Placement

Place Kaggle files in `data/raw/`:

```text
data/raw/codification_data.csv
data/raw/leaderboard_data.csv
data/raw/icd_d_p_pairs.csv
```

These files are ignored by Git because they are dataset artifacts, but the
folder structure is kept with `.gitkeep` files.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python models/v00_majority_baseline.py --dry-run
python models/v02_tfidf_word_svm.py --train --evaluate --predict --make-submission
```

Outputs are written to `outputs/metrics/`, `outputs/predictions/`,
`submissions/`, `EXPERIMENT_LOG.md`, and `REPORT_NOTES.md`.

For full reproduction instructions, see `REPRODUCIBILITY.md`.

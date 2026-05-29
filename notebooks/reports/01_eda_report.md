# Report: Notebook 01 — Exploratory Data Analysis

**Project:** Fundamentals of Natural Language Processing — ICD-10 Clinical Code Prediction  
**Students:** Phoebe Iglesias, David Redrejo & Pau Rossell  
**Notebook:** `01_eda.ipynb`

---

## 1. Objective

This notebook performs a comprehensive Exploratory Data Analysis (EDA) on the ASHO-AI ICD-10 codification task. It is structured as a **computational narrative**: each section explains *why* a given analysis is done, *what* is observed, and *how* the observation changes the next step. The goal is not merely to produce charts but to systematically demonstrate understanding of the task, the dataset, the annotations, the main challenges, and to arrive at a justified first modeling proposal.

---

## 2. Datasets Analysed

Three CSV files are loaded and enriched with surface-level features (character/token counts, presence of digits, punctuation, uppercase patterns, accents):

| Dataset | Rows | Key Columns | Unique Codes | Unique Literals/Descriptions |
|---|---:|---|---:|---:|
| `leaderboard_data.csv` | 6 667 | `id`, `Literal` | — | 6 102 |
| `codification_data.csv` | 13 700 | `Code`, `Literal` | 4 059 | 11 584 |
| `icd_d_p_pairs.csv` | 179 742 | `Code`, `D_P`, `Description` | 179 742 | 174 648 |

### Role of each file

- **Leaderboard data** — contains only `id` and `Literal`; these are the inputs the model must predict for (evaluation/submission set).
- **Codification data** — supervised `(Code, Literal)` pairs; the **main training resource**.
- **ICD catalog** — a much larger official ICD terminology catalog of codes and their official descriptions; serves as **external knowledge**, not a drop-in replacement for training data.

---

## 3. Key Analyses and Findings

### 3.1 Cross-dataset Overlap

A table of overlap checks reveals:

| Check | Share |
|---|---:|
| Leaderboard literal exactly in training literals | 27.2% |
| Leaderboard literal in training after normalization | 61.5% |
| Training code exists in ICD catalog | 72.6% |
| Training literal exactly matches ICD description | 2.0% |
| Training literal matches ICD description after norm. | 5.6% |

**Interpretation:**
- An **exact-match baseline** is mandatory because a non-trivial fraction of leaderboard literals repeat in training.
- **Normalization** (lowercasing, accent removal, punctuation simplification) significantly increases overlap, making it an essential preprocessing step.
- The ICD catalog is valuable as **knowledge** (72.6% code coverage), but the literal strings almost never match official descriptions directly (only 2–5.6%).

### 3.2 Surface-Form Analysis

Quantitative comparison of the three text sources:

| Dataset | Mean chars | Mean tokens | Contains digit | Contains punct. | All upper | Contains accent |
|---|---:|---:|---:|---:|---:|---:|
| Leaderboard literals | 17.15 | 2.22 | 7.6% | 9.0% | 11.3% | 29.6% |
| Training literals | 16.95 | 2.21 | 8.0% | 9.8% | 11.8% | 28.0% |
| ICD official descriptions | 80.88 | 10.77 | 2.9% | 85.3% | 0.0% | 85.7% |

**Key insight:** Clinical literals are **extremely short** (~2 tokens, ~17 characters) and **noisy** (mixed case, abbreviations, digits). ICD descriptions are vastly longer and more formal. This mismatch means a simple dictionary-based approach (matching literals to descriptions) will fail.

The notebook also includes **distribution histograms** of token counts and character counts, visually confirming the above findings.

### 3.3 Label Distribution and Code-Level Analysis

- **ICD code categories** span the full alphabet and digits 0–9 (36 categories when taking the first character of each code).
- The label distribution is **highly imbalanced**: categories like `Z` (1 561 samples) and `O` (1 169) dominate, while `W` (7), `X` (9), and `U` (12) are extremely rare.
- A significant number of ICD codes have very few training examples (strong long-tail effect).

### 3.4 Ambiguity: Many-to-Many Mappings

The EDA reveals a **many-to-many relationship** between literals and codes:

- Many codes have **multiple surface forms** (different literals map to the same code).
- Some literals map to **multiple different codes** — the same textual expression can correspond to different ICD codes depending on context.

**Examples of ambiguous literals** are displayed showing the same literal paired with different codes. This is a **decisive finding**: a pure official-description matcher or a simple exact-lookup will be weak. The task requires both **retrieval** (find similar known terms) and **disambiguation** (choose the right code among close candidates).

### 3.5 ICD Catalog Utility: Token Jaccard Similarity

A merge between training literals and ICD official descriptions (joined on `Code`) is performed. Token-level Jaccard similarity between the normalized literal and the normalized description is computed.

**Result:** The distribution of Jaccard similarity is **right-skewed** — most literal–description pairs have very low lexical overlap. Only a small fraction achieve Jaccard ≥ 0.5 (exact or near-exact match after normalization). The median is very low.

**Conclusion:** The ICD catalog is useful as supplementary knowledge (e.g. for data augmentation or additional training signal), but cannot serve as a direct matching table.

### 3.6 TF-IDF Retrieval Baseline

As a quick feasibility test, a TF-IDF cosine-similarity retrieval system is built:
- TF-IDF vectors are computed over character 3–6-grams on the training literals.
- For a given query literal, the system retrieves the most similar training literal and assigns its code.

**Finding:** This simple retrieval approach already achieves non-trivial performance on seen literals, validating the use of character n-grams for capturing the morphological patterns of short Spanish/Catalan medical terms.

---

## 4. Proposed Modeling Strategy

Based on all EDA findings, the notebook proposes a **two-stage pipeline**:

1. **Exact / near-exact match stage** — For literals that appear (after normalization) in the training set, directly assign the known code.
2. **Generalization stage** — For unseen literals, use a machine-learning model trained on character n-gram features (TF-IDF) to predict the ICD code or category.

### Justification

- **Character n-grams** are appropriate because the texts are very short and morphologically rich (abbreviations, typos, mixed case/accents).
- **TF-IDF + linear classifier** is a strong baseline for this type of sparse medical text classification task (supported by literature references in later notebooks).
- The ICD catalog can optionally be used for **knowledge augmentation** (adding official descriptions as extra training signal).

---

## 5. Summary

| Aspect | Finding |
|---|---|
| **Task type** | Clinical literal → ICD code classification (not full-document coding) |
| **Text characteristics** | Very short (~2 tokens), noisy, mixed Spanish/Catalan, abbreviations |
| **Key challenge** | Many-to-many literal–code mapping; extreme label imbalance; low overlap between clinical literals and official ICD descriptions |
| **Normalization** | Essential; doubles the leaderboard–training overlap from 27% to 62% |
| **ICD catalog** | Useful as supplementary knowledge but not as a direct lookup table |
| **Recommended approach** | Character n-gram TF-IDF features + supervised classifier; exact-match fallback for known literals |

This EDA establishes a solid empirical foundation for the subsequent modeling notebooks by quantifying the key challenges and validating the feasibility of the proposed approach.

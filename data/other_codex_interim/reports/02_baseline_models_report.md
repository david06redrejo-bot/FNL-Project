# Report: Notebook 02 — Baseline Models (TF-IDF + SVM)

**Project:** Fundamentals of Natural Language Processing — ICD-10 Clinical Code Prediction  
**Students:** Phoebe Iglesias, David Redrejo & Pau Rossell  
**Notebook:** `02_baseline_models.ipynb`

---

## 1. Objective

This notebook implements the **first non-Deep-Learning baseline** for the ASHO-AI ICD-10 codification task. It instantiates the modeling strategy proposed in notebook 01 (EDA) by building a complete **TF-IDF + Linear SVM pipeline** that predicts the ICD code **category** (first character of the ICD code) from a short clinical literal.

---

## 2. Pipeline Overview

The pipeline consists of four phases:

1. **Preprocessing (Normalization)** — lowercase, strip accents (á→a, ñ→n), remove punctuation/digits, collapse whitespace.
2. **Feature Extraction (TF-IDF)** — character n-grams with TF-IDF weighting.
3. **Classification (Linear SVM)** — multi-class single-label classification using `LinearSVC`.
4. **Prediction** — generate a leaderboard submission with columns `id`, `Literal`, `y_category`.

---

## 3. Data Loading and Preparation

### 3.1 Raw Data

| Metric | Value |
|---|---:|
| Training rows | 13 700 |
| Unique ICD codes | 4 059 |
| Unique literals | 11 584 |
| Leaderboard literals | 6 667 |

### 3.2 Normalization

Text normalization is performed using a shared `normalize_text()` function from `src/data_processing.py`. The function applies:
- Lowercasing
- Accent stripping (NFD unicode decomposition)
- Punctuation removal (via regex)
- Whitespace trimming and collapsing

**Examples shown in the notebook:**

| Original | Normalized |
|---|---|
| `Hiperreactividad bronquial` | `hiperreactividad bronquial` |
| `broncoespástica` | `broncoespastica` |
| `miocardiopatía dilatada` | `miocardiopatia dilatada` |
| `ALERGIA IBUPROFENO` | `alergia ibuprofeno` |
| `<font>VHC</font>` | `vhc` |
| `Hèrnia ventral` | `hernia ventral` |

> **Note:** The notebook documentation states that digits are removed, but the `normalize_text()` function actually *preserves* digits. This documentation mismatch is later identified and corrected in notebook 03.

### 3.3 Category Extraction

The target label is the **first character** of the ICD code. For literals that map to multiple codes with different first characters, a **majority-vote** category is selected — collapsing the dataset to one label per unique literal.

- After majority-vote deduplication: **11 584 unique literals** across **36 categories** (A–Z + 0–9).
- A stratified 80/20 train/validation split is used: **9 267 training** / **2 317 validation** samples.

### 3.4 Category Distribution

The label distribution is heavily imbalanced. Top categories include:
- `Z`: 1 561 samples
- `O`: 1 169 samples
- `0`: 967 samples

While tail categories include:
- `W`: 7 samples
- `X`: 9 samples
- `U`: 12 samples

---

## 4. Model Configuration

### 4.1 TF-IDF Vectorizer

```python
TfidfVectorizer(
    analyzer='char_wb',       # character n-grams respecting word boundaries
    ngram_range=(3, 6),       # trigrams to 6-grams
    sublinear_tf=True,        # log-normalized TF
    max_features=100_000,     # top features cap
    min_df=2,                 # ignore very rare n-grams
    dtype=np.float32,
)
```

- **Rationale:** Character n-grams `(3, 6)` capture typos, abbreviations, and morphological roots in short Spanish medical texts.
- **Result:** Vocabulary size of **25 667** features; feature matrix shape `(9267, 25667)`.

### 4.2 Linear SVM Classifier

```python
LinearSVC(
    C=1.0,
    max_iter=10_000,
    class_weight='balanced',
    random_state=42,
)
```

- **Rationale:** Linear SVMs handle high-dimensional sparse TF-IDF features very well. `LinearSVC` performs multi-class classification natively via one-vs-rest.
- `class_weight='balanced'` is used to give extra importance to rare classes.
- Training completed in **0.7 seconds**; **36 classes** learned.

---

## 5. Results

### 5.1 Validation Metrics

| Metric | Value |
|---|---:|
| **Accuracy** | **0.5693** |
| Weighted F1 | 0.5658 |
| Macro F1 | 0.5091 |
| Weighted Precision | 0.5759 |
| Weighted Recall | 0.5693 |

### 5.2 Per-Class Highlights

The per-class classification report reveals substantial variation:

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `Z` | 0.79 | 0.73 | 0.76 | 312 |
| `B` | 0.72 | 0.78 | 0.75 | 99 |
| `M` | 0.69 | 0.81 | 0.75 | 53 |
| `O` | 0.72 | 0.55 | 0.62 | 234 |
| `0` (numeric) | 0.71 | 0.63 | 0.67 | 193 |
| `7` (numeric) | 0.21 | 0.16 | 0.18 | 64 |
| `W` | 0.00 | 0.00 | 0.00 | 1 |
| `X` | 0.00 | 0.00 | 0.00 | 2 |

**Observations:**
- **Well-performing categories** (Z, B, M) tend to have distinctive medical terminology that character n-grams capture effectively.
- **Poorly-performing categories** (7, W, X) have very few support samples; the model simply cannot learn them reliably.
- The `class_weight='balanced'` setting forces the model to focus on rare classes, which may actually **hurt** overall strict accuracy (the official metric).

---

## 6. Leaderboard Prediction

The pipeline is applied to the 6 667 leaderboard literals (normalized, then TF-IDF transformed with the *already fitted* vectorizer):

- All 36 categories are predicted.
- No empty predictions.
- Submission saved to `submissions/svm_baseline.csv`.

---

## 7. Summary

| Step | Detail |
|---|---|
| **Preprocessing** | Lowercased, stripped accents, removed punctuation, collapsed whitespace |
| **Features** | Character n-grams (3–6) with TF-IDF, 100k max features, `min_df=2` |
| **Model** | `LinearSVC`, `class_weight='balanced'`, multi-class one-vs-rest |
| **Target** | ICD code category (first character) — single-label |
| **Evaluation** | Strict accuracy (exact match on first character) |
| **Validation Accuracy** | **0.5693** |
| **Output** | `submissions/svm_baseline.csv` |

---

## 8. Strengths and Limitations

### Strengths
- Establishes a **solid and reproducible** non-DL baseline.
- Uses a **shared code module** (`src/data_processing.py`, `src/evaluation.py`) for normalization, category extraction, metrics, and submission generation — promoting consistency across notebooks.
- Character n-grams are well-suited for the short, noisy, multilingual clinical literals observed in the EDA.

### Limitations Identified (addressed in Notebook 03)
1. `class_weight='balanced'` may hurt strict accuracy by over-weighting rare categories.
2. `min_df=2` discards rare n-grams that could be highly informative in medical text (e.g., unique abbreviations).
3. Starting character n-grams at length 3 misses short signals (e.g., `HTA`, `VHC`, two-letter medical fragments).
4. A minor documentation mismatch: the notebook says digits are removed, but `normalize_text()` preserves them.
5. Majority-vote label assignment for ambiguous literals hides real medical ambiguity.

These limitations serve as the starting point for the systematic improvement work in notebook 03.

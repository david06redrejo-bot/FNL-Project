# Data Exploration Report — ICD Coding Dataset

> **Notebook**: `notebooks/01_data_exploration.ipynb`
> **Date**: March 30, 2026
> **Dataset**: Spanish clinical text → ICD-10 code assignment (CodiEsp-style)

---

## 1. Datasets Overview

Three CSV files are used in this project:

| Dataset | File | Role | Columns |
|---|---|---|---|
| **Codification** | `codification_data.csv` | Training set (labeled) | `Code`, `Literal` |
| **ICD Dictionary** | `icd_d_p_pairs.csv` | Full ICD-10 reference | `Code`, `D_P`, `Description` |
| **Leaderboard** | `leaderboard_data.csv` | Unlabeled test set | `id`, `Literal` |

**Sample rows from training set (`codif_df`):**

| Code | Literal |
|---|---|
| J9809 | Hiperreactividad bronquial |
| I420 | miocardiopatía dilatada |

**Sample rows from ICD dictionary (`icd_df`):**

| Code | D_P | Description |
|---|---|---|
| A00 | D | Cólera |
| A000 | D | Cólera debido a Vibrio cholerae 01, biotipo cholerae |

**Sample rows from leaderboard test set (`lead_df`):**

| id | Literal |
|---|---|
| 1 | AMNIODRENAJE |
| 2 | Hiperparatiroidismo primario |
| 3 | MIGRANYA parto |
| 4 | VHC |
| 5 | Absceso mama izq |

---

## 2. Basic Dataset Statistics

### 2.1 Codification (Training) Data

| Statistic | Value |
|---|---|
| **Total training samples** | 13,700 |
| **Unique ICD codes** | 4,059 |
| **Unique literals** | 11,584 |
| **Missing values** | 0 (Code and Literal both complete) |
| **Most frequent code** | Z6740 (appears 148 times) |
| **Most frequent literal** | "obesidad parto" (appears 14 times) |

### 2.2 ICD Dictionary

| Statistic | Value |
|---|---|
| **Total entries** | 179,742 |
| **All codes unique** | Yes (179,742 unique) |
| **Diagnosis codes (D)** | 101,246 |
| **Procedure codes (P)** | 78,496 |

### 2.3 Leaderboard (Test) Data

| Statistic | Value |
|---|---|
| **Total test samples** | 6,667 |
| **First literal** | "AMNIODRENAJE" |
| **Last literal** | "vancomicina" |

---

## 3. Label Distribution Analysis

> **Key question**: Are some ICD codes much more frequent than others?

### Top 5 Most Frequent Codes

| Code | Count | Example Literal |
|---|---|---|
| Z6740 | 148 | Grupo sanguíneo materno: 0, Rh: Positivo |
| Z6710 | 93 | Grupo A positivo |
| Z3A40 | 87 | SG:40+2 |
| O99284 | 84 | intolerancia Lactosa parto |
| Z886 | 71 | ALERGIA IBUPROFENO |

### Bottom 5 Least Frequent Codes

| Code | Count | Example Literal |
|---|---|---|
| K2980 | 1 | DUODENITIS |
| 0SBC0ZZ | 1 | meniscectomía rodilla derecha |
| 2395 | 1 | Tumor renal derecho |
| 5952 | 1 | cistitis cronica |
| L42 | 1 | Pitiriasis rosada |

### Long-Tail Distribution Summary

| Metric | Value |
|---|---|
| Codes appearing **only once** | **1,915** (47.2% of all 4,059 unique codes) |
| Codes appearing **≤ 5 times** | **3,498** (86.2% of unique codes) |
| Codes appearing **> 50 times** | **7** (only 0.17% of unique codes) |

> ⚠️ **This is a severe long-tail / Zipf distribution.** Nearly half of all unique codes appear only once in the training data, making supervised learning for rare codes extremely difficult.

---

## 4. Text / Literal Analysis

> **Key question**: How long are the clinical text literals?

### Length Statistics (13,700 samples)

| Statistic | Character Length | Word Count |
|---|---|---|
| **Mean** | 16.95 chars | **2.21 words** |
| **Std** | 8.22 chars | 1.01 words |
| **Min** | 2 chars | 1 word |
| **25th percentile** | 11 chars | 2 words |
| **Median (50%)** | 16 chars | **2 words** |
| **75th percentile** | 22 chars | 3 words |
| **Max** | 63 chars | 9 words |

> **Key finding**: The literals are extremely short — the **median is just 2 words**. This is fundamentally different from the English MIMIC dataset (average 1,100–1,500 tokens per document). This short-text nature favors traditional ML methods (TF-IDF, Naive Bayes) over deep learning.

---

## 5. Annotation Format Analysis

> **Key question**: How are the ICD codes structured?

### Code Length Distribution

| Code Length (chars) | Count | Example Codes |
|---|---|---|
| 3 | 427 | R21, J90 |
| 4 | 5,507 | Z886, E119 |
| 5 | 4,198 | Z3A35, 04181 |
| 6 | 1,212 | G43909, N83202 |
| 7 | 2,356 | 3E033VJ, 0HBT0ZZ |

ICD-10 codes follow hierarchical patterns:
- **3-character codes** (e.g. `R21`) → Chapter/Category level
- **4–5 character codes** → Subcategory level
- **6–7 character codes** → Procedure codes (ICD-10-PCS style, e.g. `3E033VJ` for "Oxitocina")

### Diagnosis vs. Procedure in Training Set

| Type | Count | % of matched |
|---|---|---|
| Diagnosis (D) | 7,932 | 80% |
| Procedure (P) | 2,011 | 20% |
| **Unmatched codes** (not in ICD dict) | **3,757** | — |

> ⚠️ **3,757 training entries (27.4%) have codes not present in the ICD dictionary.** This likely means the training set uses a slightly different or older ICD version, or contains local/regional code variants.

---

## 6. Vocabulary & Special Characters

### Vocabulary Overview

| Metric | Value |
|---|---|
| **Total unique tokens** | 5,477 |
| **Literals in ALL CAPS** | 1,620 (**11.8%**) |
| **Short tokens (≤ 3 chars)** | 661 |

### Top 30 Most Frequent Words

| Word | Count | Notes |
|---|---|---|
| de | 1,083 | Preposition (stopword) |
| parto | 899 | "birth/delivery" — clinical context |
| part | 487 | Catalan truncation of "parto" |
| a | 331 | Preposition (stopword) |
| derecha | 263 | "right" (anatomical direction) |
| mama | 256 | "breast" |
| izquierda | 247 | "left" (anatomical direction) |
| bilateral | 205 | Clinical term |
| historia | 190 | "history" |
| alergia | 186 | "allergy" |
| renal | 183 | Organ/system |
| grupo | 168 | "group" (as in blood group) |
| positivo | 150 | "positive" |
| crónica | 150 | "chronic" |
| desgarro | 142 | "tear/laceration" |
| anemia | 134 | Clinical diagnosis |
| hipotiroidismo | 122 | Clinical diagnosis |
| semanas | 118 | "weeks" (gestational age) |
| aguda | 112 | "acute" |
| hernia | 109 | Clinical diagnosis |
| hta | 107 | Abbreviation (Hypertension) |
| carcinoma | 106 | Clinical diagnosis |
| rh | 105 | Blood group Rh factor |
| síndrome | 103 | "syndrome" |
| ca | 103 | Abbreviation (Carcinoma/Cancer) |
| obesidad | 101 | "obesity" |
| sanguíneo | 100 | "blood" (as in blood type) |

### Key Abbreviations Found (≤ 3 chars)

`hta`, `irc`, `rao`, `gs`, `rh`, `rx`, `itu`, `cin`, `vit`, `lma`, `cov`, `lap`, `fx`, `izq`, `geu`, `rmn`

> **Key findings**:
> - **Mixed language**: Catalan words appear alongside Spanish (e.g., "part" vs "parto", "MIGRANYA" vs "MIGRAÑA").
> - **Heavy abbreviation use**: 661 unique tokens are ≤ 3 characters. Medical abbreviations like `hta` (hypertension), `irc` (chronic renal failure), `itu` (urinary tract infection) are common.
> - **11.8% ALL CAPS literals**: Inconsistent casing is prevalent — normalization is essential in preprocessing.

---

## 7. Multi-Label Check

> **Key question**: Does a single clinical literal map to multiple ICD codes?

| Metric | Value |
|---|---|
| **Literals mapped to multiple codes** | **1,668** out of 11,584 unique literals (14.4%) |
| **Maximum codes for one literal** | **14** |

### Examples of Multi-Label Literals

| Literal | Number of Codes | Sample Codes |
|---|---|---|
| "obesidad parto" | 14 | Z6835, Z6834, Z3A36, Z6838, Z6831, ... |
| "Fumadora part" | 13 | K226, N200, Z6835, 64901, J45909, ... |
| "hipotiroidismo part" | 13 | 64811, O9962, E669, G43909, O99354, ... |
| "Asma parto" | 12 | J9811, Z6833, 64891, I340, Q767, ... |
| "Hipotiroidismo parto" | 12 | E039, 2449, E282, 64811, E038, ... |

> **Key finding**: 14.4% of unique literals are genuinely multi-label. Many of the most ambiguous cases involve obstetric contexts ("parto" = delivery), where a single clinical description can correspond to multiple condition + procedure codes simultaneously.

---

## 8. Challenges Summary (completing instruction §9)

Based on the full analysis, here is the completed **Summary Table**:

| Metric | Value |
|---|---|
| Training samples | **13,700** |
| Unique ICD codes | **4,059** |
| Unique literals | **11,584** |
| Test samples | **6,667** |
| Avg word count per literal | **2.21 words** |
| % literals with ≤ 3 words | **~75%** (median = 2 words, 75th pct = 3 words) |
| % codes appearing only once | **47.2%** (1,915 / 4,059) |
| % codes appearing > 50 times | **0.17%** (7 / 4,059) |
| Diagnosis (D) codes ratio | **80%** (of matched codes) |
| Procedure (P) codes ratio | **20%** (of matched codes) |

---

## 9. Conclusions & Task Challenges (completing instruction §10)

From the analysis, we can clearly identify and quantify the following challenges:

### 1. 🏷️ Large Label Space
- **4,059 unique ICD codes** in training; **179,742** in the full ICD dictionary.
- Our effective label space is much larger than the MIMIC-III-50 benchmark (50 codes) but smaller than MIMIC-III-Full (8,922 codes).
- **Implication for modeling**: We must decide whether to predict all 4,059 codes or subset to the most frequent ones.

### 2. 📊 Severe Unbalanced Distribution (Long-Tail)
- **47.2% of codes appear only once** in the training data.
- **86.2% of codes appear 5 times or fewer**.
- Only **7 codes** appear more than 50 times.
- **Implication**: Standard classifiers will heavily favor the few frequent codes. Macro-F1 will be low even with decent Micro-F1. Filtering to a top-N subset is essential for a viable baseline.

### 3. 📝 Very Short Text (Specific to Our Dataset)
- The **median literal has only 2 words**; the max is 9 words.
- This is the opposite of the long-document challenge described in the survey.
- **Implication (Advantage)**: TF-IDF and Naive Bayes work very well on short text. Deep learning models (which excel at long-document representation) are less necessary here.

### 4. 🌐 Spanish + Mixed-Language Clinical Text
- The dataset mixes **Spanish and Catalan** (e.g. "parto" / "part", "MIGRAÑA" / "MIGRANYA").
- Heavy use of **medical abbreviations** (HTA, IRC, ITU, RH, etc.).
- **11.8% of literals are ALL CAPS** — casing normalization is essential.
- **Implication**: Character-level N-gram TF-IDF will help handle abbreviations and language variants without needing a full medical dictionary.

### 5. 🔀 Multi-label Assignment
- **14.4% of unique literals** map to multiple ICD codes.
- Maximum 14 codes for a single literal.
- **Implication**: This is a genuine multi-label classification problem requiring `OneVsRestClassifier` or similar multi-label strategies.

### 6. ⚠️ Code-Dictionary Mismatch
- **27.4% of training codes** (3,757 entries) are **not found in the ICD dictionary** (`icd_d_p_pairs.csv`).
- This suggests version differences (ICD-9 vs ICD-10 codes mixed, or local regional code extensions).
- **Implication**: We cannot rely solely on code descriptions from the dictionary for augmentation without handling these mismatches.

---

## 10. Proposed Methods for April 8th Presentation

Based on the dataset characteristics:

| Method | Rationale |
|---|---|
| **TF-IDF + FlatSVM** (Primary baseline) | Standard literature baseline (Perotte et al. 2013). Robust with sparse high-dimensional features. Interpretable. |
| **Character N-gram TF-IDF + SVM** | Handles abbreviations, ALL CAPS, mixed Catalan/Spanish without a medical dictionary. |
| **Complement Naive Bayes** | Historically strong on short texts; fast baseline. |
| **Logistic Regression** | Fast, calibrated probabilities, comparable to SVM on short text. |

**Evaluation metrics**: Micro-F1 (primary) and Macro-F1 (secondary, to expose long-tail impact).

**Label filtering strategy**: Limit to the **top-100 or top-200 most frequent codes** for the baseline, then report results on the full 4,059-code set.

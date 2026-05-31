# Literature Review: Automated ICD Coding

> **Reference paper**: Yan, C., Fu, X., Liu, X., Zhang, Y., Gao, Y., Wu, J., & Li, Q. (2022).
> *A survey of automated International Classification of Diseases coding: development, challenges, and applications.*
> Intelligent Medicine, 2, 161–173. https://doi.org/10.1016/j.imed.2022.03.003

---

## 1. Introduction

The **International Classification of Diseases (ICD)** is a WHO standard system that classifies diseases, injuries, and health conditions using hierarchical codes. Assigning ICD codes to clinical documents—a process called **ICD coding**—is central to medical billing, epidemiology, and hospital management.

Manual ICD coding is performed by professional coders who read Electronic Medical Records (EMRs) and assign one or more codes from a catalogue of tens of thousands. This process is:
- **Error-prone**: The US CMS estimated a 6.8% error payout rate in 2000.
- **Costly**: Incorrect ICD-9-CM coding costs ~$25 billion/year to correct (US estimate).
- **Time-consuming and expertise-dependent**: Coders need continuous training as ICD versions expand (ICD-9 had ~13,500 codes; ICD-10 expanded this to **70,000+** diagnostic and **72,000** procedure codes).

Automated ICD coding is thus a high-value NLP problem.

---

## 2. ICD Code Structure and Characteristics

ICD codes form a **hierarchical taxonomy** (chapters → sections → categories). Nodes in this hierarchy exhibit three types of relationships:

| Relationship | Description | Example |
|---|---|---|
| **Inheritance** (parent→child) | Child codes are semantic refinements of parents | E11 (T2 diabetes) → E11.3 (T2 DM + ocular complications) → E11.302 (T2 diabetic cataract) |
| **Mutual exclusion** (siblings) | Sibling codes classified by the same axis are typically mutually exclusive | ICD codes for "with complications" vs "without complications" — both shouldn't coexist on one record |
| **Co-occurrence** (friend nodes) | Codes from different branches frequently appear together due to disease comorbidities | Acute myocardial infarction often co-occurs with heart failure codes |

These relationships distinguish ICD coding from standard multi-label classification: codes are **not independent labels**.

---

## 3. Developmental Stages of Automated ICD Coding

### Stage 1 — Rule-Based Methods (1990s–2000s)
Early methods manually encoded ICD coding guidelines as **if-else programs**. They identified symptoms and signs from text and mapped them to codes via rule sets.

- **Strengths**: Transparent, interpretable.
- **Weaknesses**: Fragile to code volume; rules conflict as code sets grow; no generalization beyond rule coverage.
- **Best result**: Farkas & Szarvas (2008) achieved 90.26% on the CCHMC dataset (only 45 ICD codes).

### Stage 2 — Traditional Machine Learning (2007–2016) ← **Our Focus for April 8th**

With ML, the focus shifted to **feature engineering + classic classifiers**. The key methods are:

| Paper | Model | Features | Dataset |
|---|---|---|---|
| Larkey & Croft (1996) | K-NN, Naive Bayes, Relevance Feedback | Terms and phrases | — |
| Suominen et al. (2007) | RLS + RIPPER cascade | Word segmentation, medical concepts, hypernyms | CCHMC (87.7% Micro-F1) |
| **Perotte et al. (2013)** | **FlatSVM + Hierarchy-SVM** | **TF-IDF** | MIMIC-III-50 (Micro-F1: 0.211 / 0.293) |
| **Marafiño et al. (2014)** | **Binary SVM per disease** | **N-gram features** | ICU text (4 diseases) |
| **Elyne et al. (2016)** | **Naive Bayes + Random Forests** | **BoW + TF-IDF + demographics + lab results** | UZA (Dutch EMRs) |

These methods train one classifier per ICD code (binary: "does this text have code X?"), then combine all predictions. They work well for small code sets but scale poorly to thousands of codes.

**Key performance on MIMIC-III-50** (from the survey's Table 3):
- FlatSVM: Micro-F1 = 0.211
- Hierarchy-based SVM: Micro-F1 = 0.293

### Stage 3 — Deep Learning / Neural Networks (2017–present)
Neural methods became dominant with the publication of **CAML/DR-CAML** (Mullenbach et al., 2018):
- CNN + per-label attention over discharge summaries
- MIMIC-II Micro-F1: **0.457**; MIMIC-III-50 Micro-F1: **0.633**

Subsequent models (GNNs, BERT variants, multi-scale CNNs) achieved further improvements, with best known Micro-F1 on MIMIC-III-50 of **0.725** (Fusion, Luo et al. 2021). These methods are **out of scope for our project** (non-DL constraint), but provide upper bounds to compare against.

---

## 4. Key Challenges

From the survey (Section 6), four major challenges affect all automated ICD coding systems:

### 4.1 Large Label Space
ICD-10 contains 70,000+ codes. This creates a massive search space for classifiers. Most practical systems either limit to frequent codes (MIMIC-III-50) or use label embeddings / hierarchical methods to constrain the search.

### 4.2 Unbalanced Label Distribution (Long-Tail)
> "In the MIMIC-III dataset, 10% of ICD codes appear in 85% of the data, whereas 22% of codes appear fewer than two times."

This Zipf/power-law distribution means models trained on raw data will heavily favour frequent codes. Macro-F1 scores are systematically much lower than Micro-F1 scores as a result.

### 4.3 Long Document Representation
EMRs average 1,100–5,300 tokens. Extracting the relevant diagnostic passages from this long, noisy text is a major challenge. PLMs (BERT) are limited to 512 tokens and must use chunking strategies.

> **Note**: Our dataset uses very short clinical phrases (1–5 words), which mitigates this challenge.

### 4.4 Interpretability
Clinical applications require that predicted codes are **explainable**. Attention mechanisms in deep models provide partial interpretability. Rule-based and TF-IDF-based models are inherently more interpretable.

---

## 5. Proposed Methods for Our Project

Based on the literature analysis and our dataset characteristics (Spanish, short clinical phrases, ~3,000+ unique codes):

### Primary Baseline: TF-IDF + FlatSVM (Perotte et al., 2013)
- **Why**: Standard baseline reported in all ICD coding papers. Interpretable. Works well with sparse high-dimensional features.
- **Expected performance**: Micro-F1 in range 0.20–0.35 on full label set. Higher on top-50 most frequent codes.
- **Implementation**: `TfidfVectorizer` + `OneVsRestClassifier(LinearSVC())` in scikit-learn.

### Secondary Baseline: N-gram TF-IDF + SVM (Marafiño et al., 2014)
- **Why**: Character-level N-grams handle abbreviations, mixed language (Spanish/Catalan), and morphological variants abundant in our short clinical phrases.
- **Expected performance**: Comparable or slightly better than pure word TF-IDF.
- **Implementation**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))` combined with word TF-IDF.

### Additional Baseline: Complement Naive Bayes (Elyne et al., 2016)
- **Why**: Historically strong on short text; fast training; useful for comparison.
- **Implementation**: `OneVsRestClassifier(ComplementNB())` in scikit-learn.

---

## 6. Datasets

The survey describes 17 commonly used datasets. Our dataset aligns with **CodiEsp / CLEF eHealth 2020** (CLEF-Spanish):

| Property | Our Data | CLEF-Spanish (CodiEsp) |
|---|---|---|
| Language | Spanish | Spanish |
| Document type | Clinical text literals | EMRs |
| ICD version | ICD-10 | Spanish ICD-10-CM + ICD-10-PCS |
| Total codes | ~3,427+ | ~3,427 |
| Training samples | ~666,800 pairs | 1,000 EMRs (~18.4 labels/doc) |

---

## 7. Evaluation Metrics

Following the survey and the CLEF eHealth benchmark:

- **Micro-F1**: Primary metric. Controls for label frequency — benefits frequent codes.
- **Macro-F1**: Secondary metric. Averages per label — reveals performance on rare codes.
- **Precision@k (P@k)**: Report for k=5, k=8 if applicable.

---

## 8. References

1. Perotte, A., Pivovarov, R., Natarajan, K., et al. (2013). *Diagnosis code assignment: models and evaluation metrics.* JAMIA, 21(2), 231–237.
2. Marafiño, B.J., Davies, J.M., Bardach, N.S., et al. (2014). *N-Gram support vector machines for scalable procedure and diagnosis classification.* JAMIA, 21(5), 871–875.
3. Scheurwegs, E., Luyckx, K., Luyten, L., et al. (2016). *Data integration of structured and unstructured sources for assigning clinical codes to patient stays.* JAMIA, 23, e11–e19.
4. Kavuluru, R., Rios, A., & Lu, Y. (2015). *An empirical evaluation of supervised learning approaches in assigning diagnosis codes to EMRs.* Artif. Intell. Med., 65(2), 155–166.
5. Mullenbach, J., Wiegreffe, S., Duke, J., et al. (2018). *Explainable prediction of medical codes from clinical text.* NAACL-HLT, 1101–1111.
6. Yan, C., Fu, X., Liu, X., et al. (2022). *A survey of automated ICD coding.* Intelligent Medicine, 2, 161–173.

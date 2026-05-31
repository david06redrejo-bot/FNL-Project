# ASHO-AI ICD-10 Codification: Proposed Solution

## 1. Project Context
The goal is to map Spanish medical text literals (short clinical notes) to their corresponding ICD-10 code categories. The problem is framed as a supervised classification task using historical medical records.

## 2. Data Insights & Challenges
- **Short Inputs:** Texts are extremely short (average ~2 words). This is a label-matching/normalization problem, not long-text comprehension.
- **Ambiguity:** High synonymy and ambiguity. A single literal can map to multiple codes, and a single code can be described by thousands of different literals.
- **Low Overlap:** There is minimal lexical overlap between the training literals and the official ICD descriptions. Pure exact matching or dictionary lookups are insufficient.

## 3. Proposed Solution: TF-IDF + SVM
We propose a traditional machine learning pipeline. It is efficient, interpretable, and highly effective for short, noisy clinical texts containing abbreviations and typos.

### Pipeline Steps:
1. **Preprocessing (Normalization):** Clean the literals. Lowercase text, strip accents, and handle punctuation and digits systematically.
2. **Feature Extraction (TF-IDF):** Split the normalized literals into character and word n-grams. Apply Term Frequency-Inverse Document Frequency (TF-IDF) to convert textual features into sparse numerical vectors.
3. **Classification (SVM):** Train a Support Vector Machine (SVM) to map the TF-IDF feature space to the corresponding ICD-10 codes.
4. **Prediction:** Apply the fitted vectorizer and SVM model to the leaderboard data to generate the final submissions.
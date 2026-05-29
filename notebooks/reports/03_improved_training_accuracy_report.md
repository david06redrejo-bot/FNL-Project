# Report: Notebook 03 — Improved Training Accuracy

**Project:** Fundamentals of Natural Language Processing — ICD-10 Clinical Code Prediction  
**Students:** Phoebe Iglesias, David Redrejo & Pau Rossell  
**Notebook:** `03_improved_training_accuracy.ipynb`

---

## 1. Objective

This notebook is a **careful rebuild** of the baseline from notebook 02. The goal is not merely to increase a performance number, but to **understand why the baseline behaves the way it does**, identify weak spots, and improve the model in a principled manner that remains consistent with the non-Deep-Learning project direction.

The task remains the same: predict the first character (category) of an ICD code from a short clinical literal.

---

## 2. Diagnosis: What Was Wrong in Notebook 02?

The notebook opens with a systematic critique of the baseline's design choices:

| Issue | Explanation |
|---|---|
| **`class_weight='balanced'`** | Gives extra importance to rare classes, which can improve fairness across classes but hurts overall strict accuracy — the dominant metric for this task. |
| **`min_df=2`** | Discards n-grams that appear only once. In medical text, rare fragments (unique abbreviations, disease-specific substrings) are often the most informative features. |
| **N-gram range `(3, 6)`** | Misses short signals. Inputs like `HTA`, `VHC`, `IRC`, or two-letter medical fragments carry strong category signal but cannot be captured by trigrams or longer. Shifting to `(2, 5)` provides more small building blocks. |
| **Majority-vote label** | Necessary for single-label classification, but hides genuine medical ambiguity — 1 486 literals map to multiple categories in the raw data. |
| **Documentation mismatch** | The notebook says punctuation and digits are removed, but `normalize_text()` preserves digits (which is actually beneficial). |

---

## 3. Literature Grounding

The notebook cites recent research to justify the chosen direction:

- **Dong et al.** (Nature Digital Medicine, 2022): Why automated clinical coding needs both text modeling and knowledge from the coding system itself.
- **BoW/TF-IDF competitiveness** (BMC Medical Informatics, 2022): For ICD coding with frequent and infrequent codes, bag-of-words with traditional classifiers remains very competitive — especially on the Spanish CodiEsp dataset.
- **ICD coding reviews** (PubMed, 2024): Adding code descriptions, code hierarchy, and external knowledge is a common improvement strategy.
- **AnEMIC** (PMC, 2024): Preprocessing consistency and explainability are essential for reliable ICD-coding workflows.

**Strategic conclusion:** Keep TF-IDF + Linear SVM, but tune feature extraction for short Spanish/Catalan clinical literals and test ICD-description augmentation as extra knowledge.

---

## 4. Experimental Plan

1. Load data and inspect the task.
2. Recreate the baseline settings from notebook 02 (as a control).
3. Test improved TF-IDF/SVM settings that should raise training accuracy.
4. Visualize why the improved model works better and where it still fails.
5. Train the selected model on all data and create a new submission.

---

## 5. Dataset Preparation

| Metric | Value |
|---|---:|
| Training rows | 13 700 |
| Unique literals | 11 584 |
| Unique codes | 4 059 |
| Leaderboard literals | 6 667 |
| ICD descriptions | 179 742 |
| Categories | 36 |
| Ambiguous literals (multiple categories) | 1 486 |

An explicit **label-imbalance visualization** (bar chart) is provided, reinforcing that the categories are not balanced — strict accuracy rewards the classes that appear often. Forcing equal attention to all classes (as `class_weight='balanced'` does) may lower total accuracy.

---

## 6. Model Comparison: Results

Four model configurations are trained and evaluated on a stratified 80/20 train/validation split:

| Model | Train Accuracy | Validation Accuracy |
|---|---:|---:|
| **Notebook 02 baseline** (`char(3,6)`, `C=1`, balanced) | 0.8286 | 0.5701 |
| **Improved char `(2,5)`**, `C=2`, no balanced weights | **0.8680** | **0.6034** |
| Higher-capacity `char(2,6)`, `C=5` | 0.8814 | 0.5982 |
| Word + char high-capacity model (`FeatureUnion`) | 0.9162 | 0.5883 |

### Key Observations

- The **improved character model** achieves the best validation accuracy (**0.6034**) while also raising training accuracy. It improves both memorization and generalization simultaneously.
- The **highest training accuracy** model (word + char, 0.9162) actually has *lower* validation accuracy — a classic sign of overfitting.
- The **selected model** is the improved character model: it offers the best balance between training and validation performance.

A **horizontal bar chart** comparing training vs. validation accuracy across all four models provides visual confirmation.

---

## 7. Optional: ICD Description Augmentation

An optional experiment augments the training data with official ICD descriptions from `icd_d_p_pairs.csv`:

- ICD descriptions are normalized and appended to the training set (with the original training data duplicated to preserve its weight).
- The augmented model achieves **~0.619 validation accuracy** — a further improvement over the selected model.
- However, training accuracy *on original literals* is slightly lower, meaning the model generalizes better but memorizes the specific training literals less.

This experiment is gated behind a flag (`RUN_ICD_AUGMENTATION = False` by default) because it is slower and should be validated with cross-validation before adoption.

---

## 8. Error Analysis and Explainability

### 8.1 Confusion Matrix

A full 36×36 confusion matrix is plotted for the selected improved model. **Dark diagonal cells indicate correct predictions.** Key patterns:

- Categories with strong, distinctive medical vocabulary (e.g., `Z`, `O`, `B`) are well-separated.
- Numeric categories (procedure codes `0`–`9`) show considerable cross-confusion, which makes clinical sense since procedure terms share vocabulary.
- Very rare categories (`W`, `X`, `A`, `U`) remain problematic due to insufficient training data.

### 8.2 Top TF-IDF Features per Category

For five representative categories (`O`, `Z`, `J`, `I`, `N`), the notebook extracts the **top-weighted character n-gram features** from the trained SVM coefficients.

**Examples:**

| Category | Top Features (selected) | Interpretation |
|---|---|---|
| **O** (Pregnancy/childbirth) | `part`, `parto`, `rto`, `embara` | Captures "parto" (delivery) and "embarazo" (pregnancy) |
| **Z** (Health status/services) | `grup`, `posi`, `nega`, `sang` | Captures "grupo sanguíneo" (blood type), "positivo/negativo" |
| **J** (Respiratory) | `asm`, `asma`, `bron`, `pne` | Captures "asma" (asthma), "bronquitis", "neumonía" |
| **I** (Circulatory) | `hta`, `art`, `card`, `fibr` | Captures "HTA" (hypertension), "cardio-", "fibrilación" |
| **N** (Genitourinary) | `ren`, `ir`, `u`, `stopa` | Captures "renal", "IRC" (chronic renal failure), "uro-" |

This feature inspection confirms the model is learning **medically meaningful character patterns**, not spurious correlations.

---

## 9. Final Model and Submission

### Selected Configuration

```python
TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    sublinear_tf=True,
    max_features=200_000,
    min_df=1,
    dtype=np.float32,
)
LinearSVC(
    C=2.0,
    class_weight=None,  # removed 'balanced'
    max_iter=10_000,
    random_state=42,
)
```

### Why This Model?

1. Improves training accuracy over notebook 02.
2. Also improves validation accuracy — the gain is not just memorization.
3. Still simple, fast, explainable, and consistent with the non-DL project direction.

### Final Training Results

- Trained on **all 11 584 unique literals** (no held-out set).
- Feature matrix: `(11584, 28473)`.
- **Final training accuracy: 0.8596**.
- Submission saved to `submissions/svm_improved_training_accuracy.csv`.
- All 36 categories predicted; no empty predictions.

---

## 10. Conclusions

### What Changed and Why

| Change | Rationale | Effect |
|---|---|---|
| N-gram range `(3,6)` → `(2,5)` | Short literals need short fragments; 2-char n-grams capture abbreviations like `HTA`, `VHC` | More discriminative features for short medical terms |
| `min_df=2` → `min_df=1` | Rare medical fragments are informative; removing them loses useful signal | Better coverage of unique abbreviations |
| `class_weight='balanced'` → `None` | Strict accuracy is the target metric; balanced weights divert effort to rare classes | Higher overall accuracy |
| `C=1.0` → `C=2.0` | Slightly stronger regularization relaxation allows the model to fit the data better without overfitting | Better train/val accuracy balance |

### Performance Summary

| Metric | Baseline (NB02) | Improved (NB03) | Δ |
|---|---:|---:|---:|
| Train accuracy | 0.8286 | 0.8680 | +3.9 pp |
| Validation accuracy | 0.5701 | 0.6034 | +3.3 pp |

### Recommended Next Steps

The notebook explicitly states that the strongest next step is **not** to blindly increase model capacity. Instead:

1. **Use ICD knowledge more carefully** — descriptions, aliases, and the code hierarchy.
2. **Cross-validate** the ICD-description augmentation approach before adopting it for the official submission.
3. Consider more sophisticated methods (e.g., Deep Learning with RoBERTa) only after exhausting the interpretable pipeline improvements.

This notebook demonstrates a disciplined, evidence-based approach to model improvement: diagnose the baseline's weaknesses, ground improvements in literature, compare systematically, and verify that gains transfer to held-out data.

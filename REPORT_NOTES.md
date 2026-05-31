# Report Notes

This file accumulates observations, explanations, and writing ideas for the final report. The final report should be written in English and should sound like a student team progressively understanding the task through the course.

## 2026-05-31 — Initial Storyline After Repository Takeover

The repository already suggests a good narrative arc, but it needs to be rebuilt and verified:

1. We first understood that the task is not full-document ICD coding, but short clinical literal classification.
2. The target for the Kaggle task is exactly one prefix category, `y_category`, defined as the first character of `Code`.
3. The expected label space is 36 categories: digits `0`-`9` and letters `a`-`z` / `A`-`Z` after normalization of case.
4. The early EDA reports describe very short inputs, around two words on average, with mixed Spanish/Catalan, abbreviations, accents, digits, and inconsistent casing.
5. Existing reports claim strong label imbalance and many-to-many literal/code ambiguity. These points are central to the report, but must be rerun from the raw dataset before final claims.
6. The survey/literature-review material motivates classic TF-IDF + SVM methods as reference baselines and also explains why neural models such as RoBERTa are worth testing.
7. The final narrative should not pretend that the team knew the best method from the beginning. It should show a progression: inspect annotations, understand the challenge, build simple baselines, improve them, then compare with a RoBERTa backbone and decide based on evidence.

Important writing rule: do not fabricate leaderboard scores or validation results. Use only rerun metrics or clearly label historical notebook values as previous/reported results.

Potential report section order:

1. Introduction and task definition.
2. Dataset and annotation analysis.
3. Main challenges.
4. Related work and reference methods.
5. Preprocessing and reproducibility.
6. Baseline models.
7. Improved traditional models.
8. RoBERTa backbone experiment.
9. Evaluation and error analysis.
10. Submission and conclusions.

## 2026-05-31 — Repository Skeleton Implemented

The final project structure now reflects the story we want the report to tell:
we start with the data and annotations, then explain the main task challenges,
then move from survey-inspired baselines to RoBERTa and improved models.

The report should explicitly say that notebooks are narrative and scripts are
the reproducibility backbone. This is a good way to show that the team learned
to separate exploration from reusable implementation.

## 1. Analyzing the Data and the Annotations

We inspected all CSV files under `data/` and validated them against the competition annotation contract.

| File | Role | Shape | Schema status |
|---|---|---:|---|
| `codification_data.csv` | `codification` | 13700 x 2 | valid |
| `icd_d_p_pairs.csv` | `icd_catalog` | 179742 x 3 | valid |
| `leaderboard_data.csv` | `leaderboard` | 6667 x 2 | valid |

The training labels are derived as `y_category = Code.astype(str).str[0]`.
The observed label set has 36 classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.

Warnings:
- `y_category` is absent from leaderboard data; this is normal for unlabeled Kaggle test files but differs from the stated expected columns.

## Visual EDA Conclusions

We continued the first phase with a visual EDA inspired by a Data Engineering
mindset: before modeling, we inspected the data-generating process, data
quality, distributions, missingness, duplicates, and possible leakage.

Core size and annotation facts:

- Training rows: 13700
- Leaderboard rows: 6667
- Unique full ICD codes in training: 4059
- Unique `y_category` labels: 36
- Unique full ICD codes in the ICD catalog: 179742

The top 10 categories by row count are: [{'y_category': 'Z', 'count': 1715}, {'y_category': 'O', 'count': 1505}, {'y_category': '0', 'count': 1141}, {'y_category': '6', 'count': 637}, {'y_category': '3', 'count': 592}, {'y_category': 'B', 'count': 579}, {'y_category': 'N', 'count': 536}, {'y_category': 'E', 'count': 500}, {'y_category': 'V', 'count': 491}, {'y_category': '5', 'count': 408}].
The bottom 10 categories by row count are: [{'y_category': 'W', 'count': 7}, {'y_category': 'X', 'count': 10}, {'y_category': 'A', 'count': 22}, {'y_category': 'U', 'count': 22}, {'y_category': 'Y', 'count': 36}, {'y_category': 'T', 'count': 70}, {'y_category': 'S', 'count': 71}, {'y_category': 'P', 'count': 101}, {'y_category': 'L', 'count': 110}, {'y_category': 'Q', 'count': 145}].

Duplicate and ambiguity findings:

- Duplicate literals: 1668
- Same literal with multiple full ICD codes: 1668
- Same literal with multiple `y_category` labels: 1486

Train/leaderboard comparison:

- 51.8% of unique normalized leaderboard
  literals appear in the training set.
- Mean character length is 16.95 in train and
  17.15 in leaderboard.

Normalization risk:

- Raw unique literals: 11584
- Unique literals after lowercase/accent-stripped normalization:
  7844
- Normalization collision keys:
  2186

No model has been trained yet; these are EDA conclusions only.

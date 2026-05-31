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

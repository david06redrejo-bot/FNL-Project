# Final Model Card: v09 Ensemble

## Model Identity

- **Project:** UAB-ASHO AI Codification
- **Course:** Fundamentals of Natural Language / NLP-I, Universitat Autonoma de Barcelona, 2025-2026
- **Team:** Team 10: Phoebe Iglesias, David Redrejo, Pau Rossell
- **Supervisors/professors:** Ernest Valveny and Lei Kang
- **Final candidate:** `v09_ensemble`
- **Submission file:** `submissions/final_submission.csv`

## Task

Predict exactly one ICD category prefix, `y_category`, for each clinical literal.
The target is the first character of `Code`, with 36 expected categories:
digits `0`-`9` and letters `A`-`Z`.

## Model Description

The final model is a validation-selected majority-vote ensemble. It combines:

- `v04_roberta_cls`
- `v05_roberta_mean`
- `v08_roberta_mean_dedupe`
- `v08_roberta_mean_weighted_sampler`
- `v01_tfidf_char_logreg`

Ties are resolved using average probabilities. The ensemble uses saved
validation and leaderboard predictions from completed model runs. It does not
use leaderboard labels or public leaderboard feedback.

## Selection Rationale

The ensemble was selected because it improved over the best single model on the
shared validation split:

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| `v04_roberta_cls` | 0.569343 | 0.494329 | 0.554347 |
| `v09_ensemble` | 0.576642 | 0.506277 | 0.561544 |

Accuracy matters because it is the competition-oriented metric. Macro F1 also
matters because the EDA showed a long-tailed and imbalanced label distribution.

## Inputs and Outputs

- **Input:** clinical literal text from `leaderboard_data.csv`.
- **Output:** one uppercase `y_category` prefix per row.
- **Kaggle submission contract:** exactly two columns, `id` and `y_category`.

## Preprocessing

The final RoBERTa pipeline uses required light preprocessing only:

```python
text = str(text)
text = text.strip()
text = re.sub(r"\s+", " ", text)
```

It preserves case, accents, punctuation, digits, and abbreviations because the
backbone tokenizer is trained for Spanish biomedical and clinical text.

## Known Strengths

- Improves over individual RoBERTa and classical baselines on validation.
- Combines contextual RoBERTa signals with character-level TF-IDF surface
  patterns.
- More robust than relying on a single pooling strategy.

## Known Limitations

- Clinical literals are short and can lack context.
- Some literals are ambiguous or duplicated with different labels.
- Rare categories still have weaker recall.
- Abbreviations and broad ICD-prefix categories remain difficult.
- Confidence is a diagnostic signal, not a guarantee of correctness.

## Interpretability

The final analysis uses confusion matrices, per-class metrics, confidence
distributions, confidence margins, and representative validation examples.
Attention/token attribution was not used as a clinical explanation. If added in
future work, it should be treated as exploratory and limited.

## Competition Result

Final Kaggle/public/private score is **not verified in repository files**.
Evidence should be placed here before the final report is submitted:

```text
reports/figures/kaggle_final_ranking.png
reports/tables/kaggle_submission_scores.csv
```

The team narrative notes that after the early baseline, short presentation, and
follow-up with Lei Kang, the team was on the right track and at one point was
second in the competition. This should be supported by a screenshot or score
table if it is used as a formal claim in the report.

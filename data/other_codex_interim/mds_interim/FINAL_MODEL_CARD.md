# Final Model Card: v10 Diverse Ensemble

## Model Identity

- **Project:** UAB-ASHO AI Codification
- **Course:** Fundamentals of Natural Language / NLP-I, Universitat Autonoma de Barcelona, 2025-2026
- **Team:** Team 10: Phoebe Iglesias, David Redrejo, Pau Rossell
- **Supervisors/professors:** Ernest Valveny and Lei Kang
- **Final Kaggle public candidate:** `v10_vote_diverse_no_retrieval`
- **Best validation-accuracy recipe from v10 search:** `v10_vote_diverse_no_retrieval_val_weighted`
- **Best macro-F1-oriented validation candidate:** `v09_ensemble`
- **Submission file:** `submissions/final_submission.csv`

## Task

Predict exactly one ICD category prefix, `y_category`, for each clinical literal.
The target is the first character of `Code`, with 36 expected categories:
digits `0`-`9` and letters `A`-`Z`.

## Model Description

The final Kaggle public model is `v10_vote_diverse_no_retrieval`, a majority
vote over deliberately different model families:

- `v08_safe_dedupe`
- `v04_roberta_cls`
- `v01_tfidf_char_logreg`
- `v02_word_tfidf_svm`

This combines neural contextual representations with classical vector-space
models. It does not use leaderboard labels.

## Selection Rationale

The final public-leaderboard submission was selected after submitting the main
candidates to Kaggle:

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| `v04_roberta_cls` | 0.569343 | 0.494329 | 0.554347 |
| `v09_ensemble` | 0.576642 | 0.506277 | 0.561544 |
| `v08_safe_dedupe` | 0.568613 | 0.481648 | 0.549150 |
| `v10_vote_diverse_no_retrieval_val_weighted` | 0.583577 | 0.501074 | 0.564988 |
| `v10_vote_diverse_no_retrieval` | 0.579562 | 0.496677 | 0.561294 |

Kaggle public scores:

| model | public score |
|---|---:|
| `v10_vote_diverse_no_retrieval` | **0.587** |
| `v10_vote_diverse_no_retrieval_val_weighted` | 0.586 |
| `v08_safe_dedupe` | 0.583 |
| `v04_roberta_cls` | 0.573 |
| `v09_ensemble` | 0.573 |

`v10_vote_diverse_no_retrieval_val_weighted` is the strongest validation-accuracy
recipe from the diverse ensemble search. `v09_ensemble` remains useful for
macro-F1 discussion, but `v10_vote_diverse_no_retrieval` is the best verified
Kaggle public submission.

## Inputs and Outputs

- **Input:** clinical literal text from `leaderboard_data.csv`.
- **Output:** one uppercase `y_category` prefix per row.
- **Kaggle submission contract:** exactly `id` and `y_category`.

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

- Best verified public leaderboard score among submitted candidates.
- Conservative duplicate handling avoids hiding conflicting duplicate literals.
- Uses a Spanish biomedical-clinical RoBERTa backbone.

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

Best verified public score: `0.587`.

Private/final ranking is **not verified in repository files**. Evidence should
be placed here before the final report is submitted:

```text
reports/figures/kaggle_final_ranking.png
reports/tables/kaggle_submission_scores.csv
```

The team narrative notes that after the early baseline, short presentation, and
follow-up with Lei Kang, the team was on the right track and at one point was
second in the competition. This should be supported by a screenshot or score
table if it is used as a formal claim in the report.

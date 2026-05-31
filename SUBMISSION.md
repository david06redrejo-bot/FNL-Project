# Submission Instructions

## Final File

Upload this file to Kaggle:

```text
submissions/final_submission.csv
```

The file was copied from the best verified public-leaderboard candidate:

```text
submissions/v10_vote_diverse_no_retrieval_kaggle.csv
```

It contains exactly:

```text
id,y_category
```

and `6667` prediction rows.

## Final Model

The final Kaggle public candidate is `v10_vote_diverse_no_retrieval`. It
replaced `v09_ensemble` after a broader search over diverse Machine Learning
ensembles. The related `v10_vote_diverse_no_retrieval_val_weighted` recipe had
the best validation accuracy, while `v09_ensemble` remains useful for macro-F1
discussion.

Validation metrics:

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
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

## Kaggle Upload Steps

1. Open the Kaggle competition page:
   `uab-asho-ai-codification`.
2. Go to the **Submit Predictions** page.
3. Upload:
   `submissions/final_submission.csv`.
4. Add a description such as:
   `Team 10 final public-best v10 diverse ensemble`.
5. Submit and wait for Kaggle validation.
6. Record the public/private score and ranking in:
   `reports/tables/kaggle_submission_scores.csv`.
7. Add screenshot evidence, if used in the report, under:
   `reports/figures/`.

## Reproducibility Commands

Regenerate the final ensemble artifacts:

```bash
python models/v09_ensemble.py
python models/v10_diverse_ensemble_search.py
```

Regenerate final evaluation tables and figures:

```bash
python -m src.evaluation
```

Regenerate the final submission copy:

```bash
python - <<'PY'
import pandas as pd
sub = pd.read_csv('submissions/v10_vote_diverse_no_retrieval_kaggle.csv')
assert list(sub.columns) == ['id', 'Literal', 'y_category']
assert len(sub) == 6667
sub[['id', 'y_category']].to_csv('submissions/final_submission.csv', index=False)

detailed = pd.read_csv('outputs/predictions/v10_vote_diverse_no_retrieval_leaderboard_detailed.csv')
detailed.to_csv('outputs/predictions/final_leaderboard_detailed.csv', index=False)
PY
```

## Score Evidence Placeholder

Best public score is verified in `reports/tables/kaggle_submission_scores.csv`.
Do not state a final private placement in the report until the team adds private
leaderboard evidence.

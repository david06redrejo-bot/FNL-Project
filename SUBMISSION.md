# Submission Instructions

## Final File

Upload this file to Kaggle:

```text
submissions/final_submission.csv
```

The file was copied from the selected final model:

```text
submissions/v09_ensemble_submission.csv
```

It contains exactly:

```text
id,y_category
```

and `6667` prediction rows.

## Final Model

The final submission candidate is `v09_ensemble`, selected by internal
validation accuracy and supported by macro/weighted metrics and error analysis.

Validation metrics:

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| `v09_ensemble` | 0.576642 | 0.506277 | 0.561544 |

## Kaggle Upload Steps

1. Open the Kaggle competition page:
   `uab-asho-ai-codification`.
2. Go to the **Submit Predictions** page.
3. Upload:
   `submissions/final_submission.csv`.
4. Add a description such as:
   `Team 10 final validation-selected v09 ensemble`.
5. Submit and wait for Kaggle validation.
6. Record the public/private score and ranking in:
   `reports/tables/kaggle_submission_scores.csv`.
7. Add screenshot evidence, if used in the report, under:
   `reports/figures/`.

## Reproducibility Commands

Regenerate the final ensemble artifacts:

```bash
python models/v09_ensemble.py
```

Regenerate final evaluation tables and figures:

```bash
python -m src.evaluation
```

Regenerate the final submission copy:

```bash
python - <<'PY'
import pandas as pd
sub = pd.read_csv('submissions/v09_ensemble_submission.csv')
assert list(sub.columns) == ['id', 'y_category']
assert len(sub) == 6667
sub.to_csv('submissions/final_submission.csv', index=False)

detailed = pd.read_csv('outputs/predictions/v09_ensemble_leaderboard_detailed.csv')
detailed.to_csv('outputs/predictions/final_leaderboard_detailed.csv', index=False)
PY
```

## Score Evidence Placeholder

The final Kaggle result is not verified in repository files yet. Do not state a
final placement in the report until the team adds evidence.

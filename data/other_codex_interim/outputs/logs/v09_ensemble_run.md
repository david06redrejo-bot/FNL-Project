# Run Summary: v09_ensemble

## Config

```text
version_name: v09_ensemble
inputs: ['v04_roberta_cls', 'v05_roberta_mean', 'v08_roberta_mean_dedupe', 'v08_roberta_mean_weighted_sampler', 'v06_roberta_mean_class_weighted', 'v06_roberta_mean_focal_gamma2', 'v01_tfidf_char_logreg']
```

## Metrics

- `accuracy`: 0.576642
- `macro_precision`: 0.511047
- `macro_recall`: 0.518636
- `macro_f1`: 0.506277
- `weighted_f1`: 0.561544

## Artifacts

- `comparison_table`: `/home/iadlG010/FNL-Project/reports/tables/v09_ensemble_comparison.csv`
- `recipe`: `/home/iadlG010/FNL-Project/outputs/logs/v09_ensemble_recipe.md`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v09_ensemble_metrics.json`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v09_ensemble_per_class_metrics.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v09_ensemble_val_predictions.csv`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v09_ensemble_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v09_ensemble_submission.csv`

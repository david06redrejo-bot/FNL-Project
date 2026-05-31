# Run Summary: v07_roberta_mean_tuning

## Config

```text
version_name: v07_roberta_mean_tuning
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
seed: 42
validation_size: 0.2
stage: stage_a
use_amp: False
```

## Metrics

- `accuracy`: 0.564599
- `macro_precision`: 0.504834
- `macro_recall`: 0.504509
- `macro_f1`: 0.494151
- `weighted_f1`: 0.549054
- `log_loss`: 1.370536
- `top_2_accuracy`: 0.818613
- `top_3_accuracy`: 0.878832
- `top_5_accuracy`: 0.929927

## Artifacts

- `tuning_results`: `/home/iadlG010/FNL-Project/reports/tables/v07_tuning_results.csv`
- `canonical_submission`: `/home/iadlG010/FNL-Project/submissions/v07_roberta_mean_tuning_submission.csv`
- `canonical_leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v07_roberta_mean_tuning_leaderboard_detailed.csv`
- `canonical_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v07_roberta_mean_tuning_metrics.json`

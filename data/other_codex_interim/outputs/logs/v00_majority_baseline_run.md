# Run Summary: v00_majority_baseline

## Config

```text
version_name: v00_majority_baseline
model_family: majority
seed: 42
validation_size: 0.2
text_col: Literal_required_clean
target_col: y_category
label_id_col: label_id
debug_sample: None
save_model: True
top_k: 5
```

## Metrics

- `accuracy`: 0.125182
- `macro_precision`: 0.003477
- `macro_recall`: 0.027778
- `macro_f1`: 0.006181
- `weighted_f1`: 0.027854
- `log_loss`: 31.531619
- `top_2_accuracy`: 0.127737
- `top_3_accuracy`: 0.128467
- `top_5_accuracy`: 0.164599

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v00_majority_baseline_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v00_majority_baseline_metrics.json`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v00_majority_baseline_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v00_majority_baseline_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v00_majority_baseline.joblib`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v00_majority_baseline_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v00_majority_baseline_submission.csv`

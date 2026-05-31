# Run Summary: v05_roberta_mean_class_weighted_debug

## Config

```text
version_name: v05_roberta_mean_class_weighted_debug
model_family: debug_logreg
seed: 42
validation_size: 0.2
text_col: Literal_required_clean
target_col: y_category
label_id_col: label_id
debug_sample: 120
save_model: True
top_k: 5
```

## Metrics

- `accuracy`: 0.045455
- `macro_precision`: 0.033333
- `macro_recall`: 0.033333
- `macro_f1`: 0.033333
- `weighted_f1`: 0.045455
- `log_loss`: 3.969581
- `top_2_accuracy`: 0.045455
- `top_3_accuracy`: 0.045455
- `top_5_accuracy`: 0.045455

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_class_weighted_debug_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v05_roberta_mean_class_weighted_debug_metrics.json`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v05_roberta_mean_class_weighted_debug_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v05_roberta_mean_class_weighted_debug_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v05_roberta_mean_class_weighted_debug.joblib`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v05_roberta_mean_class_weighted_debug_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v05_roberta_mean_class_weighted_debug_submission.csv`

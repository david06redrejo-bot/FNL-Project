# Run Summary: v05_roberta_mean

## Config

```text
version_name: v05_roberta_mean
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
pooling: mean
seed: 42
max_length: 32
learning_rate: 2e-05
weight_decay: 0.01
batch_size: 128
max_epochs: 50
patience: 10
dropout: 0.1
validation_size: 0.2
use_amp: False
debug: False
debug_train_examples: 16
debug_val_examples: 16
debug_leaderboard_examples: 16
```

## Metrics

- `accuracy`: 0.564599
- `macro_precision`: 0.499098
- `macro_recall`: 0.511428
- `macro_f1`: 0.496567
- `weighted_f1`: 0.549541
- `log_loss`: 1.366717
- `top_2_accuracy`: 0.822263
- `top_3_accuracy`: 0.880657
- `top_5_accuracy`: 0.931022

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v05_roberta_mean_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v05_roberta_mean_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v05_roberta_mean_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v05_roberta_mean_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v05_roberta_mean.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v05_roberta_mean_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v05_roberta_mean_submission.csv`

# Run Summary: v06_roberta_mean_imbalance_aware_focal_gamma1

## Config

```text
version_name: v06_roberta_mean_imbalance_aware
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
selection_metric: accuracy
candidate_id: focal_gamma1
loss_name: focal_loss
focal_gamma: 1.0
use_class_weights: False
```

## Metrics

- `accuracy`: 0.557299
- `macro_precision`: 0.494966
- `macro_recall`: 0.485935
- `macro_f1`: 0.480394
- `weighted_f1`: 0.539776
- `log_loss`: 1.302235
- `top_2_accuracy`: 0.814599
- `top_3_accuracy`: 0.873358
- `top_5_accuracy`: 0.932847

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma1_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma1_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma1_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_focal_gamma1_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma1_per_class_metrics.csv`
- `confusion_matrix`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma1_confusion_matrix.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v06_roberta_mean_imbalance_aware_focal_gamma1.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_focal_gamma1_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v06_roberta_mean_imbalance_aware_focal_gamma1_submission.csv`

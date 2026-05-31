# Run Summary: v06_roberta_mean_imbalance_aware_debug_focal_gamma1

## Config

```text
version_name: v06_roberta_mean_imbalance_aware_debug
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
pooling: mean
seed: 42
max_length: 32
learning_rate: 2e-05
weight_decay: 0.01
batch_size: 4
max_epochs: 1
patience: 1
dropout: 0.1
validation_size: 0.2
use_amp: False
debug: True
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

- `accuracy`: 0.000000
- `macro_precision`: 0.000000
- `macro_recall`: 0.000000
- `macro_f1`: 0.000000
- `weighted_f1`: 0.000000
- `log_loss`: 3.506743
- `top_2_accuracy`: 0.062500
- `top_3_accuracy`: 0.125000
- `top_5_accuracy`: 0.250000

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_per_class_metrics.csv`
- `confusion_matrix`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_confusion_matrix.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v06_roberta_mean_imbalance_aware_debug_focal_gamma1.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v06_roberta_mean_imbalance_aware_debug_focal_gamma1_submission.csv`

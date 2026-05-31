# Run Summary: v06_roberta_mean_imbalance_aware_debug_class_weight_balanced

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
candidate_id: class_weight_balanced
loss_name: class_weighted_cross_entropy
focal_gamma: None
use_class_weights: True
```

## Metrics

- `accuracy`: 0.000000
- `macro_precision`: 0.000000
- `macro_recall`: 0.000000
- `macro_f1`: 0.000000
- `weighted_f1`: 0.000000
- `log_loss`: 3.573907
- `top_2_accuracy`: 0.062500
- `top_3_accuracy`: 0.125000
- `top_5_accuracy`: 0.187500

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_per_class_metrics.csv`
- `confusion_matrix`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_confusion_matrix.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v06_roberta_mean_imbalance_aware_debug_class_weight_balanced_submission.csv`

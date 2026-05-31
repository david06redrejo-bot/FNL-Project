# Run Summary: v06_roberta_mean_imbalance_aware_class_weight_balanced

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
candidate_id: class_weight_balanced
loss_name: class_weighted_cross_entropy
focal_gamma: None
use_class_weights: True
```

## Metrics

- `accuracy`: 0.544526
- `macro_precision`: 0.494840
- `macro_recall`: 0.578469
- `macro_f1`: 0.518270
- `weighted_f1`: 0.537546
- `log_loss`: 1.487585
- `top_2_accuracy`: 0.802555
- `top_3_accuracy`: 0.871898
- `top_5_accuracy`: 0.925547

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_class_weight_balanced_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_class_weight_balanced_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_class_weight_balanced_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_class_weight_balanced_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_class_weight_balanced_per_class_metrics.csv`
- `confusion_matrix`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_class_weight_balanced_confusion_matrix.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v06_roberta_mean_imbalance_aware_class_weight_balanced.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_class_weight_balanced_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v06_roberta_mean_imbalance_aware_class_weight_balanced_submission.csv`

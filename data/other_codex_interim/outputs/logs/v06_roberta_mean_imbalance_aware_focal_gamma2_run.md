# Run Summary: v06_roberta_mean_imbalance_aware_focal_gamma2

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
candidate_id: focal_gamma2
loss_name: focal_loss
focal_gamma: 2.0
use_class_weights: False
```

## Metrics

- `accuracy`: 0.555474
- `macro_precision`: 0.486794
- `macro_recall`: 0.513841
- `macro_f1`: 0.492846
- `weighted_f1`: 0.542396
- `log_loss`: 1.275351
- `top_2_accuracy`: 0.824088
- `top_3_accuracy`: 0.884307
- `top_5_accuracy`: 0.936131

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma2_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma2_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v06_roberta_mean_imbalance_aware_focal_gamma2_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_focal_gamma2_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma2_per_class_metrics.csv`
- `confusion_matrix`: `/home/iadlG010/FNL-Project/outputs/metrics/v06_roberta_mean_imbalance_aware_focal_gamma2_confusion_matrix.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v06_roberta_mean_imbalance_aware_focal_gamma2.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v06_roberta_mean_imbalance_aware_focal_gamma2_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v06_roberta_mean_imbalance_aware_focal_gamma2_submission.csv`

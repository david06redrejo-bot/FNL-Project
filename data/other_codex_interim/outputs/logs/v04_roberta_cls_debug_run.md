# Run Summary: v04_roberta_cls_debug

## Config

```text
version_name: v04_roberta_cls_debug
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
pooling: cls
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
```

## Metrics

- `accuracy`: 0.000000
- `macro_precision`: 0.000000
- `macro_recall`: 0.000000
- `macro_f1`: 0.000000
- `weighted_f1`: 0.000000
- `log_loss`: 3.613947
- `top_2_accuracy`: 0.000000
- `top_3_accuracy`: 0.125000
- `top_5_accuracy`: 0.187500

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_cls_debug_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v04_roberta_cls_debug_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_cls_debug_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v04_roberta_cls_debug_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v04_roberta_cls_debug_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v04_roberta_cls_debug.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v04_roberta_cls_debug_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v04_roberta_cls_debug_submission.csv`

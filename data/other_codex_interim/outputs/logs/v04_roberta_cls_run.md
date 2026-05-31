# Run Summary: v04_roberta_cls

## Config

```text
version_name: v04_roberta_cls
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
pooling: cls
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

- `accuracy`: 0.569343
- `macro_precision`: 0.498815
- `macro_recall`: 0.504519
- `macro_f1`: 0.494329
- `weighted_f1`: 0.554347
- `log_loss`: 1.327770
- `top_2_accuracy`: 0.822263
- `top_3_accuracy`: 0.878467
- `top_5_accuracy`: 0.930657

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_cls_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v04_roberta_cls_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v04_roberta_cls_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v04_roberta_cls_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v04_roberta_cls_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v04_roberta_cls.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v04_roberta_cls_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v04_roberta_cls_submission.csv`

# Run Summary: v08_roberta_mean_augmented_debug

## Config

```text
version_name: v08_roberta_mean_augmented_debug
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
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
debug_train_examples: 32
debug_val_examples: 32
debug_leaderboard_examples: 32
```

## Metrics

- `accuracy`: 0.156250
- `macro_precision`: 0.047237
- `macro_recall`: 0.098039
- `macro_f1`: 0.053333
- `weighted_f1`: 0.085000
- `log_loss`: 3.501691
- `top_2_accuracy`: 0.281250
- `top_3_accuracy`: 0.312500
- `top_5_accuracy`: 0.312500

## Artifacts

- `data_strategy_results`: `/home/iadlG010/FNL-Project/reports/tables/v08_data_strategy_results.csv`
- `duplicate_report`: `/home/iadlG010/FNL-Project/reports/tables/v08_roberta_mean_augmented_debug_duplicate_report.csv`
- `safety_note`: `/home/iadlG010/FNL-Project/reports/safe_augmentation_note.md`
- `canonical_submission`: `/home/iadlG010/FNL-Project/submissions/v08_roberta_mean_augmented_debug_submission.csv`
- `canonical_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_debug_metrics.json`

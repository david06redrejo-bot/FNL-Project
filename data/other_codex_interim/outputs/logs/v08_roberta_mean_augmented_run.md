# Run Summary: v08_roberta_mean_augmented

## Config

```text
version_name: v08_roberta_mean_augmented
backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
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
debug_train_examples: 32
debug_val_examples: 32
debug_leaderboard_examples: 32
```

## Metrics

- `accuracy`: 0.568613
- `macro_precision`: 0.503278
- `macro_recall`: 0.493996
- `macro_f1`: 0.481648
- `weighted_f1`: 0.549150
- `log_loss`: 1.344398
- `top_2_accuracy`: 0.809124
- `top_3_accuracy`: 0.868248
- `top_5_accuracy`: 0.922993

## Artifacts

- `data_strategy_results`: `/home/iadlG010/FNL-Project/reports/tables/v08_data_strategy_results.csv`
- `duplicate_report`: `/home/iadlG010/FNL-Project/reports/tables/v08_roberta_mean_augmented_duplicate_report.csv`
- `safety_note`: `/home/iadlG010/FNL-Project/reports/safe_augmentation_note.md`
- `canonical_submission`: `/home/iadlG010/FNL-Project/submissions/v08_roberta_mean_augmented_submission.csv`
- `canonical_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_metrics.json`

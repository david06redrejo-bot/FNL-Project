# Run Summary: v08_roberta_mean_augmented_dedupe_non_conflicting_literals

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
candidate_id: dedupe_non_conflicting_literals
strategy: drop duplicate train literals only when all copies share y_category
use_weighted_sampler: False
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

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v08_roberta_mean_augmented_dedupe_non_conflicting_literals.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v08_roberta_mean_augmented_dedupe_non_conflicting_literals_submission.csv`

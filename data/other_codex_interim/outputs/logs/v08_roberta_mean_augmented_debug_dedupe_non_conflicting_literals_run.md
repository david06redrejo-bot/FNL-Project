# Run Summary: v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals

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
candidate_id: dedupe_non_conflicting_literals
strategy: drop duplicate train literals only when all copies share y_category
use_weighted_sampler: False
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

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v08_roberta_mean_augmented_debug_dedupe_non_conflicting_literals_submission.csv`

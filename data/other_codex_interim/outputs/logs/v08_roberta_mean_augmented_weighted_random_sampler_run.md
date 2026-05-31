# Run Summary: v08_roberta_mean_augmented_weighted_random_sampler

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
candidate_id: weighted_random_sampler
strategy: keep all train rows and sample inversely to y_category frequency
use_weighted_sampler: True
```

## Metrics

- `accuracy`: 0.542336
- `macro_precision`: 0.510743
- `macro_recall`: 0.569113
- `macro_f1`: 0.522584
- `weighted_f1`: 0.536961
- `log_loss`: 1.576592
- `top_2_accuracy`: 0.802920
- `top_3_accuracy`: 0.870803
- `top_5_accuracy`: 0.921168

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_weighted_random_sampler_config.json`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_weighted_random_sampler_metrics.json`
- `history`: `/home/iadlG010/FNL-Project/outputs/logs/v08_roberta_mean_augmented_weighted_random_sampler_history.csv`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_weighted_random_sampler_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v08_roberta_mean_augmented_weighted_random_sampler_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v08_roberta_mean_augmented_weighted_random_sampler.pt`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v08_roberta_mean_augmented_weighted_random_sampler_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v08_roberta_mean_augmented_weighted_random_sampler_submission.csv`

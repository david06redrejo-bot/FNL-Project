# Run Summary: v03_similarity_retrieval_baseline

## Config

```text
version_name: v03_similarity_retrieval_baseline
model_family: similarity_retrieval
seed: 42
validation_size: 0.2
text_col: Literal_required_clean
target_col: y_category
label_id_col: label_id
debug_sample: None
save_model: True
top_k: 5
description_file_available: True
selected_by: highest validation accuracy; ties by macro_f1 and weighted_f1
best_params: {'method': 'train_literal_char_wb_3-5_k1', 'retrieval_index': 'training_literals', 'analyzer': 'char_wb', 'ngram_range': [3, 5], 'k': 1}
grid: [{'method': 'train_literal_word_1-2_k1', 'retrieval_index': 'training_literals', 'analyzer': 'word', 'ngram_range': '1-2', 'k': 1, 'accuracy': 0.458029197080292, 'macro_f1': 0.42431802071248376, 'weighted_f1': 0.46410076329264893}, {'method': 'train_literal_word_1-2_k3', 'retrieval_index': 'training_literals', 'analyzer': 'word', 'ngram_range': '1-2', 'k': 3, 'accuracy': 0.42956204379562046, 'macro_f1': 0.38081309726265644, 'weighted_f1': 0.43535609860164104}, {'method': 'train_literal_word_1-2_k5', 'retrieval_index': 'training_literals', 'analyzer': 'word', 'ngram_range': '1-2', 'k': 5, 'accuracy': 0.44416058394160585, 'macro_f1': 0.39332570207660017, 'weighted_f1': 0.44616457376949326}, {'method': 'train_literal_char_wb_3-5_k1', 'retrieval_index': 'training_literals', 'analyzer': 'char_wb', 'ngram_range': '3-5', 'k': 1, 'accuracy': 0.49744525547445256, 'macro_f1': 0.46278856539912183, 'weighted_f1': 0.49611957322951167}, {'method': 'train_literal_char_wb_3-5_k3', 'retrieval_index': 'training_literals', 'analyzer': 'char_wb', 'ngram_range': '3-5', 'k': 3, 'accuracy': 0.481021897810219, 'macro_f1': 0.42758096397338247, 'weighted_f1': 0.4751234592883174}, {'method': 'train_literal_char_wb_3-5_k5', 'retrieval_index': 'training_literals', 'analyzer': 'char_wb', 'ngram_range': '3-5', 'k': 5, 'accuracy': 0.47335766423357667, 'macro_f1': 0.42266330051395334, 'weighted_f1': 0.4647967219205447}, {'method': 'icd_description_word_1-2_k1', 'retrieval_index': 'icd_descriptions', 'analyzer': 'word', 'ngram_range': '1-2', 'k': 1, 'accuracy': 0.2697080291970803, 'macro_f1': 0.24025573850981788, 'weighted_f1': 0.25669477878212626}, {'method': 'icd_description_char_wb_3-5_k1', 'retrieval_index': 'icd_descriptions', 'analyzer': 'char_wb', 'ngram_range': '3-5', 'k': 1, 'accuracy': 0.33138686131386863, 'macro_f1': 0.248032382135319, 'weighted_f1': 0.28182574741326233}]
```

## Metrics

- `accuracy`: 0.497445
- `macro_precision`: 0.481510
- `macro_recall`: 0.454437
- `macro_f1`: 0.462789
- `weighted_f1`: 0.496120

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v03_similarity_retrieval_baseline_config.json`
- `grid_results`: `/home/iadlG010/FNL-Project/reports/tables/v03_similarity_retrieval_grid.csv`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v03_similarity_retrieval_baseline_metrics.json`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v03_similarity_retrieval_baseline_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v03_similarity_retrieval_baseline_per_class_metrics.csv`
- `correct_neighbor_examples`: `/home/iadlG010/FNL-Project/reports/tables/v03_similarity_correct_neighbors.csv`
- `wrong_neighbor_examples`: `/home/iadlG010/FNL-Project/reports/tables/v03_similarity_wrong_neighbors.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v03_similarity_retrieval_baseline.joblib`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v03_similarity_retrieval_baseline_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v03_similarity_retrieval_baseline_submission.csv`
- `classical_comparison`: `/home/iadlG010/FNL-Project/reports/tables/classical_baseline_comparison.csv`
- `classical_comparison_figure`: `/home/iadlG010/FNL-Project/reports/figures/fig_10_classical_baseline_comparison.png`

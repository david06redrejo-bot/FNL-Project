# Run Summary: v02_tfidf_word_svm

## Config

```text
version_name: v02_tfidf_word_svm
model_family: tfidf_word
seed: 42
validation_size: 0.2
text_col: Literal_required_clean
target_col: y_category
label_id_col: label_id
debug_sample: None
save_model: True
top_k: 5
grid: [{'ngram_range': '1-1', 'classifier': 'linear_svc', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-1', 'classifier': 'linear_svc', 'min_df': 2, 'sublinear_tf': True}, {'ngram_range': '1-1', 'classifier': 'logreg', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-1', 'classifier': 'logreg', 'min_df': 2, 'sublinear_tf': True}, {'ngram_range': '1-2', 'classifier': 'linear_svc', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-2', 'classifier': 'linear_svc', 'min_df': 2, 'sublinear_tf': True}, {'ngram_range': '1-2', 'classifier': 'logreg', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-2', 'classifier': 'logreg', 'min_df': 2, 'sublinear_tf': True}, {'ngram_range': '1-3', 'classifier': 'linear_svc', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-3', 'classifier': 'linear_svc', 'min_df': 2, 'sublinear_tf': True}, {'ngram_range': '1-3', 'classifier': 'logreg', 'min_df': 1, 'sublinear_tf': True}, {'ngram_range': '1-3', 'classifier': 'logreg', 'min_df': 2, 'sublinear_tf': True}]
selected_by: highest validation accuracy; ties by macro_f1 and weighted_f1
best_params: {'ngram_range': [1, 1], 'classifier': 'linear_svc', 'min_df': 1, 'sublinear_tf': True}
```

## Metrics

- `accuracy`: 0.520073
- `macro_precision`: 0.502503
- `macro_recall`: 0.465712
- `macro_f1`: 0.474196
- `weighted_f1`: 0.514018

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v02_tfidf_word_svm_config.json`
- `grid_results`: `/home/iadlG010/FNL-Project/reports/tables/v02_tfidf_word_grid.csv`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v02_tfidf_word_svm_metrics.json`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v02_tfidf_word_svm_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v02_tfidf_word_svm_per_class_metrics.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v02_tfidf_word_svm.joblib`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v02_tfidf_word_svm_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v02_tfidf_word_svm_submission.csv`
- `classical_comparison`: `/home/iadlG010/FNL-Project/reports/tables/classical_baseline_comparison.csv`
- `classical_comparison_figure`: `/home/iadlG010/FNL-Project/reports/figures/fig_10_classical_baseline_comparison.png`

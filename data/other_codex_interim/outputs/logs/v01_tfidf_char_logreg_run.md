# Run Summary: v01_tfidf_char_logreg

## Config

```text
version_name: v01_tfidf_char_logreg
model_family: tfidf_char_logreg
seed: 42
validation_size: 0.2
text_col: Literal_required_clean
target_col: y_category
label_id_col: label_id
debug_sample: None
save_model: True
top_k: 5
grid: [{'preprocessing_variant': 'required_clean', 'ngram_range': '2-4', 'class_weight': 'none'}, {'preprocessing_variant': 'required_clean', 'ngram_range': '2-4', 'class_weight': 'balanced'}, {'preprocessing_variant': 'required_clean', 'ngram_range': '3-5', 'class_weight': 'none'}, {'preprocessing_variant': 'required_clean', 'ngram_range': '3-5', 'class_weight': 'balanced'}, {'preprocessing_variant': 'required_clean', 'ngram_range': '2-6', 'class_weight': 'none'}, {'preprocessing_variant': 'required_clean', 'ngram_range': '2-6', 'class_weight': 'balanced'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '2-4', 'class_weight': 'none'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '2-4', 'class_weight': 'balanced'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '3-5', 'class_weight': 'none'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '3-5', 'class_weight': 'balanced'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '2-6', 'class_weight': 'none'}, {'preprocessing_variant': 'lowercase', 'ngram_range': '2-6', 'class_weight': 'balanced'}]
selected_by: highest validation accuracy; ties by macro_f1 and weighted_f1
best_params: {'preprocessing_variant': 'required_clean', 'ngram_range': [3, 5], 'class_weight': None}
```

## Metrics

- `accuracy`: 0.522628
- `macro_precision`: 0.469827
- `macro_recall`: 0.381912
- `macro_f1`: 0.402554
- `weighted_f1`: 0.494943
- `log_loss`: 1.710548
- `top_2_accuracy`: 0.748905
- `top_3_accuracy`: 0.824088
- `top_5_accuracy`: 0.883577

## Artifacts

- `config`: `/home/iadlG010/FNL-Project/outputs/logs/v01_tfidf_char_logreg_config.json`
- `grid_results`: `/home/iadlG010/FNL-Project/reports/tables/v01_tfidf_char_grid.csv`
- `metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v01_tfidf_char_logreg_metrics.json`
- `validation_predictions`: `/home/iadlG010/FNL-Project/outputs/predictions/v01_tfidf_char_logreg_val_predictions.csv`
- `per_class_metrics`: `/home/iadlG010/FNL-Project/outputs/metrics/v01_tfidf_char_logreg_per_class_metrics.csv`
- `top_ngrams`: `/home/iadlG010/FNL-Project/reports/tables/v01_tfidf_char_top_ngrams.csv`
- `model_artifact`: `/home/iadlG010/FNL-Project/outputs/checkpoints/v01_tfidf_char_logreg.joblib`
- `leaderboard_detailed`: `/home/iadlG010/FNL-Project/outputs/predictions/v01_tfidf_char_logreg_leaderboard_detailed.csv`
- `submission`: `/home/iadlG010/FNL-Project/submissions/v01_tfidf_char_logreg_submission.csv`

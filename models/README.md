# Model Version Interface

Every file in `models/` is a deliverable model-version script. The shared
contract is strict: each version must be able to load data, train or load a
model, evaluate on validation, predict the leaderboard, generate a named Kaggle
submission, write metrics, write predictions, and update experiment logs.

## Commands

Smoke-test the full interface on a tiny debug sample:

```bash
python models/v00_majority_baseline.py --dry-run
python models/v01_tfidf_char_logreg.py --dry-run
```

Run the full default flow for a lightweight model:

```bash
python models/v00_majority_baseline.py
```

Specific flags are also supported:

```bash
python models/v02_tfidf_word_svm.py --train --evaluate --predict --make-submission
python models/v02_tfidf_word_svm.py --load-model-path outputs/checkpoints/v02_tfidf_word_svm.joblib --evaluate --predict --make-submission
```

## Output Contract

For a version named `vXX_name`, every run writes:

- `outputs/logs/vXX_name_config.json`
- `outputs/metrics/vXX_name_metrics.json`
- `outputs/metrics/vXX_name_per_class_metrics.csv`
- `outputs/predictions/vXX_name_val_predictions.csv`
- `outputs/predictions/vXX_name_leaderboard_detailed.csv`
- `submissions/vXX_name_submission.csv`
- `outputs/logs/vXX_name_run.md`
- `outputs/checkpoints/vXX_name.joblib` when applicable

Kaggle submissions contain exactly:

```text
id,y_category
```

Detailed prediction files may include literals, validation truth labels,
probabilities, logits, or top-k fields depending on the model.

## Metrics

The interface writes:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- confusion matrix
- per-class metrics
- top-k accuracy when probabilities are available
- log loss when probabilities are available

## Split

The internal split is `80/20`, stratified by `label_id`, with configurable seed
defaulting to `42`.

## Versions

| Version | Purpose |
|---|---|
| `v00_majority_baseline.py` | Majority-class lower bound |
| `v01_tfidf_char_logreg.py` | Character n-gram TF-IDF logistic regression |
| `v02_tfidf_word_svm.py` | TF-IDF linear SVM baseline |
| `v03_similarity_retrieval_baseline.py` | TF-IDF nearest-neighbor retrieval baseline |
| `v04_roberta_cls.py` | RoBERTa with CLS pooling |
| `v05_roberta_mean.py` | RoBERTa with mean pooling |
| `v06_roberta_mean_imbalance_aware.py` | RoBERTa mean pooling with weighted/focal losses |
| `v07_roberta_mean_tuning.py` | Controlled mean-pooling hyperparameter tuning |
| `v08_roberta_mean_augmented.py` | Safe data-strategy ablations: conservative deduplication and weighted sampling |
| `v09_ensemble.py` | Validation-driven ensemble over completed model predictions |

Legacy skeleton scripts are still present where useful for continuity, but the
validated model sequence is `v00` through `v08` as listed above.

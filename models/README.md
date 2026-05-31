# Model Versions

Each file in this folder is a runnable model-version entry point. The shared
interface is:

```bash
python models/v00_majority_baseline.py --dry-run
python models/v02_tfidf_word_svm.py --train --evaluate --predict --make-submission
```

If no action flag is provided, the script runs the full train/evaluate/predict
and submission flow. All model versions write outputs to:

- `outputs/metrics/`
- `outputs/predictions/`
- `submissions/`
- `EXPERIMENT_LOG.md`
- `REPORT_NOTES.md`

The classical models are already wired to reusable code in `src/`. The RoBERTa
and ensemble files currently provide the final traceable command interface and
will receive their full transformer/ensemble internals during the modeling
phase once data and compute availability are confirmed.

## Planned Versions

| Version | Purpose |
|---|---|
| `v00_majority_baseline.py` | Majority-class lower bound |
| `v01_tfidf_char_logreg.py` | Character n-gram TF-IDF logistic regression |
| `v02_tfidf_word_svm.py` | TF-IDF linear SVM baseline |
| `v03_roberta_cls.py` | RoBERTa with CLS pooling |
| `v04_roberta_mean.py` | RoBERTa with mean pooling |
| `v05_roberta_mean_class_weighted.py` | RoBERTa mean pooling with class weighting |
| `v06_roberta_mean_augmented.py` | RoBERTa with ICD-description augmentation |
| `v07_ensemble.py` | Final ensemble candidate |


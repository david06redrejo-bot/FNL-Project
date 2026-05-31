# v09 Ensemble Recipe

Selection policy: choose the best predefined ensemble by validation accuracy.
No leaderboard labels or public leaderboard feedback were used.

Selected candidate: `v09_majority_vote`
Recipe: majority vote over ['v04_roberta_cls', 'v05_roberta_mean', 'v08_roberta_mean_dedupe', 'v08_roberta_mean_weighted_sampler', 'v01_tfidf_char_logreg'] with average-probability tie-break

Inputs:
- `v04_roberta_cls`: roberta_cls
- `v05_roberta_mean`: roberta_mean
- `v08_roberta_mean_dedupe`: roberta_mean_safe_data
- `v08_roberta_mean_weighted_sampler`: roberta_mean_safe_data
- `v06_roberta_mean_class_weighted`: roberta_mean_imbalance
- `v06_roberta_mean_focal_gamma2`: roberta_mean_imbalance
- `v01_tfidf_char_logreg`: tfidf_char

Important caution: ensembling can reduce variance only if models make partially different errors. If model errors are highly correlated, averaging may not improve over the best single model.
# Report Notes

This file accumulates observations, explanations, and writing ideas for the final report. The final report should be written in English and should sound like a student team progressively understanding the task through the course.

## 2026-05-31 — Initial Storyline After Repository Takeover

The repository already suggests a good narrative arc, but it needs to be rebuilt and verified:

1. We first understood that the task is not full-document ICD coding, but short clinical literal classification.
2. The target for the Kaggle task is exactly one prefix category, `y_category`, defined as the first character of `Code`.
3. The expected label space is 36 categories: digits `0`-`9` and letters `a`-`z` / `A`-`Z` after normalization of case.
4. The early EDA reports describe very short inputs, around two words on average, with mixed Spanish/Catalan, abbreviations, accents, digits, and inconsistent casing.
5. Existing reports claim strong label imbalance and many-to-many literal/code ambiguity. These points are central to the report, but must be rerun from the raw dataset before final claims.
6. The survey/literature-review material motivates classic TF-IDF + SVM methods as reference baselines and also explains why neural models such as RoBERTa are worth testing.
7. The final narrative should not pretend that the team knew the best method from the beginning. It should show a progression: inspect annotations, understand the challenge, build simple baselines, improve them, then compare with a RoBERTa backbone and decide based on evidence.

Important writing rule: do not fabricate leaderboard scores or validation results. Use only rerun metrics or clearly label historical notebook values as previous/reported results.

Potential report section order:

1. Introduction and task definition.
2. Dataset and annotation analysis.
3. Main challenges.
4. Related work and reference methods.
5. Preprocessing and reproducibility.
6. Baseline models.
7. Improved traditional models.
8. RoBERTa backbone experiment.
9. Evaluation and error analysis.
10. Submission and conclusions.

## 2026-05-31 — Repository Skeleton Implemented

The final project structure now reflects the story we want the report to tell:
we start with the data and annotations, then explain the main task challenges,
then move from survey-inspired baselines to RoBERTa and improved models.

The report should explicitly say that notebooks are narrative and scripts are
the reproducibility backbone. This is a good way to show that the team learned
to separate exploration from reusable implementation.

## 1. Analyzing the Data and the Annotations

We inspected all CSV files under `data/` and validated them against the competition annotation contract.

| File | Role | Shape | Schema status |
|---|---|---:|---|
| `codification_data.csv` | `codification` | 13700 x 2 | valid |
| `icd_d_p_pairs.csv` | `icd_catalog` | 179742 x 3 | valid |
| `leaderboard_data.csv` | `leaderboard` | 6667 x 2 | valid |

The training labels are derived as `y_category = Code.astype(str).str[0]`.
The observed label set has 36 classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.

Warnings:
- `y_category` is absent from leaderboard data; this is normal for unlabeled Kaggle test files but differs from the stated expected columns.

## Visual EDA Conclusions

We continued the first phase with a visual EDA inspired by a Data Engineering
mindset: before modeling, we inspected the data-generating process, data
quality, distributions, missingness, duplicates, and possible leakage.

Core size and annotation facts:

- Training rows: 13700
- Leaderboard rows: 6667
- Unique full ICD codes in training: 4059
- Unique `y_category` labels: 36
- Unique full ICD codes in the ICD catalog: 179742

The top 10 categories by row count are: [{'y_category': 'Z', 'count': 1715}, {'y_category': 'O', 'count': 1505}, {'y_category': '0', 'count': 1141}, {'y_category': '6', 'count': 637}, {'y_category': '3', 'count': 592}, {'y_category': 'B', 'count': 579}, {'y_category': 'N', 'count': 536}, {'y_category': 'E', 'count': 500}, {'y_category': 'V', 'count': 491}, {'y_category': '5', 'count': 408}].
The bottom 10 categories by row count are: [{'y_category': 'W', 'count': 7}, {'y_category': 'X', 'count': 10}, {'y_category': 'A', 'count': 22}, {'y_category': 'U', 'count': 22}, {'y_category': 'Y', 'count': 36}, {'y_category': 'T', 'count': 70}, {'y_category': 'S', 'count': 71}, {'y_category': 'P', 'count': 101}, {'y_category': 'L', 'count': 110}, {'y_category': 'Q', 'count': 145}].

Duplicate and ambiguity findings:

- Duplicate literals: 1668
- Same literal with multiple full ICD codes: 1668
- Same literal with multiple `y_category` labels: 1486

Train/leaderboard comparison:

- 51.8% of unique normalized leaderboard
  literals appear in the training set.
- Mean character length is 16.95 in train and
  17.15 in leaderboard.

Normalization risk:

- Raw unique literals: 11584
- Unique literals after lowercase/accent-stripped normalization:
  7844
- Normalization collision keys:
  2186

No model has been trained yet; these are EDA conclusions only.

## Preprocessing Design: Light Cleaning for RoBERTa

The final preprocessing decision is deliberately conservative. For the RoBERTa
pipeline, we use only required light cleanup: convert null-safe values to text,
strip leading/trailing spaces, and collapse repeated whitespace.

We do **not** lowercase, remove accents, or remove punctuation for the final
RoBERTa pipeline because the backbone tokenizer is pretrained on Spanish
biomedical and clinical text. Aggressive normalization could remove useful
signals from accents, uppercase abbreviations, punctuation, digits, and compact
clinical notation.

Processed files created:

- `data/processed/train_required_clean.csv`
- `data/processed/leaderboard_required_clean.csv`

Ablation artifacts for analysis/classical baselines:

- `data/interim/preprocessing_ablation/preprocessing_ablation_summary.csv`
- `data/interim/preprocessing_ablation/preprocessing_ablation_examples.csv`

Observed ablation impact:

```text
                    variant  changed_rows  changed_share
required whitespace cleanup             3       0.000219
                  lowercase          8399       0.613066
             accent removal          3833       0.279781
        punctuation removal          1338       0.097664
```

No model has been trained yet; these are preprocessing-design conclusions only.

## Tokenizer Analysis Blocked

Tokenizer download/load failed for `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`.

Exact error:
OSError: Can't load the configuration of 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es' is the correct path to a directory containing a config.json file

Retry command:
python scripts/analyze_tokenization.py

No tokenization statistics were fabricated.

## RoBERTa Tokenization Analysis

We analyzed tokenization with `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` using the required-clean literals
and without training any model.

Subword tokenization matters because RoBERTa does not see whitespace words
directly. It sees subword pieces created by the pretrained tokenizer. Therefore,
`max_length` is not arbitrary: too small a value truncates information; too
large a value wastes compute.

Token length summary including special tokens:

- Train p50/p75/p90/p95/p99/max:
  5/6/8/9/12/24
- Leaderboard p50/p75/p90/p95/p99/max:
  5/7/8/9/12/20

Recommended default `max_length`: **32**.

This confirms that our task is much easier than long EMR ICD coding with full
discharge summaries from a sequence-length point of view. The inputs are short
literals, not multi-page documents. However, the same shortness creates another
problem: ambiguous literals often lack the surrounding context that would
disambiguate the correct ICD category.

No model has been trained yet; these are tokenizer-design conclusions only.

## Survey-to-Project Method Strategy

After reading Yan et al. (2022), we understood that automated ICD coding is not
ordinary text classification. The survey frames ICD coding as a clinical,
administrative, and hierarchical NLP problem: manual coding is slow, coding
errors affect reimbursement and hospital management, and ICD supports
statistics, standardization, DRGs, and medical-record management.

The survey also helped us decide what is realistic for this Kaggle assignment.
We are not solving full multi-label ICD coding over long EMRs. Our target is one
first-character category for short clinical literals. This makes the sequence
length problem much smaller, but the task is still meaningful because the data
is imbalanced, abbreviated, clinically ambiguous, and evaluated with strict
category accuracy.

Concrete strategy from the survey:

- Implement: majority baseline, TF-IDF character/word n-grams, possible fuzzy or
  nearest-neighbor lookup, Spanish biomedical-clinical RoBERTa, class-weighting
  ablations, pooling strategies, simple ensembling, and confidence/error
  analysis.
- Future work: full ICD hierarchy GNNs, label-description matching as a central
  model, full-code prediction, multi-label modeling, knowledge graphs, and
  clinical deployment/interpretability.

No model has been trained in this step; this is conceptual grounding and project
strategy.

## First Modeling Baseline: Majority Category

We implemented `v00_majority_baseline` as the first real model version. This
baseline deliberately ignores the literal text and predicts the most frequent
`y_category` observed in the training portion of the stratified 80/20 split.

With seed 42, the split contained 10,960 training rows and 2,740 validation
rows. The most frequent category in the training split was `Z`, with 1,372
examples, so the baseline predicted `Z` for every validation and leaderboard
example.

Validation results:

```text
accuracy: 0.125182
macro_f1: 0.006181
weighted_f1: 0.027854
```

This result is useful because it separates class-prior performance from actual
language understanding. The accuracy is not zero because the dataset is
imbalanced, but the macro F1 is almost zero because the model never predicts the
minority categories. In the report, we will use this as the minimum modeling
threshold: every serious baseline must beat it and must predict a broader set
of categories.

## Character TF-IDF Logistic Regression Baseline

We implemented `v01_tfidf_char_logreg` as the first traditional machine
learning model. The motivation came from Basic Text Processing and from the ICD
coding survey's historical stage of traditional ML methods: before using
Transformers, a strong vector-space baseline can already capture many surface
patterns.

Character n-grams are especially appropriate for these clinical literals
because the text is short and often compact. They can capture Spanish
morphology, abbreviations, punctuation, digits, laterality fragments, and pieces
of medical terms even when whitespace tokenization is unreliable.

The internal grid compared:

- n-gram ranges `(2,4)`, `(3,5)`, and `(2,6)`;
- `class_weight=None` and `class_weight="balanced"`;
- required-clean text versus a lowercase ablation.

The best validation configuration was required-clean text, `char_wb` TF-IDF
with n-grams `(3,5)`, LogisticRegression, and no class weighting.

Validation results:

```text
accuracy: 0.522628
macro_f1: 0.402554
weighted_f1: 0.494943
```

Lowercasing tied the required-clean setting in accuracy but did not improve it,
so we keep the conservative preprocessing decision. The model is much stronger
than the majority baseline, but it still has clear limitations: it uses surface
form evidence only, cannot understand clinical context, and may fail on
synonyms, ambiguity, negation, and categories that require semantic knowledge.

## Word TF-IDF Linear SVM Baseline

We implemented `v02_tfidf_word_svm` to compare character-level surface patterns
against explicit word-level lexical evidence. This is still a traditional
vector-space model, but it asks a different question: are complete clinical
tokens and short word phrases enough to classify the ICD category?

The grid compared unigram, unigram+bigram, and unigram+bigram+trigram TF-IDF
features, LinearSVC and LogisticRegression classifiers, and `min_df` values 1
and 2. The best validation configuration was word unigram TF-IDF with
LinearSVC, `min_df=1`, and `sublinear_tf=True`.

Validation results:

```text
accuracy: 0.520073
macro_f1: 0.474196
weighted_f1: 0.514018
```

Compared with the character n-gram model, word TF-IDF is slightly lower in
accuracy but better in macro F1 and weighted F1. This suggests that lexical
tokens help the model distribute predictions across classes more evenly, while
character n-grams are very strong for exact surface-form cues. Both baselines
are useful before moving to Transformers because they define what can be solved
with sparse vector representations alone.

## Similarity Retrieval Baseline

We implemented `v03_similarity_retrieval_baseline` as an optional traditional
method inspired by the information-retrieval view of ICD coding. Instead of
training a discriminative classifier, the model retrieves the nearest known
clinical literal or ICD description and assigns the retrieved item's
`y_category`.

The dataset contains `data/raw/icd_d_p_pairs.csv`, with ICD `Code`, `D_P`, and
`Description`, so we tested both training-literal retrieval and literal-to-ICD
description retrieval.

Best validation configuration:

```text
retrieval index: training literals
representation: TF-IDF char_wb
ngram_range: (3, 5)
k: 1
```

Validation results:

```text
accuracy: 0.497445
macro_f1: 0.462789
weighted_f1: 0.496120
```

The training-literal nearest-neighbor variant worked better than direct
literal-to-description retrieval. The description methods were weaker, likely
because the competition literals are short, informal, abbreviated, and sometimes
closer to hospital coding phrases than to formal ICD descriptions.

The nearest-neighbor examples were very informative. Some correct predictions
come from exact or near-exact literal matches. Some wrong predictions also have
cosine similarity 1.0, which reveals an important risk: identical or nearly
identical literals can map to different categories when the original code
differs. This supports the decision to keep retrieval as an ablation and move on
to stronger learned models.

## Deep Learning Infrastructure Prepared

Before training RoBERTa, we prepared the PyTorch infrastructure. This step was
intentionally separated from full model training so that we could verify the
data contract first.

Implemented components:

- `ICDLiteralDataset` for train, validation, and leaderboard modes;
- tokenizer loading from `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`;
- default `max_length=32`, based on the previous tokenization analysis;
- deterministic label mappings;
- device detection and seed control for Python, NumPy, PyTorch, and CUDA;
- one-epoch training, evaluation, prediction, checkpoint save/load utilities;
- tests for dataset shapes, label mapping, batch structure, leaderboard mode,
  and submission format.

The smoke test used only 16 training examples and 8 validation examples with a
tiny classifier. It did not train the full RoBERTa model. Its purpose was to
verify that the tokenizer, datasets, dataloaders, metrics, checkpointing, and
submission contract work together before the expensive Transformer experiments.

## RoBERTa CLS-Pooling Baseline

We implemented `v04_roberta_cls` as the required deep learning baseline. The
model uses `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` as the backbone,
takes the first token representation (`hidden_states[:, 0, :]`) as the CLS
feature, applies dropout 0.1, and predicts the 36 ICD categories with a linear
classification layer.

Training configuration:

```text
learning_rate: 2e-5
weight_decay: 0.01
batch_size: 128
max_epochs: 50
patience: 10
best_epoch: 10
device: cuda
```

Validation results from the best checkpoint:

```text
accuracy: 0.569343
macro_f1: 0.494329
weighted_f1: 0.554347
```

The public/reference CLS accuracy mentioned for orientation is 0.565, but we do
not treat that as our reproduced Kaggle public score. Our number above is the
internal validation result on the project's stratified 80/20 split.

This result is an important milestone because it beats the classical baselines
in accuracy and weighted F1. However, macro F1 remains close to the best
traditional methods, so the next Transformer experiments should focus on
minority-category robustness, pooling choices, class weighting, and error
analysis rather than only improving overall accuracy.

## RoBERTa Mean-Pooling Baseline

We implemented `v05_roberta_mean` to test whether averaging contextual token
representations is better than relying only on the first `<s>` token. This is
plausible for short clinical literals because the informative evidence may be
spread across the few biomedical terms, abbreviations, digits, or modifiers in
the literal.

The pooling operation is attention-mask aware:

```python
mask = attention_mask.unsqueeze(-1)
features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

Training used the same split and hyperparameters as the CLS baseline.

Validation results:

```text
accuracy: 0.564599
macro_f1: 0.496567
weighted_f1: 0.549541
best_epoch: 10
```

The reference orientation mentioned in the assignment was mean pooling 0.570 vs
CLS 0.565, but our actual internal validation comparison is slightly different:
CLS has higher accuracy, while mean pooling has slightly higher macro F1. Since
the competition is accuracy-oriented, CLS remains the current candidate final
model, but mean pooling is a strong alternative when looking at minority-class
balance.

## Advanced Model Experiment Roadmap

Before implementing more advanced models, we created a structured roadmap. This
was important because the project already had many possible directions, and we
wanted the next experiments to answer clear questions rather than become random
model stacking.

What we learned so far:

- EDA showed imbalance, ambiguity, duplicated literals, and possible leaderboard
  distribution shift.
- Preprocessing showed that clinical Spanish should not be aggressively
  normalized for RoBERTa.
- Classical baselines showed that surface lexical information is already strong.
- Retrieval showed that similarity is intuitive but fragile when literals are
  ambiguous.
- CLS pooling is the current best model by validation accuracy.
- Mean pooling slightly improves macro F1 but does not beat CLS accuracy.

Remaining weaknesses:

- rare categories still have very low recall;
- some labels have zero recall in the RoBERTa validation reports;
- validation loss rises after the best epoch, suggesting overfitting;
- short literals can lack context;
- leaderboard shift cannot be ruled out from observed fields alone.

Next planned priorities:

- class-weighted loss;
- learning-rate and warmup tuning;
- dropout tuning;
- ensembling best classical and RoBERTa models;
- calibration and confidence/error analysis.

Maybe/future directions include focal loss, label smoothing, freezing/unfreezing,
safe augmentation, layer-wise learning-rate decay, and pseudo-labeling. These
are useful ideas, but they need controlled validation before becoming final
claims.

## Imbalance-Aware RoBERTa Experiments

We implemented `v06_roberta_mean_imbalance_aware` to test whether changing the
loss function helps the long-tail label distribution observed in EDA and
discussed in the ICD coding survey.

The class weights were computed from the training split only, so no validation
or leaderboard information leaked into the loss. We tested:

- mean pooling with class-weighted CrossEntropyLoss;
- mean pooling with focal loss gamma 1;
- mean pooling with focal loss gamma 2.

Results:

```text
v05 standard mean          accuracy 0.564599  macro_f1 0.496567
v06 focal gamma 1          accuracy 0.557299  macro_f1 0.480394
v06 focal gamma 2          accuracy 0.555474  macro_f1 0.492846
v06 class-weighted CE      accuracy 0.544526  macro_f1 0.518270
```

The main lesson is a trade-off. Class-weighted CE improved macro F1 and helped
some categories, but it reduced accuracy. Focal loss preserved accuracy better
than class weighting, but did not improve over the standard mean-pooling
baseline. Therefore, imbalance-aware losses are useful ablations and evidence
for the report, but they should not replace CLS as the current final candidate
under the competition's accuracy-oriented objective.

## RoBERTa Mean-Pooling Hyperparameter Tuning

We implemented `v07_roberta_mean_tuning` as a controlled staged search rather
than a large grid. The tuned parameters were chosen because they matter for
Transformer fine-tuning and because they are connected to previous evidence:

- `max_length`: informed by tokenization analysis, with 32 as the default and 64
  as a sanity check;
- learning rate: 1e-5, 2e-5, and 3e-5 were considered, with controlled runs for
  2e-5 and 3e-5;
- batch size: 32 for quick checks, 64 for medium runs, 128 for full training;
- dropout and weight decay: regularization knobs because validation loss rises
  after the best epoch;
- warmup scheduler and gradient clipping: standard stabilizers for PLM
  fine-tuning;
- AMP: tested in Stage A only, kept off for the final run for reproducibility.

The final Stage C configuration used max_length 32, learning rate 2e-5, batch
size 128, dropout 0.1, weight decay 0.01, linear warmup ratio 0.06, and gradient
clipping 1.0.

Validation result:

```text
accuracy: 0.564599
macro_f1: 0.494151
weighted_f1: 0.549054
best_epoch: 13
```

This matched the standard mean-pooling accuracy but did not beat CLS. Therefore
the recommended mean-pooling training configuration is useful for future runs,
but CLS remains the current final candidate by validation accuracy.

## Safe Data Strategies and Clinical Augmentation

We explored data augmentation only under a clinical-safety constraint. After the
EDA and preprocessing phases, we understood that “augmentation” in medical text
is risky: a literal may be only a few tokens long, and changing a word,
punctuation mark, abbreviation, number, or negation could silently change the
diagnosis category.

For this reason, `v08_roberta_mean_augmented` deliberately avoids random word
deletion, unverified synonym replacement, negation changes, and back-translation.
Those ideas are left as future work unless they can be reviewed with domain
experts or controlled with verified medical terminology resources.

The safe experiments were data-handling strategies:

- a reference to the original `v05` mean-pooling run;
- conservative deduplication, where duplicate literals are dropped only when all
  copies have the same `y_category`;
- `WeightedRandomSampler`, which keeps all rows but samples rare categories more
  often during training.
- a custom class-balanced batch sampler was considered, but we kept it as future
  work because `WeightedRandomSampler` already tests the sampling hypothesis
  with less implementation risk.

The duplicate report for the training split showed:

```text
train rows: 10960
duplicate literal rows: 1442
conflicting duplicate literals: 1032
```

This confirmed that duplicate handling is not trivial. If the same literal maps
to different categories, removing it as a duplicate would hide ambiguity instead
of solving it.

Validation results:

```text
v05 original mean reference          accuracy 0.564599  macro_f1 0.496567
v08 conservative deduplication       accuracy 0.568613  macro_f1 0.481648
v08 weighted random sampler          accuracy 0.542336  macro_f1 0.522584
```

Conservative deduplication almost reached the CLS model, but it did not surpass
it. Weighted sampling improved macro F1, which means it helped the long-tail
view of the task, but the accuracy drop is too large for the competition
objective. Our responsible-AI conclusion is that safe data strategies are useful
for analysis, but they should not be oversold as clinical augmentation.

## Ensemble After Individual Models

After several individual models existed, we implemented `v09_ensemble` as a
validation-only ensemble. This was intentionally delayed until we had different
families of evidence: RoBERTa CLS, RoBERTa mean pooling, safe-data mean pooling,
weighted-sampling mean pooling, imbalance-aware variants, and TF-IDF character
logistic regression.

The ensemble did not use leaderboard labels or public leaderboard feedback. It
only collected saved validation and leaderboard predictions from completed
model runs, checked that the validation rows and label probability columns were
aligned, and compared predefined recipes:

- unweighted probability averaging over neural models;
- weighted probability averaging over neural models;
- neural averaging with the weighted-sampler model;
- probability averaging with TF-IDF;
- majority vote over selected complementary models;
- a low-confidence neural / high-confidence TF-IDF fallback rule.

Validation results:

```text
v04 RoBERTa CLS                  accuracy 0.569343  macro_f1 0.494329
v09 weighted neural average      accuracy 0.571898  macro_f1 0.498788
v09 TF-IDF fallback              accuracy 0.572628  macro_f1 0.499071
v09 majority vote                accuracy 0.576642  macro_f1 0.506277
```

The selected ensemble is majority vote over `v04_roberta_cls`,
`v05_roberta_mean`, `v08_roberta_mean_dedupe`,
`v08_roberta_mean_weighted_sampler`, and `v01_tfidf_char_logreg`, with
average-probability tie-breaking. This became the new best validation result.

The Machine Learning lesson is that ensembling can reduce variance when models
make partially different errors. The limitation is that it can fail when errors
are highly correlated, which is why not all probability averages beat the best
single model by a large margin. Future work should compare our approach with
other podium teams' models and test a podium ensemble only if competition rules
and academic reporting allow it.

## Final Evaluation and Error Analysis

Notebook 07 asks: “Which model should we trust as our final submission?” The
answer from the shared validation split is `v09_ensemble`.

The final comparison table is saved at
`reports/tables/final_experiment_comparison.csv`. The key result is:

```text
v09_ensemble accuracy 0.576642 macro_f1 0.506277 weighted_f1 0.561544
```

This improves over the best single model, `v04_roberta_cls`, which reached
accuracy `0.569343`, macro F1 `0.494329`, and weighted F1 `0.554347`.

The error analysis uses real validation literals and saves:

- `reports/tables/final_per_class_metrics.csv`;
- `reports/tables/final_error_examples.csv`;
- `reports/tables/final_top_confusions.csv`;
- final figures `fig_10` through `fig_14`.

The examples show all four confidence/error groups: correct high-confidence,
correct low-confidence, wrong high-confidence, and wrong low-confidence. The
main causes we identified are ambiguous literals, insufficient context,
abbreviations, class imbalance, similar ICD categories, and the broad nature of
first-character ICD prefixes.

Interpretability is treated carefully. We use confusion matrices, confidence
analysis, margins, and representative examples. We do not claim that
probabilities or future attention/token attributions are faithful clinical
explanations; they are diagnostics that help us understand model behavior.

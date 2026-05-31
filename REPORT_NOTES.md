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

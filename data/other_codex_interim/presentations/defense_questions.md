# Defense Questions and Answers

## 1. Why did you not lowercase the text?

Because case can carry useful clinical signal. Many literals contain uppercase
abbreviations such as `VHC`, `HTA`, or procedure shorthand. Since the final
RoBERTa tokenizer is pretrained on Spanish biomedical and clinical text, we
preferred light cleaning only: strip spaces and collapse repeated whitespace.
Lowercasing was treated as an ablation for classical baselines, not as the
default.

## 2. Why did you test mean pooling?

The inputs are very short clinical literals. In that setting, the informative
evidence may be spread across all tokens rather than concentrated in the first
RoBERTa token. Mean pooling averages non-padding token representations, so it is
a reasonable alternative to CLS pooling. In our runs, CLS had slightly better
accuracy, while mean pooling was competitive and sometimes better for macro-F1.

## 3. Why not predict the full ICD code?

The Kaggle task asks for the first-character ICD category prefix, not the full
ICD code. Full ICD prediction would be a much larger and more clinically
specific problem, with thousands of possible labels and stronger hierarchy
requirements. We kept the project aligned with the assignment.

## 4. Why not multi-label classification?

The competition requires exactly one `y_category` per literal. Multi-label ICD
coding is common for full medical records, but this dataset and leaderboard are
single-label. A multi-label setup would not match the submission contract.

## 5. Why use biomedical-clinical Spanish RoBERTa?

The literals are clinical and mostly Spanish/Catalan hospital text. A
biomedical-clinical Spanish pretrained language model should tokenize and
represent medical terminology better than a generic or English model. This
connects to the course idea of pretrained language models transferring
linguistic knowledge to downstream tasks.

## 6. How did you avoid overfitting to the leaderboard?

We kept a stratified 80/20 internal validation split and reported validation
metrics separately from Kaggle public scores. The final report states when a
model is validation-selected and when it is public-leaderboard-best. We did not
use leaderboard labels, and the private/final ranking is not claimed without
evidence.

## 7. How did you handle imbalance?

First, we measured it in EDA and used macro-F1 in addition to accuracy. Then we
tested class-weighted cross entropy, focal loss, weighted sampling, and
ensembles. Class weighting and sampling helped macro-F1 in some cases but often
hurt accuracy, so we reported the trade-off instead of hiding it.

## 8. What would be needed for real hospital deployment?

A hospital system would need expert clinical validation, privacy review,
security, monitoring, bias checks, explainability support, integration with
professional coders, and clear accountability. It should suggest or prioritize
codes, not replace coders directly.

## 9. What did the classical baselines teach you?

They taught us that short clinical literals contain a lot of surface-form
signal. Character TF-IDF captured abbreviations, morphology, punctuation, and
digits. Word TF-IDF captured lexical evidence and had strong macro-F1. These
models were not just baselines; they became useful members of the final diverse
ensemble.

## 10. What would you do with more time?

We would explore full ICD-code prediction, explicit ICD hierarchy modeling,
label-description retrieval, calibration, stronger confidence thresholds,
careful error review with domain experts, and possibly graph neural networks
over ICD relations. We would also compare with other teams' models only if
allowed and available.

## 11. Why is accuracy not enough?

Accuracy is the leaderboard metric, but the dataset is imbalanced. A model can
score well by performing strongly on frequent categories while failing rare
ones. Macro-F1 treats every class equally, so it tells us more about minority
category behavior.

## 12. How is this connected to the course?

The project used many course ideas in one pipeline: corpora and dataset
inspection, Basic Text Processing, regular expressions, tokenization, TF-IDF,
vector-space models, Transformers, pretrained language models, fine-tuning,
evaluation metrics, error analysis, and ensembles from Machine Learning.

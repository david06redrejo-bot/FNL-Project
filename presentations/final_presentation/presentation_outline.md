# Final Presentation Outline

Target duration: 6-8 minutes.

## 1. Opening: Who We Are

- Team 10: Phoebe Iglesias, David Redrejo, Pau Rossell.
- Fundamentals of Natural Language / NLP-I.
- BSc Artificial Intelligence, Universitat Autonoma de Barcelona.
- Supervisors: Ernest Valveny and Lei Kang.

## 2. Why ICD Coding Matters

- ICD codes support hospital records, statistics, DRGs, reimbursement, and
  management.
- Manual coding is expert work and can be slow or inconsistent.
- Automated coding can support professionals, but it must be handled carefully.

## 3. Task Formulation

- Input: one short clinical literal.
- Output: one ICD category prefix, `y_category`.
- Annotation: `y_category = first character of Code`.
- Single-label 36-class classification problem.

## 4. Dataset and EDA

- Training rows: 13,700.
- Leaderboard rows: 6,667.
- ICD catalogue rows: 179,742.
- Training labels cover 36 categories.
- EDA revealed imbalance, duplicate literals, ambiguous mappings, abbreviations,
  accents, punctuation, and digits.

## 5. Main Challenges

- Class imbalance.
- Short literals with limited clinical context.
- Ambiguous or duplicated literals.
- Medical abbreviations and shorthand.
- Broad prefix categories rather than full ICD codes.
- Public leaderboard feedback is useful but not the same as private/final
  evaluation.

## 6. Methods and Model Evolution

- `v00`: majority baseline.
- `v01`: character TF-IDF logistic regression.
- `v02`: word TF-IDF SVM.
- `v03`: similarity/retrieval baseline.
- `v04`: RoBERTa CLS pooling.
- `v05`: RoBERTa mean pooling.
- `v06-v08`: imbalance-aware losses, tuning, and safe data strategies.
- `v09-v10`: ensembles over complementary model families.

## 7. Results

- Majority baseline: validation accuracy 0.125.
- Best single RoBERTa CLS: validation accuracy 0.569.
- Best macro-F1-oriented ensemble: `v09_ensemble`, macro F1 0.506.
- Final public candidate: `v10_vote_diverse_no_retrieval`.
- Best verified public score in repository files: 0.587.
- Private/final ranking should only be stated if verified evidence is added.

## 8. Error Analysis and Interpretability

- Accuracy is the leaderboard metric, but macro-F1 matters because the data is
  imbalanced.
- Common errors come from ambiguity, abbreviations, rare categories, and broad
  ICD-prefix confusion.
- Confidence helps debugging but should not be treated as a clinical
  explanation.

## 9. What Would Happen in a Hospital?

- Benefits: faster suggestions, consistency, statistics, coder support.
- Risks: wrong reimbursement, wrong statistics, overtrust, privacy, bias.
- The model should support professional coders, not replace them.

## 10. Limitations

- Predicts broad prefix category, not full ICD code.
- No full ICD hierarchy or GNN.
- No multi-label full-document coding.
- Limited context from short literals.
- No comparison or ensemble with podium teams' models unless they become
  available and allowed.

## 11. Final Reflection

- The project connected course concepts to a real healthcare language problem.
- EDA and preprocessing mattered as much as modeling.
- Classical baselines remained useful even after RoBERTa.
- Language is central to healthcare records, so NLP systems must be useful,
  careful, and accountable.

## Suggested Timing

- Opening and task: 1 minute.
- EDA and challenges: 1.5 minutes.
- Methods and evolution: 2 minutes.
- Results and error analysis: 1.5 minutes.
- Hospital implications, limitations, reflection: 1.5 minutes.

# Short Presentation Archive

This folder documents the short presentation stage of Team 10's UAB-ASHO AI
Codification project.

The short presentation happened after the initial task understanding, EDA, and
baseline planning. At that point, the team had already started to understand the
competition as a short clinical-literal classification task rather than full
multi-label ICD coding from complete medical records.

## What Was Presented

The presentation focused on:

- the task formulation: predict one ICD category prefix, `y_category`;
- the annotation contract: `y_category` is the first character of `Code`;
- early EDA findings: short literals, class imbalance, duplicates, ambiguity,
  abbreviations, accents, punctuation, and digits;
- initial baseline ideas: majority baseline, TF-IDF character n-grams, word
  TF-IDF, and a future RoBERTa backbone;
- the project roadmap: EDA -> preprocessing -> baselines -> RoBERTa ->
  improvements -> evaluation -> submission.

After the presentation, the follow-up with Lei Kang confirmed that the team was
moving in a good direction. Around that stage, after the first model tests, the
team was around second in the competition.

## Slides

No short-presentation slide file is currently stored in this repository.

Manual slide files can be added here later, for example:

```text
presentations/short_presentation/team10_short_presentation.pdf
presentations/short_presentation/team10_short_presentation.pptx
```

Do not invent slide files in the report unless they are actually added to this
folder.

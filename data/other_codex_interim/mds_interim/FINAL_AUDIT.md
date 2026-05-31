# Final Quality Audit

Project: Fundamentals of Natural Language / NLP-I, Team 10, UAB-ASHO AI Codification  
Audit date: 2026-05-31  
Audited from repository root: `FNL-Project`

## Executive Status

The repository is ready for final academic review with documented limitations.
The final Kaggle upload file exists and now follows the strict two-column
submission contract: `id,y_category`.

Final selected model:

- `v10_vote_diverse_no_retrieval`
- validation accuracy: `0.579562`
- macro F1: `0.496677`
- weighted F1: `0.561294`
- best verified public Kaggle score recorded in repository files: `0.587`

No private/final Kaggle ranking is claimed because no verified private
leaderboard evidence is stored in the repository.

## Checklist

| Check | Status | Evidence |
|---|---|---|
| 1. No credentials | PASS | Tracked-file grep found no credential patterns. |
| 2. No passwords | PASS | Tracked-file grep found no `password`/`passwd` matches. |
| 3. No tokens | PASS | Tracked-file grep found no API token or bearer-token patterns. |
| 4. No private paths | PASS | Tracked-file grep found no `/home/`, `/Users/`, or `C:\Users` paths. |
| 5. No accidental huge files committed | PASS | Large local checkpoints exist under ignored `outputs/checkpoints/`; tracked largest files are small PDFs/tables. |
| 6. Data paths documented | PASS | `README.md` and `REPRODUCIBILITY.md` document `data/raw/`. |
| 7. README complete | PASS | README includes hook, team, competition, architecture, results, reproduction, ethics, report/presentation links. |
| 8. Notebooks complete/readable | PASS | 13 notebooks parsed as valid JSON. |
| 9. Core code modular | PASS | Reusable logic is in `src/`; model scripts are versioned under `models/`. |
| 10. Model versions runnable | PASS | Python syntax/import compilation passed for `src`, `scripts`, and `models`. |
| 11. Each model can produce submission | PASS | Model version interface and generated submissions exist; debug variants are clearly named. |
| 12. Final submission exists | PASS | `submissions/final_submission.csv` exists. |
| 13. Report exists | PASS | `reports/final_report.tex` and `reports/compiled/final_report.pdf` exist. |
| 14. Presentations folder exists | PASS | `presentations/short_presentation/` and `presentations/final_presentation/` exist. |
| 15. Reproducibility commands exist | PASS | `REPRODUCIBILITY.md` contains EDA, preprocessing, training, evaluation, submission, and report commands. |
| 16. Tests pass | PASS | `pytest`: 14 passed, 1 warning. |
| 17. Submission format valid | PASS | Final submission shape is `(6667, 2)` with columns `id,y_category`. |
| 18. Figures/tables referenced in report exist | PASS | 12 referenced report figures exist. |
| 19. No fabricated results | PASS | Report uses metrics from `outputs/metrics/` and score table; private ranking not claimed. |
| 20. Debug results clearly labeled | PASS | Debug artifacts include `debug`/`dry_run`; final tables/report use full-run metrics. |

## Commands Run

### Repository and Safety Checks

```bash
find . -maxdepth 3 -type f | sort
git status --short
find . -type f -size +50M -printf '%s %p\n' | sort -nr
git ls-files | while read f; do test -f "$f" && wc -c "$f"; done | sort -nr | head -30
git grep -n -F "/home/" -- . ':!FINAL_AUDIT.md' ':!*.pdf' ':!*.png' ':!*.jpg' ':!*.jpeg' ':!*.pt' ':!*.joblib'
git grep -n -F "C:\\Users" -- . ':!FINAL_AUDIT.md' ':!*.pdf' ':!*.png' ':!*.jpg' ':!*.jpeg' ':!*.pt' ':!*.joblib'
git grep -n -E "(password|passwd|api[_-]?key|secret|kaggle\.json|BEGIN (RSA|OPENSSH|PRIVATE) KEY|Bearer )" -- . ':!FINAL_AUDIT.md' ':!*.pdf' ':!*.png' ':!*.jpg' ':!*.jpeg' ':!*.pt' ':!*.joblib'
```

Result: PASS. The only large local artifacts are ignored checkpoints and local
LaTeX binaries. No credential or private-path matches remain in tracked text
files.

### Tests

```bash
pytest
```

Result: PASS.

```text
14 passed, 1 warning
```

### Python Import/Syntax Checks

```bash
python -m compileall src scripts models
```

Result: PASS.

### Data Validation

```bash
python scripts/analyze_data_annotations.py
```

Result: PASS with warnings.

Observed data:

- `data/raw/codification_data.csv`: `13700 x 2`
- `data/raw/leaderboard_data.csv`: `6667 x 2`
- `data/raw/icd_d_p_pairs.csv`: `179742 x 3`
- `data/processed/train_required_clean.csv`: `13700 x 5`
- `data/processed/leaderboard_required_clean.csv`: `6667 x 3`

Warnings are expected:

- leaderboard data has no `y_category`, which is normal for Kaggle test data;
- preprocessing ablation CSVs are non-competition CSVs and are reported with
  unknown schema warnings instead of failing the competition-data validation.

### Submission Validation

```bash
python -c "import pandas as pd; raw=pd.read_csv('data/raw/leaderboard_data.csv'); df=pd.read_csv('submissions/final_submission.csv'); assert list(df.columns)==['id','y_category']; assert len(df)==len(raw); assert set(df['id'])==set(raw['id']); print('PASS final submission', df.shape)"
```

Result: PASS.

```text
PASS final submission (6667, 2)
```

### Notebook Syntax Check

```bash
python -c "exec(\"\"\"import json, pathlib\nbad=[]\npaths=sorted(pathlib.Path('notebooks').glob('*.ipynb'))\nfor p in paths:\n    try:\n        nb=json.loads(p.read_text(encoding='utf-8'))\n        assert nb.get('nbformat')\n        assert isinstance(nb.get('cells'), list)\n    except Exception as e:\n        bad.append((str(p), repr(e)))\nprint(f'Checked {len(paths)} notebooks')\nassert not bad, bad\nprint('PASS notebook JSON syntax')\n\"\"\")"
```

Result: PASS.

```text
Checked 13 notebooks
PASS notebook JSON syntax
```

### Report Figure Reference Check

```bash
python -c "exec(\"\"\"from pathlib import Path\ntex=Path('reports/final_report.tex').read_text(encoding='utf-8')\nfigs=[]\nfor line in tex.splitlines():\n    if 'includegraphics' in line and '{' in line and '}' in line:\n        figs.append(line.split('{')[-1].split('}')[0])\nmissing=[str(Path('reports')/f) for f in figs if not (Path('reports')/f).exists()]\nprint('Figures referenced:', len(figs))\nassert not missing, missing\nprint('PASS all referenced figures exist')\n\"\"\")"
```

Result: PASS.

```text
Figures referenced: 12
PASS all referenced figures exist
```

### Final Evaluation

```bash
python -m src.evaluation
```

Result: PASS. Regenerated final comparison tables and figures under
`reports/tables/` and `reports/figures/`.

### LaTeX Compile

```bash
export PATH="$PWD/.texlive/2026/bin/x86_64-linux:$PATH"
cd reports
pdflatex -interaction=nonstopmode final_report.tex
bibtex final_report
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
mkdir -p compiled
cp final_report.pdf compiled/final_report.pdf
```

Result: PASS.

Compiled PDF:

```text
reports/compiled/final_report.pdf
```

The LaTeX log contains typographic overfull/underfull box warnings, but no fatal
errors and no unresolved TODO placeholders in `reports/final_report.tex`.

## Exact Final Commands

### Final Training / Artifact Reproduction

The final v10 ensemble depends on completed base-model predictions. To reproduce
the final candidate from scratch, run the base models first, then the ensemble
search:

```bash
python models/v01_tfidf_char_logreg.py
python models/v02_tfidf_word_svm.py
python models/v04_roberta_cls.py
python models/v08_roberta_mean_augmented.py
python models/v10_diverse_ensemble_search.py
```

The selected candidate is:

```text
v10_vote_diverse_no_retrieval
```

### Final Evaluation

```bash
python -m src.evaluation
```

### Final Submission File Creation

```bash
python - <<'PY'
import pandas as pd
sub = pd.read_csv('submissions/v10_vote_diverse_no_retrieval_kaggle.csv')
assert list(sub.columns) == ['id', 'Literal', 'y_category']
assert len(sub) == 6667
sub[['id', 'y_category']].to_csv('submissions/final_submission.csv', index=False)

detailed = pd.read_csv('outputs/predictions/v10_vote_diverse_no_retrieval_leaderboard_detailed.csv')
detailed.to_csv('outputs/predictions/final_leaderboard_detailed.csv', index=False)
PY
```

### Final Kaggle Submission

```bash
kaggle competitions submit \
  -c uab-asho-ai-codification \
  -f submissions/final_submission.csv \
  -m "Team 10 final public-best v10 diverse ensemble"
```

### Final Report Compile

```bash
export PATH="$PWD/.texlive/2026/bin/x86_64-linux:$PATH"
cd reports
pdflatex -interaction=nonstopmode final_report.tex
bibtex final_report
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
mkdir -p compiled
cp final_report.pdf compiled/final_report.pdf
```

## Known Limitations

- Private/final Kaggle ranking is not verified in repository files.
- Raw competition data is intentionally ignored by Git and must be placed in
  `data/raw/` before rerunning the full pipeline.
- Large local checkpoints exist under `outputs/checkpoints/`; they are ignored
  by `.gitignore` and should not be committed unless explicitly required.
- RoBERTa full training requires GPU time. If no GPU is available, use debug
  commands and rely on stored full-run artifacts only when they are present.
- The task predicts a broad ICD prefix, not the full ICD code.
- The system is an academic competition project and is not suitable for direct
  clinical use.
- The report compile has non-fatal typography warnings due to compact two-column
  layout and long model/path names.

## Manual Tasks Left

- Add verified private leaderboard/final ranking evidence if the team wants to
  claim final placement.
- Optionally add a Kaggle screenshot under `reports/figures/`.
- Decide whether the compiled final PDF should be committed or only submitted
  externally.
- Before pushing, confirm that ignored local folders such as `.texlive/`,
  `.local/`, `data/raw/`, and `outputs/checkpoints/` are not staged.

# Compiled Report

The final compiled PDF path is:

```text
reports/compiled/final_report.pdf
```

On 2026-05-31, TeX Live 2026 was installed locally inside the workspace under:

```text
.texlive/2026/
```

This local installation is ignored by Git. To use it in this VM, prepend its
binary directory to `PATH`:

```bash
export PATH=.texlive/2026/bin/x86_64-linux:$PATH
```

Compile from the repository root with:

```bash
cd reports
pdflatex final_report.tex
bibtex final_report || true
pdflatex final_report.tex
pdflatex final_report.tex
mkdir -p compiled
cp final_report.pdf compiled/final_report.pdf
```

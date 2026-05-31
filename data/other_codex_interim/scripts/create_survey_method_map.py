"""Create survey-to-project strategy artifacts.

This is conceptual grounding only. It does not train models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import REPORTS_DIR


TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def upsert_section(path: Path, title: str, body: str) -> None:
    """Create or replace a markdown section."""
    if not path.exists():
        path.write_text(f"# {path.stem.replace('_', ' ').title()}\n", encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    heading = f"## {title}"
    replacement = f"{heading}\n\n{body.strip()}\n"
    if heading not in content:
        path.write_text(content.rstrip() + "\n\n" + replacement, encoding="utf-8")
        return
    start = content.index(heading)
    next_heading = content.find("\n## ", start + len(heading))
    if next_heading == -1:
        updated = content[:start].rstrip() + "\n\n" + replacement
    else:
        updated = content[:start].rstrip() + "\n\n" + replacement + content[next_heading:]
    path.write_text(updated, encoding="utf-8")


def method_map() -> pd.DataFrame:
    """Return the survey method map table."""
    rows = [
        {
            "survey method family": "Rule-based ICD coding",
            "core idea": "Encode coding guidelines and keyword/rule patterns manually.",
            "course connection": "Linguistic rules, regular expressions, symbolic NLP.",
            "feasible in our assignment?": "Partly",
            "implementation decision": "Use only as conceptual baseline/future error-analysis aid; not a main model.",
            "expected benefit": "Transparent and easy to explain for frequent literal patterns.",
            "limitation": "Brittle, labor-intensive, poor coverage for noisy Spanish/Catalan literals.",
        },
        {
            "survey method family": "Majority and frequency baselines",
            "core idea": "Predict the most frequent class or use simple lookup frequencies.",
            "course connection": "Baseline design and empirical evaluation from ML.",
            "feasible in our assignment?": "Yes",
            "implementation decision": "Implement majority baseline before stronger models.",
            "expected benefit": "Lower bound that shows whether later models really learn signal.",
            "limitation": "Ignores text and fails rare categories.",
        },
        {
            "survey method family": "Traditional ML with BoW/TF-IDF",
            "core idea": "Represent text with sparse lexical features and train a classifier.",
            "course connection": "Basic Text Processing, vectorization, supervised classification.",
            "feasible in our assignment?": "Yes",
            "implementation decision": "Implement TF-IDF word n-gram and character n-gram baselines.",
            "expected benefit": "Strong, fast, interpretable baseline for short clinical literals.",
            "limitation": "Limited semantics; sensitive to preprocessing and ambiguity.",
        },
        {
            "survey method family": "Nearest-neighbor / retrieval methods",
            "core idea": "Assign labels from similar known literals or descriptions.",
            "course connection": "Vector spaces, similarity, information retrieval.",
            "feasible in our assignment?": "Yes, if scoped",
            "implementation decision": "Consider fuzzy matching or nearest-neighbor lookup after baselines.",
            "expected benefit": "Useful because many leaderboard literals overlap train after normalization.",
            "limitation": "Weak for truly unseen synonyms and ambiguous repeated literals.",
        },
        {
            "survey method family": "CNN/RNN neural models",
            "core idea": "Learn dense representations from token sequences with neural encoders.",
            "course connection": "Neural Networks and sequence modeling.",
            "feasible in our assignment?": "Not priority",
            "implementation decision": "Do not implement separately; use PLM backbone instead.",
            "expected benefit": "Can learn local phrase patterns.",
            "limitation": "Extra engineering for limited gain on very short literals.",
        },
        {
            "survey method family": "GNN / hierarchy-aware methods",
            "core idea": "Use ICD hierarchy, code relationships, or graph structure.",
            "course connection": "Structured prediction, graph representations, knowledge-based NLP.",
            "feasible in our assignment?": "Future work",
            "implementation decision": "Do not implement in final scope unless time remains.",
            "expected benefit": "Could exploit parent-child inheritance and related codes.",
            "limitation": "Our target is only first-character category; full hierarchy modeling is overkill.",
        },
        {
            "survey method family": "PLM / Transformer methods",
            "core idea": "Fine-tune a pretrained language model for clinical text classification.",
            "course connection": "Transformers, pretrained language models, tokenization.",
            "feasible in our assignment?": "Yes",
            "implementation decision": "Use Spanish biomedical-clinical RoBERTa with pooling ablations.",
            "expected benefit": "Better representation of Spanish biomedical terms and abbreviations.",
            "limitation": "May overfit small/ambiguous data; needs careful validation and compute.",
        },
        {
            "survey method family": "Class imbalance strategies",
            "core idea": "Use class weights, sampling, or thresholding to handle rare labels.",
            "course connection": "Evaluation metrics, imbalanced classification.",
            "feasible in our assignment?": "Yes",
            "implementation decision": "Test class weighting as an ablation, not the default assumption.",
            "expected benefit": "May improve rare-category behavior.",
            "limitation": "May reduce strict accuracy, the competition metric.",
        },
        {
            "survey method family": "Pooling and representation variants",
            "core idea": "Compare CLS pooling, mean pooling, and possibly ensemble representations.",
            "course connection": "Representation learning and neural architecture choices.",
            "feasible in our assignment?": "Yes",
            "implementation decision": "Compare RoBERTa CLS vs mean pooling.",
            "expected benefit": "Small architecture changes may matter for short literals.",
            "limitation": "Requires controlled experiments to avoid overclaiming.",
        },
        {
            "survey method family": "Ensembles and confidence/error analysis",
            "core idea": "Combine complementary systems and inspect prediction confidence/errors.",
            "course connection": "Model comparison, generalization, interpretability.",
            "feasible in our assignment?": "Yes, simple version",
            "implementation decision": "Use simple ensembling only after individual models are validated.",
            "expected benefit": "Can combine lexical memorization and neural generalization.",
            "limitation": "More moving parts; must remain reproducible and explainable.",
        },
        {
            "survey method family": "Label-description matching",
            "core idea": "Use official ICD descriptions as label text for matching or augmentation.",
            "course connection": "Semantic similarity, retrieval, weak supervision.",
            "feasible in our assignment?": "Future work / optional",
            "implementation decision": "Keep as future work unless baselines prove stable.",
            "expected benefit": "Could help rare categories and unseen literals.",
            "limitation": "Official descriptions differ strongly from short clinical literals.",
        },
        {
            "survey method family": "Full multi-label ICD prediction",
            "core idea": "Predict all ICD codes assigned to a clinical document.",
            "course connection": "Multi-label classification and hierarchical evaluation.",
            "feasible in our assignment?": "No",
            "implementation decision": "Explicitly out of scope; our target is one category prefix.",
            "expected benefit": "Closer to real hospital ICD coding.",
            "limitation": "The Kaggle assignment does not provide full-document multi-label targets.",
        },
    ]
    return pd.DataFrame(rows)


def plot_timeline(path: Path) -> None:
    """Create a compact method-evolution timeline."""
    stages = pd.DataFrame(
        [
            {"year": 1995, "stage": "Rule-based", "note": "Manual guidelines\nand keyword rules"},
            {"year": 2007, "stage": "Traditional ML", "note": "BoW, TF-IDF,\nSVM/NB/KNN"},
            {"year": 2017, "stage": "CNN/RNN", "note": "Neural encoders\nfor clinical text"},
            {"year": 2019, "stage": "GNN/knowledge", "note": "Hierarchy and\ncode relations"},
            {"year": 2020, "stage": "PLM/Transformers", "note": "BERT/RoBERTa\nfine-tuning"},
            {"year": 2026, "stage": "Our project", "note": "Category prefix\nfrom short literals"},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 4.4))
    plt.hlines(1, stages["year"].min() - 2, stages["year"].max() + 2, color="#b7b7b7", linewidth=2)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2"]
    for idx, row in stages.iterrows():
        plt.scatter(row["year"], 1, s=260, color=colors[idx], zorder=3)
        plt.text(row["year"], 1.18, row["stage"], ha="center", va="bottom", fontsize=10, weight="bold")
        plt.text(row["year"], 0.62, row["note"], ha="center", va="top", fontsize=9)
    plt.title("Evolution of automated ICD coding methods and our scoped assignment")
    plt.yticks([])
    plt.xlabel("Approximate period")
    plt.xlim(stages["year"].min() - 3, stages["year"].max() + 3)
    plt.subplots_adjust(top=0.78, bottom=0.30, left=0.04, right=0.98)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table = method_map()
    table_path = TABLES_DIR / "survey_method_map.csv"
    figure_path = FIGURES_DIR / "fig_09_method_evolution_timeline.png"
    table.to_csv(table_path, index=False)
    plot_timeline(figure_path)

    body = """
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
"""
    upsert_section(PROJECT_ROOT / "REPORT_NOTES.md", "Survey-to-Project Method Strategy", body)
    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

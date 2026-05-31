"""Create clean architecture diagrams for the final report and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


COLORS = {
    "data": "#E8F1FA",
    "analysis": "#EAF6EF",
    "model": "#F1ECFA",
    "output": "#FFF4DE",
    "feedback": "#FDEEEF",
    "stroke": "#243447",
    "muted": "#607080",
}


def box(ax, xy, wh, text, facecolor, fontsize=10, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=lw,
        edgecolor=COLORS["stroke"],
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17202A",
        linespacing=1.15,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.35, rad=0.0, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["stroke"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def setup_canvas(width=15, height=7.8):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_title(ax, title, subtitle):
    ax.text(
        0.5,
        0.955,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#14213D",
    )
    ax.text(
        0.5,
        0.913,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def diagram_v1():
    """Linear project pipeline with a report feedback loop."""
    fig, ax = setup_canvas()
    add_title(
        ax,
        "UAB-ASHO ICD-10 Category Codification Pipeline",
        "From raw clinical literals to validation evidence, Kaggle submission, notebooks, and final report",
    )

    y = 0.55
    w, h = 0.118, 0.14
    xs = [0.045, 0.18, 0.315, 0.45, 0.585, 0.72, 0.855]
    labels = [
        "Raw CSV files\ncodification_data\nleaderboard_data\nICD catalogue",
        "EDA\nschemas, labels,\nimbalance, duplicates",
        "Preprocessing\nstrip + collapse\nwhitespace only",
        "Tokenizer\nSpanish biomedical\nRoBERTa subwords",
        "RoBERTa encoder\nclinical contextual\nrepresentations",
        "Pooling + head\nCLS / mean\nlinear classifier",
        "Predictions\nvalidation +\nleaderboard",
    ]
    colors = [
        COLORS["data"],
        COLORS["analysis"],
        COLORS["analysis"],
        COLORS["model"],
        COLORS["model"],
        COLORS["model"],
        COLORS["output"],
    ]

    centers = []
    for x, label, c in zip(xs, labels, colors):
        box(ax, (x, y), (w, h), label, c, fontsize=9.1)
        centers.append((x + w / 2, y + h / 2))

    for i in range(len(centers) - 1):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    box(ax, (0.19, 0.23), (0.18, 0.13), "Validation metrics\naccuracy, macro-F1,\nper-class errors", COLORS["output"], fontsize=9.3)
    box(ax, (0.43, 0.23), (0.18, 0.13), "Kaggle submission\nfinal_submission.csv\nid, y_category", COLORS["output"], fontsize=9.3)
    box(ax, (0.67, 0.23), (0.20, 0.13), "Report + notebooks\nexplain decisions\nand limitations", COLORS["feedback"], fontsize=9.3)

    arrow(ax, (0.91, y), (0.52, 0.36), rad=0.08)
    arrow(ax, (0.52, 0.36), (0.28, 0.36))
    arrow(ax, (0.61, 0.295), (0.67, 0.295))
    arrow(ax, (0.77, 0.36), (0.24, 0.55), color="#B13E53", rad=0.24)

    ax.text(0.52, 0.14, "The feedback loop keeps the repository traceable: results update notebooks, report notes, decisions, and the final model choice.", ha="center", fontsize=9.5, color=COLORS["muted"])
    return fig


def diagram_v2():
    """Layered architecture view with classical and RoBERTa evidence."""
    fig, ax = setup_canvas()
    add_title(
        ax,
        "Final System View",
        "Data understanding feeds preprocessing; preprocessing feeds models; evaluation feeds submission and reporting",
    )

    # Top data layer
    box(ax, (0.055, 0.75), (0.22, 0.11), "Raw CSV files\ntraining, leaderboard, ICD catalogue", COLORS["data"], fontsize=10)
    box(ax, (0.39, 0.75), (0.22, 0.11), "EDA + annotation contract\n36 labels, duplicates, imbalance", COLORS["analysis"], fontsize=10)
    box(ax, (0.725, 0.75), (0.22, 0.11), "Required-clean text\npreserve case, accents,\npunctuation, digits", COLORS["analysis"], fontsize=10)
    arrow(ax, (0.275, 0.805), (0.39, 0.805))
    arrow(ax, (0.61, 0.805), (0.725, 0.805))

    # Model layer
    box(ax, (0.12, 0.50), (0.24, 0.13), "Classical baselines\nmajority, char TF-IDF,\nword TF-IDF, retrieval", COLORS["model"], fontsize=10)
    box(ax, (0.44, 0.50), (0.24, 0.13), "RoBERTa path\ntokenizer -> encoder\nCLS / mean pooling", COLORS["model"], fontsize=10)
    box(ax, (0.72, 0.50), (0.18, 0.13), "Classifier head\nsoftmax over\n36 categories", COLORS["model"], fontsize=10)
    arrow(ax, (0.835, 0.75), (0.56, 0.63), rad=0.06)
    arrow(ax, (0.835, 0.75), (0.24, 0.63), rad=0.12)
    arrow(ax, (0.68, 0.565), (0.72, 0.565))

    # Evaluation/output layer
    box(ax, (0.08, 0.25), (0.22, 0.12), "Validation metrics\naccuracy, macro-F1,\nconfusions, confidence", COLORS["output"], fontsize=9.8)
    box(ax, (0.39, 0.25), (0.22, 0.12), "Diverse ensemble\nRoBERTa + TF-IDF\nmodel agreement", COLORS["output"], fontsize=9.8)
    box(ax, (0.70, 0.25), (0.22, 0.12), "Kaggle submission\nfinal_submission.csv\nid, y_category", COLORS["output"], fontsize=9.8)
    arrow(ax, (0.24, 0.50), (0.19, 0.37), rad=-0.03)
    arrow(ax, (0.56, 0.50), (0.50, 0.37), rad=-0.03)
    arrow(ax, (0.81, 0.50), (0.50, 0.37), rad=0.05)
    arrow(ax, (0.61, 0.31), (0.70, 0.31))

    box(ax, (0.28, 0.06), (0.44, 0.10), "Report + narrative notebooks + decision log\nwhat we expected -> what we found -> how it changed the next step", COLORS["feedback"], fontsize=10)
    arrow(ax, (0.19, 0.25), (0.38, 0.16), rad=-0.08)
    arrow(ax, (0.50, 0.25), (0.50, 0.16))
    arrow(ax, (0.81, 0.25), (0.62, 0.16), rad=0.08)
    arrow(ax, (0.50, 0.16), (0.50, 0.75), color="#B13E53", rad=-0.42)

    return fig


def final_diagram():
    """Selected final diagram: clean layered pipeline with explicit feedback loop."""
    fig, ax = setup_canvas(width=16, height=8.8)
    add_title(
        ax,
        "Final Architecture: ICD-10 Category Codification from Clinical Literals",
        "Raw data, EDA, conservative preprocessing, biomedical RoBERTa, validation evidence, Kaggle submission, and report feedback",
    )

    # Data and design layer
    box(ax, (0.05, 0.74), (0.23, 0.11), "Raw CSV files\ncodification_data.csv\nleaderboard_data.csv\nicd_d_p_pairs.csv", COLORS["data"], fontsize=9.6)
    box(ax, (0.385, 0.74), (0.23, 0.11), "EDA\nschema validation, labels,\nimbalance, duplicates,\ntext patterns", COLORS["analysis"], fontsize=9.6)
    box(ax, (0.72, 0.74), (0.23, 0.11), "Required preprocessing\nstrip + collapse spaces\npreserve accents, case,\npunctuation, digits", COLORS["analysis"], fontsize=9.6)
    arrow(ax, (0.28, 0.795), (0.385, 0.795))
    arrow(ax, (0.615, 0.795), (0.72, 0.795))

    # Model layer
    box(ax, (0.08, 0.50), (0.21, 0.11), "Classical baselines\nmajority, char TF-IDF,\nword TF-IDF, retrieval", COLORS["model"], fontsize=9.4)
    box(ax, (0.36, 0.50), (0.16, 0.11), "Tokenization\nbiomedical Spanish\nRoBERTa subwords", COLORS["model"], fontsize=9.4)
    box(ax, (0.56, 0.50), (0.16, 0.11), "RoBERTa encoder\ncontextual clinical\nrepresentations", COLORS["model"], fontsize=9.4)
    box(ax, (0.76, 0.50), (0.16, 0.11), "CLS / mean pooling\nclassifier head\n36-way softmax", COLORS["model"], fontsize=9.4)
    arrow(ax, (0.835, 0.74), (0.44, 0.61), rad=0.04)
    arrow(ax, (0.835, 0.74), (0.185, 0.61), rad=0.13)
    arrow(ax, (0.52, 0.555), (0.56, 0.555))
    arrow(ax, (0.72, 0.555), (0.76, 0.555))

    # Evidence and output layer
    box(ax, (0.08, 0.27), (0.22, 0.11), "Validation metrics\naccuracy, macro-F1,\nweighted-F1, confusions,\nconfidence/error analysis", COLORS["output"], fontsize=9.1)
    box(ax, (0.39, 0.27), (0.22, 0.11), "Final ensemble\ncombine RoBERTa and\nTF-IDF model families\nby complementary errors", COLORS["output"], fontsize=9.1)
    box(ax, (0.70, 0.27), (0.22, 0.11), "Kaggle submission\nfinal_submission.csv\ncolumns: id, y_category", COLORS["output"], fontsize=9.1)
    arrow(ax, (0.185, 0.50), (0.19, 0.38))
    arrow(ax, (0.84, 0.50), (0.50, 0.38), rad=0.08)
    arrow(ax, (0.29, 0.325), (0.39, 0.325))
    arrow(ax, (0.61, 0.325), (0.70, 0.325))

    # Narrative feedback layer
    box(ax, (0.25, 0.075), (0.50, 0.095), "Report, narrative notebooks, decision log, and experiment log\nWhat we expected -> what we found -> how it changed the next step", COLORS["feedback"], fontsize=9.5)
    arrow(ax, (0.19, 0.27), (0.37, 0.17), rad=-0.08, color="#B13E53")
    arrow(ax, (0.50, 0.27), (0.50, 0.17), color="#B13E53")
    arrow(ax, (0.81, 0.27), (0.63, 0.17), rad=0.08, color="#B13E53")
    arrow(ax, (0.25, 0.12), (0.40, 0.74), rad=0.40, color="#B13E53")

    return fig


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save(diagram_v1(), FIGURES_DIR / "architecture_diagram_v1.png")
    save(diagram_v2(), FIGURES_DIR / "architecture_diagram_v2.png")
    save(final_diagram(), FIGURES_DIR / "final_architecture_diagram.png")
    print(f"Wrote {FIGURES_DIR / 'architecture_diagram_v1.png'}")
    print(f"Wrote {FIGURES_DIR / 'architecture_diagram_v2.png'}")
    print(f"Wrote {FIGURES_DIR / 'final_architecture_diagram.png'}")


if __name__ == "__main__":
    main()

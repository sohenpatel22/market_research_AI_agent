"""Quality-versus-cost chart for a provider comparison.

Usage:
    uv run python -m market_research_agent.eval.plot_comparison --n 69
    (writes docs/images/provider-frontier.png; override with --out)
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from market_research_agent.eval.compare import load_runs


def comparison_figure(table: pd.DataFrame, n: int) -> Figure:
    """Faithfulness (y) against dollars per question on a log axis (x), one point per model."""
    part = table[table["n"] == n].dropna(subset=["faithfulness", "$/question"])
    fig = Figure(figsize=(7.5, 4.5))
    ax = fig.subplots()
    for label, row in part.iterrows():
        model = label.split("(")[1].rstrip(")").split("/")[-1]
        ax.scatter(row["$/question"], row["faithfulness"], s=90, zorder=3)
        ax.annotate(
            model,
            (row["$/question"], row["faithfulness"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per question, USD (log scale; includes the grading call)")
    ax.set_ylabel("RAGAS faithfulness (judge: DeepSeek)")
    ax.set_title(f"Quality vs cost: same {n} golden questions, same agent code")
    ax.grid(True, which="both", alpha=0.3, zorder=0)
    fig.text(
        0.5,
        0.005,
        "Faithfulness is measured on 20 filing questions (one run per model): gaps of a few "
        "hundredths are within noise.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="question-set size to plot")
    parser.add_argument("--out", type=Path, default=Path("docs/images/provider-frontier.png"))
    args = parser.parse_args()
    table = load_runs()
    if (table["n"] == args.n).sum() < 2:
        raise SystemExit(f"Need at least two runs over {args.n} questions.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    comparison_figure(table, args.n).savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

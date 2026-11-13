#!/usr/bin/env python3
"""Render Figure 1 from computed results without hardcoded statistical counts.

Usage:
  python make_figure.py --results results/results.json --output figures

Requires matplotlib. Produces Figure_1.png, Figure_1.pdf and Figure_1.svg.
Panel A uses question-assessor-pair comparisons; panel B uses question-gate
units. Counts are printed inside bars, whose lengths represent percentages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=base / "results" / "results.json")
    parser.add_argument("--output", type=Path, default=base / "figures")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    strata = ["Overall", "Limited", "Stronger"]
    rows_a, rows_b = [], []
    for stratum in strata:
        a = results["pair_vector_diagnostics"][stratum]
        row_a = [a["identical_vectors"], a["matching_outcomes_with_different_vectors"],
                 a["different_outcomes"]]
        if sum(row_a) != a["pairs"]:
            raise ValueError(f"Panel A counts do not reconcile for {stratum}")
        if row_a[0] + row_a[1] != a["matching_outcomes"]:
            raise ValueError(f"Matching outcomes do not reconcile for {stratum}")
        b = results["disagreement"][stratum]["question_gate_units"]
        row_b = [b.get("Unanimous", 0), b.get("Pass/Not required only", 0),
                 b.get("Other disagreement", 0)]
        if any(v < 0 for v in row_a + row_b) or min(sum(row_a), sum(row_b)) <= 0:
            raise ValueError(f"Invalid counts for {stratum}")
        rows_a.append(row_a)
        rows_b.append(row_b)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.4), gridspec_kw={"hspace": .88})
    colors = ["#355F7A", "#D2A35B", "#DADFE3"]
    labels_a = ["Identical gate vectors", "Different vectors, same category", "Different final categories"]
    labels_b = ["Unanimous gate states", "Pass / Not required only", "Other disagreement"]
    titles = ["A  Question–assessor-pair comparisons", "B  Question–gate units"]
    for ax, rows, labels, title in zip(axes, [rows_a, rows_b], [labels_a, labels_b], titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=30, fontsize=12)
        for yi, counts in enumerate(rows):
            denominator, left = sum(counts), 0
            for j, count in enumerate(counts):
                width = 100 * count / denominator
                ax.barh(yi, width, left=left, height=.58, color=colors[j],
                        edgecolor="white", linewidth=.6, label=labels[j] if yi == 0 else None)
                if count:
                    ax.text(left + width / 2, yi, str(count), ha="center", va="center",
                            fontsize=10, color="white" if j == 0 else "#111111")
                left += width
        ax.set_yticks(range(len(strata)), [f"{s}  (n = {sum(r)})" for s, r in zip(strata, rows)])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#DDE2E5", linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04), frameon=False,
                  fontsize=8.5, handlelength=1.1, columnspacing=1.2, borderaxespad=0)
    fig.subplots_adjust(left=.22, right=.98, top=.87, bottom=.07)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(args.output / f"Figure_1.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"source": str(args.results), "output_directory": str(args.output),
                      "matplotlib": matplotlib.__version__}, indent=2))


if __name__ == "__main__":
    main()

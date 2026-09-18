"""Rebuild the three-panel reliability figure from the original summary.

Contract: compare mean empirical versus nominal coverage for T, TD, P;
retain all six levels per method and the original method colors. Export a
static PNG/PDF with larger type, a shared legend, panel letters, and no titles.
Panel order: (a) T, (b) TD, (c) P. No data are re-estimated.
"""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


OUTPUT = Path(__file__).resolve().parent
METHODS = (
    ("uncalibrated", "Raw uncertainty", "#d62728", "o"),
    ("holdout", "Split CP", "#2ca02c", "s"),
    ("cross_fitted", "CFSC", "#ff7f0e", "^"),
)


def main():
    with (OUTPUT.parent / "reliability_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 21,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.85, wspace=0.12)
    for ax, target, letter in zip(axes, ("T", "TD", "P"), "abc"):
        ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=2.2, label="Ideal")
        for method, label, color, marker in METHODS:
            values = sorted(
                (row for row in rows if row["target"] == target
                 and row["calibration_method"] == method),
                key=lambda row: float(row["nominal_coverage"]),
            )
            assert len(values) == 6, (target, method, len(values))
            x = [float(row["nominal_coverage"]) for row in values]
            y = [float(row["mean"]) for row in values]
            ax.plot(x, y, color=color, marker=marker, linewidth=2.5,
                    markersize=8, label=label)
        ax.set(xlim=(0.45, 1.0), ylim=(0.45, 1.0), xlabel="Nominal coverage")
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=5, width=1.1)
        ax.text(0.5, -0.25, f"({letter})", transform=ax.transAxes,
                ha="center", va="top", fontsize=24)
    axes[0].set_ylabel("Empirical coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, columnspacing=2.0, handlelength=2.4)
    assert all(not ax.get_title() for ax in axes)
    for extension in ("png", "pdf"):
        path = OUTPUT / f"reliability_combined_abc.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

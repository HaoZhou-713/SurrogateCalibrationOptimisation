"""Extend the Li notebook's dataset/hull/inset plot to all three frameworks.

Uses saved results only. The observed-data hull and zoom limits follow
CaseStudy_battery_Li.ipynb cells 35/36. No PF filtering or fitting is performed.
Run this script directly; default paths are relative to this file.
"""
from pathlib import Path
import argparse
import hashlib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
PANELS = (("raw", "Raw"), ("raw_uncertainty", "Raw + uncertainty"), ("cal", "CFSC"))


def plot_comparison(store, observed):
    """Return the figure, retaining every saved solution in main and inset axes."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    all_points = [observed]
    for key, _ in PANELS:
        if not store[key]:
            raise ValueError(f"Empty PF group: {key}")
        for seed, xp, yp in store[key]:
            yp = np.asarray(yp, dtype=float)
            if yp.ndim != 2 or yp.shape[1] != 2 or not len(yp) or not np.isfinite(yp).all():
                raise ValueError(f"Invalid saved objectives for {key}, seed {seed}")
            all_points.append(yp)
    combined = np.concatenate(all_points)
    lower, upper = combined.min(axis=0), combined.max(axis=0)
    padding = .03 * (upper - lower + 1e-12)
    # Build once from all observed data; reuse the identical polygon in insets.
    unique_observed = np.unique(observed, axis=0)
    hull = ConvexHull(unique_observed)
    polygon = unique_observed[hull.vertices]
    seeds = sorted({run[0] for key, _ in PANELS for run in store[key]})
    seed_colors = {seed: plt.get_cmap("tab10")(i % 10) for i, seed in enumerate(seeds)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.9))
    fig.subplots_adjust(left=.045, right=.99, bottom=.15, top=.89, wspace=.20)

    def draw(ax, runs, inset=False):
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor="lightgrey", edgecolor="0.65",
                alpha=.35, linewidth=1., zorder=0)
        ax.scatter(observed[:, 0], observed[:, 1], c="0.35", s=18 if not inset else 25,
                   alpha=.48, zorder=3 if inset else 1, label="Training dataset")
        for seed, xp, yp in runs:
            yp = np.asarray(yp)
            artist = ax.scatter(yp[:, 0], yp[:, 1], color=seed_colors[seed],
                                s=10, alpha=.28, linewidths=0, zorder=2)
            # Guard against accidental row filtering when this script is revised.
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), yp)
        ax.grid(alpha=.15 if inset else .20)

    for index, (ax, (key, title)) in enumerate(zip(axes, PANELS)):
        runs = store[key]
        draw(ax, runs)
        # Display the volume scale explicitly requested by the user;
        # retain the saved numerical coordinates without conversion.
        ax.set(title=f"({chr(97 + index)}) {title}", xlabel=r"V ($\times 10^{-3}$ m$^3$)", ylabel="TD (K)",
               xlim=(lower[0] - padding[0], upper[0] + padding[0]),
               ylim=(lower[1] - padding[1], upper[1] + padding[1]))
        ax.legend(frameon=False, loc="upper left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        axins = ax.inset_axes([.55, .065, .42, .42])
        draw(axins, runs, inset=True)
        axins.set(xlim=(2.9, 4.2), ylim=(.65, 1.45))
        axins.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axins.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axins.tick_params(labelsize=11)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-store", type=Path, default=HERE / "li_pf_store.pkl")
    parser.add_argument("--observed", type=Path, default=HERE.parents[1] / "battery_data_Li.xlsx")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    digest = hashlib.sha256(args.pf_store.read_bytes()).hexdigest()
    with args.pf_store.open("rb") as stream:
        store = pickle.load(stream)["pf_store"]
    observed = pd.read_excel(args.observed)[["V", "TD"]].to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("Observed data contain missing/nonfinite values")
    print(f"Observed rows: {len(observed)}; objective order: V, TD")
    for key, title in PANELS:
        print(f"{title}: {len(store[key])} runs, {sum(len(r[2]) for r in store[key])} saved points")
    fig = plot_comparison(store, observed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = args.output_dir / f"li_pf_three_frameworks_with_zoom.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=.08)
        print(path.resolve())
    plt.close(fig)
    assert hashlib.sha256(args.pf_store.read_bytes()).hexdigest() == digest
    print("Verified: all saved rows plotted in both main panels and insets; input pickle unchanged.")


if __name__ == "__main__":
    main()

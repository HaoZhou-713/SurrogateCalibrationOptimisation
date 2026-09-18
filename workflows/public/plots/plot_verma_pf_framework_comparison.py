"""Publication figures and geometric summaries from saved Verma PF runs.

No optimisation, model fitting, Pareto filtering, coordinate averaging, or
downsampling occurs here. Y_pareto contains predictive means of recommendations
selected in each framework's original three-objective optimisation space.

Usage (from any working directory)::

    python plot_verma_pf_framework_comparison.py --dpi 600

Requires numpy, pandas, scipy, scikit-learn, matplotlib and openpyxl.
The pickle input is the user-provided, trusted local results store.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TARGETS = ("T", "P", "TD")
LABELS = ("T (K)", "P (Pa)", "TD (K)")
PAIRS = ((0, 1), (0, 2), (1, 2))
# Preserve PF_METHOD_COLORS in generate_section6_public_btms_figures.py,
# adding its conventional orange for the additional uncertainty comparator.
METHODS = {
    "raw": ("Raw predictive mean", "#3B3B3B", "o"),
    "raw_uncertainty": ("Raw + GP uncertainty", "#FF7F0E", "^"),
    "cfsc": ("CFSC", "#9467BD", "s"),
}
HULL_TOL = 1e-10


@dataclass(frozen=True)
class FrontRun:
    run_id: int
    x: np.ndarray
    y: np.ndarray


def load_fronts(path: Path) -> tuple[dict[str, list[FrontRun]], dict]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    store = payload["pf_store"]
    if set(store) != set(METHODS):
        raise ValueError(f"Expected method keys {tuple(METHODS)}, found {tuple(store)}")
    result = {}
    for method in METHODS:
        runs = []
        for run_id, xp, yp in store[method]:
            x, y = np.asarray(xp, dtype=float), np.asarray(yp, dtype=float)
            if y.ndim != 2 or y.shape[1] != 3 or len(y) == 0:
                raise ValueError(f"{method}/{run_id}: expected a nonempty N x 3 front")
            if x.ndim != 2 or len(x) != len(y) or not np.isfinite(y).all():
                raise ValueError(f"{method}/{run_id}: inconsistent arrays or nonfinite objectives")
            # No deduplication: preserve every saved row and its original order.
            runs.append(FrontRun(int(run_id), x, y))
        if not runs or len({r.run_id for r in runs}) != len(runs):
            raise ValueError(f"{method}: empty run list or duplicate run identifiers")
        result[method] = sorted(runs, key=lambda r: r.run_id)
    return result, payload.get("metadata", {})


def symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Exact symmetric Hausdorff distance, with Euclidean nearest neighbours."""
    a_to_b = cKDTree(b).query(a, k=1)[0].max()
    b_to_a = cKDTree(a).query(b, k=1)[0].max()
    return float(max(a_to_b, b_to_a))


def select_medoid(fronts: list[np.ndarray]) -> tuple[int, float, np.ndarray]:
    """Return actual run index, mean unordered-pair distance, and full matrix.

    Excludes self-distances from both scores and dispersion. Exact medoid ties
    resolve to the first run (the loader sorts runs by identifier).
    """
    if not fronts:
        raise ValueError("At least one nonempty front is required")
    n = len(fronts)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = symmetric_hausdorff(fronts[i], fronts[j])
    if n == 1:
        return 0, float("nan"), distances
    scores = distances.sum(axis=1) / (n - 1)
    return int(np.argmin(scores)), float(distances[np.triu_indices(n, 1)].mean()), distances


def outside_support(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """Notebook's convex-hull half-space membership rule and tolerance."""
    return np.any(points @ hull.equations[:, :-1].T + hull.equations[:, -1] > HULL_TOL, axis=1)


def distance_to_support(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """Exact Euclidean distance to a full-dimensional 3D convex polytope.

    Inputs must already share the chosen normalized coordinates. Interior
    points have distance zero. For outside points, minimize over triangular
    hull faces, including each face interior, its edges and its vertices.
    This is NOT nearest-observation distance or maximum half-space violation.
    """
    distances = np.zeros(len(points))
    outside = outside_support(points, hull)
    query = points[outside]
    if not len(query):
        return distances
    best_squared = np.full(len(query), np.inf)
    for triangle in hull.points[hull.simplices]:
        a, b, c = triangle
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        normal /= np.linalg.norm(normal)
        signed = (query - a) @ normal
        projected = query - signed[:, None] * normal
        rel = projected - a
        d00, d01, d11 = ab @ ab, ab @ ac, ac @ ac
        denominator = d00 * d11 - d01 * d01
        if denominator > 0:
            u = (d11 * (rel @ ab) - d01 * (rel @ ac)) / denominator
            v = (d00 * (rel @ ac) - d01 * (rel @ ab)) / denominator
            on_face = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
            best_squared = np.minimum(best_squared, np.where(on_face, signed ** 2, np.inf))
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            t = np.clip(((query - start) @ edge) / (edge @ edge), 0., 1.)
            delta = query - (start + t[:, None] * edge)
            best_squared = np.minimum(best_squared, np.einsum("ij,ij->i", delta, delta))
    distances[outside] = np.sqrt(best_squared)
    return distances


def local_support_union_3d(observed: np.ndarray, runs: list[FrontRun], k: int = 5) -> np.ndarray:
    """Port of notebook cell 31, preserving physical-space kNN selection.

    This method-specific local support is used ONLY in the supplementary
    diagnostic. Quantitative support uses the notebook's full observed hull.
    """
    neighbours = NearestNeighbors(n_neighbors=k).fit(observed)
    indices = [neighbours.kneighbors(r.y, return_distance=False).ravel() for r in runs]
    return observed[np.unique(np.concatenate(indices))]


def summarize(fronts, observed):
    pool = np.concatenate([r.y for runs in fronts.values() for r in runs])
    lo, hi = pool.min(axis=0), pool.max(axis=0)
    span = hi - lo
    if np.any(span <= 0):
        raise ValueError("Pooled PF objective ranges must be positive in all three dimensions")
    normalized_hull = ConvexHull((observed - lo) / span)
    physical_hull = ConvexHull(observed)
    representatives, summary_rows, run_rows, pair_rows, point_frames = {}, [], [], [], []
    for method, runs in fronts.items():
        normalized = [(r.y - lo) / span for r in runs]
        medoid_index, dispersion, matrix = select_medoid(normalized)
        representatives[method] = runs[medoid_index]
        all_distances, negative_rates, outside_rates = [], [], []
        for i, (run, yn) in enumerate(zip(runs, normalized)):
            # Physical-space membership exactly preserves the notebook's rate.
            outside = outside_support(run.y, physical_hull)
            if not np.array_equal(outside, outside_support(yn, normalized_hull)):
                raise ValueError("Physical/normalized hull membership differs at the tolerance boundary")
            distance = distance_to_support(yn, normalized_hull)
            negative = run.y[:, 1] < 0
            negative_rates.append(float(negative.mean()))
            outside_rates.append(float(outside.mean()))
            all_distances.append(distance)
            run_rows.append({
                "method": method, "run_id": run.run_id, "n_pf_points": len(run.y),
                "is_representative": i == medoid_index,
                "mean_hausdorff_to_other_runs": matrix[i].sum() / (len(runs) - 1) if len(runs) > 1 else np.nan,
                "negative_pressure_rate": negative_rates[-1], "outside_support_rate": outside_rates[-1],
                "mean_distance_to_support": float(distance.mean()),
            })
            point_frames.append(pd.DataFrame({
                "method": method, "run_id": run.run_id, "point_index": np.arange(len(run.y)),
                "T": run.y[:, 0], "P": run.y[:, 1], "TD": run.y[:, 2],
                "negative_pressure": negative, "outside_support": outside,
                "distance_to_support_normalized_3d": distance,
            }))
            for j in range(i + 1, len(runs)):
                pair_rows.append({"method": method, "run_id_a": run.run_id,
                                  "run_id_b": runs[j].run_id, "hausdorff_distance_3d": matrix[i, j]})
        distance = np.concatenate(all_distances)
        summary_rows.append({
            "method": method, "method_label": METHODS[method][0], "n_runs": len(runs),
            "n_pf_points": len(distance), "representative_run_id": runs[medoid_index].run_id,
            "pf_dispersion_3d": dispersion,
            "negative_pressure_rate": float(np.mean(negative_rates)),
            "outside_support_rate": float(np.mean(outside_rates)),
            "median_distance_to_support": float(np.median(distance)),
            "mean_distance_to_support": float(np.mean(distance)),
            "p95_distance_to_support": float(np.quantile(distance, .95)),
        })
    return (representatives, pd.DataFrame(summary_rows), pd.DataFrame(run_rows),
            pd.DataFrame(pair_rows), pd.concat(point_frames, ignore_index=True), lo, hi)


def setup_style():
    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
        "axes.linewidth": .7, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def draw_support(ax, observed, pair):
    projected = observed[:, pair]
    # Convex hull commutes with linear projection; no PF filtering is involved.
    hull = ConvexHull(np.unique(projected, axis=0))
    boundary = hull.points[hull.vertices]
    ax.fill(boundary[:, 0], boundary[:, 1], facecolor="#F1F1F1",
            edgecolor="#D8D8D8", linewidth=.5, zorder=0)
    ax.scatter(projected[:, 0], projected[:, 1], color="#555555", marker="x",
               s=8, alpha=.30, linewidths=.45, zorder=1)


def draw_fronts(ax, runs, method, pair, representative=None):
    _, color, marker = METHODS[method]
    points = np.concatenate([r.y for r in runs])
    ax.scatter(points[:, pair[0]], points[:, pair[1]], c=color, marker=marker,
               s=2.5, alpha=.10, linewidths=0, zorder=2)
    if representative is not None:
        ax.scatter(representative.y[:, pair[0]], representative.y[:, pair[1]],
                   c=color, marker=marker, s=8, alpha=.88, linewidths=.15,
                   edgecolors="white", zorder=3)


def axis_limits(fronts, observed):
    points = np.concatenate([observed] + [r.y for runs in fronts.values() for r in runs])
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = .055 * (hi - lo)
    return list(zip(lo - padding, hi + padding))


def configure_axis(ax, pair, limits, letter):
    i, j = pair
    ax.set(xlim=limits[i], ylim=limits[j], xlabel=LABELS[i], ylabel=LABELS[j])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(.02, .98, f"({letter})", ha="left", va="top",
            transform=ax.transAxes, fontsize=11, fontweight="bold", zorder=5)


def shared_legend(fig, local=False):
    method_handles = [Line2D([], [], marker=marker, linestyle="none", color=color,
                            markersize=4.5, label=label) for label, color, marker in METHODS.values()]
    support_label = "Local observed-data support (k=5)" if local else "Observed-data support region"
    background_handles = [
        Line2D([], [], marker="x", linestyle="none", color="#555555", alpha=.65,
               markersize=4, label="Observed data"),
        Patch(facecolor="#F1F1F1", edgecolor="#D8D8D8", label=support_label),
    ]
    # Matplotlib lays out entries column-first; arrange as 3 methods above
    # the 2 neutral background entries within a single shared legend.
    handles = [method_handles[0], background_handles[0], method_handles[1],
               background_handles[1], method_handles[2]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, handletextpad=.5, columnspacing=1.2,
               borderaxespad=0, labelspacing=.7)


def make_figures(fronts, observed, representatives, output, dpi=600, width=7.2):
    setup_style()
    limits = axis_limits(fronts, observed)
    main, axes = plt.subplots(1, 3, figsize=(width, 3.05))
    main.subplots_adjust(left=.075, right=.99, bottom=.16, top=.76, wspace=.38)
    for ax, pair, letter in zip(axes, PAIRS, "abc"):
        draw_support(ax, observed, pair)
        # Draw every faint cloud first so highlights stay above all clouds.
        for method, runs in fronts.items():
            draw_fronts(ax, runs, method, pair, representatives[method])
        configure_axis(ax, pair, limits, letter)
    shared_legend(main)
    main.savefig(output / "verma_pf_framework_comparison.pdf")
    main.savefig(output / "verma_pf_framework_comparison.png", dpi=dpi)
    plt.close(main)

    supplementary, axes = plt.subplots(3, 3, figsize=(width, 7.7))
    supplementary.subplots_adjust(left=.085, right=.99, bottom=.065, top=.86, wspace=.39, hspace=.34)
    for column, (method, runs) in enumerate(fronts.items()):
        support = local_support_union_3d(observed, runs, k=5)
        for row, pair in enumerate(PAIRS):
            ax = axes[row, column]
            draw_support(ax, support, pair)
            draw_fronts(ax, runs, method, pair)
            configure_axis(ax, pair, limits, chr(ord("a") + row * 3 + column))
            if row == 0:
                ax.set_title(METHODS[method][0], pad=9)
    shared_legend(supplementary, local=True)
    supplementary.savefig(output / "verma_pf_all_runs_supplementary.pdf")
    supplementary.savefig(output / "verma_pf_all_runs_supplementary.png", dpi=dpi)
    plt.close(supplementary)


def caption_text(fronts):
    counts = {len(runs) for runs in fronts.values()}
    run_description = (f"{next(iter(counts))} repeated optimisation runs per framework"
                       if len(counts) == 1 else "repeated optimisation runs for each framework")
    caption = r"""\caption{Pairwise predictive-mean projections of the three-objective Pareto
recommendations obtained using the raw predictive mean, conventional GP
uncertainty, and cross-fitted conformal scale calibration (CFSC) on the Verma
dataset: (a) $T$--$P$, (b) $T$--$TD$, and (c) $P$--$TD$.
Faint points show all saved solutions from RUN_DESCRIPTION.
Larger, opaque markers highlight an actual representative run:
the medoid minimising the mean symmetric Hausdorff distance to the other runs
in three-dimensional predictive-mean objective space. A common objective-wise
min--max normalisation, pooled across all saved runs and frameworks, is used
for these distances; the same medoid run is highlighted in every projection.
Grey crosses denote observed data and light-grey shading shows pairwise
projections of the complete observed-data convex hull (the observed-data
support region). Pareto optimality was determined jointly over all three
objectives using each framework's optimisation criterion (predictive means or
uncertainty-adjusted objectives); the panels display only the corresponding
predictive-mean projections. Consequently, apparent domination in an individual
two-dimensional projection does not imply domination in the original
three-objective optimisation problem. No projected Pareto filtering or
coordinate averaging is applied.}
\label{fig:verma-pf-framework-comparison}
"""
    return caption.replace("RUN_DESCRIPTION", run_description)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pf-store", type=Path, default=HERE / "verma_pf_store.pkl")
    parser.add_argument("--observed", type=Path, default=PROJECT / "battery_data_Verma_modify.xlsx")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=7.2, help="Figure width in inches (default: journal two-column width)")
    args = parser.parse_args()
    if args.dpi < 600 or args.width < 7:
        parser.error("Use dpi >= 600 and width >= 7 inches for publication readability")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    originals = list(HERE.parent.glob("pf_overlay_support_3d_pairwise*.*"))
    protected = [args.pf_store, args.observed] + originals
    before = {str(p.resolve()): sha256(p) for p in protected if p.is_file()}
    fronts, saved_metadata = load_fronts(args.pf_store)
    observed_frame = pd.read_excel(args.observed)
    observed = observed_frame[list(TARGETS)].to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("Observed objective data contain missing/nonfinite values; nothing was dropped")
    print(f"PF source: {args.pf_store.resolve()}\nObjective order: {TARGETS}")
    for method, runs in fronts.items():
        print(f"  {method}: seeds {[r.run_id for r in runs]}, point counts {[len(r.y) for r in runs]}")
    print(f"Observed source: {args.observed.resolve()}, {len(observed)} rows")
    representatives, summary, runs, pairs, points, lo, hi = summarize(fronts, observed)
    for frame, filename in (
        (summary, "verma_pf_framework_summary.csv"),
        (runs, "verma_pf_run_summary.csv"),
        (pairs, "verma_pf_pairwise_hausdorff.csv"),
        (points, "verma_pf_point_support_metrics.csv"),
    ):
        frame.to_csv(args.output_dir / filename, index=False, float_format="%.17g")
    make_figures(fronts, observed, representatives, args.output_dir, args.dpi, args.width)
    (args.output_dir / "verma_pf_framework_comparison_caption.tex").write_text(caption_text(fronts), encoding="utf-8")
    after = {p: sha256(Path(p)) for p in before}
    if before != after:
        raise RuntimeError("An input or original diagnostic file changed during generation")
    provenance = {
        "objective_order": TARGETS, "objective_units": ["K", "Pa", "K"],
        "normalization_reference": "All saved PF points pooled across all methods and runs; observed data excluded from fitting ranges",
        "normalization_min": lo.tolist(), "normalization_max": hi.tolist(),
        "distance": "Exact symmetric Hausdorff, Euclidean normalized 3D predictive-mean coordinates",
        "medoid": "Minimum mean distance to other runs; exact ties resolved by smallest run ID",
        "dispersion": "Mean over unordered pairs of distinct runs; NaN if fewer than two runs",
        "rates_aggregation": "Arithmetic mean of per-run fractions, matching compute_nonphysical_rate in source notebook",
        "distances_aggregation": "Pooled saved solutions including repeats/duplicates and zero distances for inside candidates; NumPy linear quantiles",
        "quantitative_support": "3D convex hull of all observed objective vectors, matching notebook cells 33/35",
        "support_distance": "Minimum Euclidean distance to all triangular hull facets in normalized 3D space; zero inside",
        "support_membership_tolerance": HULL_TOL,
        "main_background": "Projected full observed hull; visual context, not an optimisation constraint or feasibility certificate",
        "supplementary_background": "Original method-specific union of k=5 nearest observed neighbours in unnormalized 3D objective space; projected convex hull",
        "palette_source": "revision/results/For paper/generate_section6_public_btms_figures.py: PF_METHOD_COLORS; orange added for raw_uncertainty",
        "saved_metadata": saved_metadata,
        "membership_caveat": "Saved Y_pareto values are means, while membership uses full 3D framework-specific objectives. Y_for_pareto and predictive standard deviations are not in this store; original uncertainty-aware nondominance cannot be revalidated from this pickle alone.",
        "optional_metric_missing_data": [],
        "input_and_original_diagnostic_sha256": before,
        "protected_files_unchanged": True,
        "png_dpi": args.dpi, "pdf_all_vector": True,
    }
    (args.output_dir / "verma_pf_framework_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=lambda x: list(x) if isinstance(x, range) else str(x)), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}\nInputs and original diagnostic figures: SHA-256 unchanged")


if __name__ == "__main__":
    main()

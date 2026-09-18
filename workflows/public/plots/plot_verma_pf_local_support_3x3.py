"""Regenerate the original Verma 3x3 diagnostic from saved PF data only.

Reuse the notebook's three plotting/support functions without executing any
model-fitting or optimisation cells. Keep original k=5 physical-space 3D
nearest-neighbour support, run colours, and all saved PF memberships.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
FUNCTIONS = {"local_support_union_3d", "draw_convex_hull_2d",
             "plot_pf_overlay_with_support_3d_pairwise"}


def load_plot_function(notebook):
    """Load only the named function definitions, not executable notebook cells."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    definitions = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if not any(f"def {name}(" in source for name in FUNCTIONS):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                definitions[node.name] = node
    if set(definitions) != FUNCTIONS:
        raise ValueError(f"Missing notebook plotting functions: {FUNCTIONS - set(definitions)}")
    namespace = {"np": np, "plt": plt, "ConvexHull": ConvexHull,
                 "NearestNeighbors": NearestNeighbors}
    module = ast.Module(body=list(definitions.values()), type_ignores=[])
    exec(compile(module, str(notebook), "exec"), namespace)
    return namespace["plot_pf_overlay_with_support_3d_pairwise"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "verma_pf_local_support_3x3.pdf")
    parser.add_argument("--preview", type=Path, help="Optional 150-dpi PNG for visual inspection")
    args = parser.parse_args()
    notebook = HERE.parent / "source" / "verma_plot_source.ipynb"
    source = HERE.parent / "reference" / "pareto" / "verma_pf_store.pkl"
    observed_path = HERE.parent / "data" / "battery_data_Verma_modify.xlsx"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (notebook, source, observed_path)}
    with source.open("rb") as stream:
        store = pickle.load(stream)["pf_store"]
    observed = pd.read_excel(observed_path)[["T", "P", "TD"]].to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("Observed objectives contain missing/nonfinite data")
    keys = ("raw", "raw_uncertainty", "cfsc")
    for key in keys:
        for seed, xp, yp in store[key]:
            if np.asarray(yp).ndim != 2 or yp.shape[1] != 3 or not np.isfinite(yp).all():
                raise ValueError(f"Invalid objective array: {key}, seed {seed}")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 13})
    plot = load_plot_function(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(store, observed, k_support=5, max_seeds=max(len(store[key]) for key in keys),
         axis_labels=("T(K)", "P (W)", "TD (K)"), dpi=150,
         save_pdf_path=args.output)
    fig = plt.gcf()
    assert len(fig.axes) == 9
    for row, pair in enumerate(((0, 1), (0, 2), (1, 2))):
        for column, key in enumerate(keys):
            collections = fig.axes[row * 3 + column].collections
            assert len(collections) == 1 + len(store[key])
            for artist, (_, _, yp) in zip(collections[1:], store[key]):
                np.testing.assert_array_equal(np.asarray(artist.get_offsets()), yp[:, pair])
    if args.preview:
        fig.savefig(args.preview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    print(args.output.resolve())
    print("Verified: 9 panels, all 7680 saved PF points retained in every projection; input files unchanged.")


if __name__ == "__main__":
    main()

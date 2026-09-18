"""Portable entry point for the selected in-house notebook workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import warnings


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
REFERENCE_DIR = HERE / "reference"
STEM = "simulation_three_pf_validation_from_csv"


def load_plotting():
    spec = importlib.util.spec_from_file_location("inhouse_original_plotting", HERE / "source" / "plotting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figures(output_dir):
    import pandas as pd

    plotting = load_plotting()
    table = pd.read_csv(REFERENCE_DIR / f"{STEM}_pf.csv")
    targets = ["predicted_Twall_avg", "predicted_\u0394P(Pa)"]
    fronts = {
        method: table.loc[table["framework"].eq(method), targets].to_numpy(float)
        for method in ("raw", "raw_uncertainty", "cfsc")
    }
    training = pd.read_csv(REFERENCE_DIR / f"{STEM}_training.csv").to_numpy(float)
    _, _, validation = plotting.plot_three_pf_with_validation(
        Y_raw=fronts["raw"], Y_raw_uncertainty=fronts["raw_uncertainty"],
        Y_cfsc=fronts["cfsc"],
        validation_csv=DATA_DIR / "simulation_selected_validation_points.csv",
        Y_train=training, save_dir=output_dir,
    )
    return {
        "result_source": "archived Pareto fronts and training objectives; supplied simulation truth",
        "pareto_counts": {method: len(front) for method, front in fronts.items()},
        "simulation_records": len(validation),
        "complete_simulation_truth": int(validation["truth_available"].sum()),
    }


def adapted_cell(cell, smoke):
    """Keep scientific statements unchanged except explicitly labelled smoke reductions."""
    index, source = cell["cell_index"], cell["source"]
    if index == 4:
        source = source.replace('DATA_PATH = "../Contraction Channel Data.xlsx"',
                                'DATA_PATH = str(DATA_DIR / "Contraction Channel Data.xlsx")')
    if index == 41:
        source = source.replace('save_dir="pareto_results"', 'save_dir=OUTPUT_DIR / "selected_validation"')
    if smoke:
        if index == 11:
            source = source.replace("bags=20,", "bags=0,\n    kernel_grid=[(\"RBF\", None)],")
        if index == 12:
            source = source.replace("n = 50", "n = 20")
        if index == 23:
            source = source.replace("K=5,", "K=2,")
        if index in (17, 24):
            source = source.replace('model_key="final_models",',
                                    'model_key="final_models",\n    pop_size=12,\n    n_gen=2,')
    return source


def train(output_dir, smoke):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    sys.path.insert(0, str(ROOT / "legacy"))
    if smoke:
        # Runtime cap only for the explicitly reduced integration check.
        torch.set_num_threads(1)
    cells = json.loads((HERE / "source" / "training_cells.json").read_text(encoding="utf-8"))
    namespace = {"DATA_DIR": DATA_DIR, "OUTPUT_DIR": output_dir, "__name__": "inhouse_selected_cells"}
    for cell in cells:
        index = cell["cell_index"]
        print(f"Executing original notebook cell {index} ({'SMOKE' if smoke else 'full settings'})", flush=True)
        before = set(plt.get_fignums())
        exec(compile(adapted_cell(cell, smoke), f"CaseStudy_battery_sim.ipynb:cell-{index}", "exec"), namespace)
        if index in (14, 16):
            label = "evaluation" if index == 14 else "after_finalize"
            for number, figure_number in enumerate(sorted(set(plt.get_fignums()) - before), start=1):
                figure = plt.figure(figure_number)
                figure.savefig(output_dir / f"predicted_vs_actual_{label}_{number}.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    # Additional exports preserve the rerun state; no re-selection or re-optimization.
    np.savez_compressed(
        output_dir / "trained_pareto_fronts.npz",
        X_train=namespace["X"], Y_train=namespace["Y"],
        X_raw=namespace["X_p"], Y_raw=namespace["Y_p"],
        X_raw_uncertainty=namespace["X_p_unc"], Y_raw_uncertainty=namespace["Y_p_unc"],
        X_cfsc=namespace["X_p_cali"], Y_cfsc=namespace["Y_p_cali"],
    )
    _, _, validation = load_plotting().plot_three_pf_with_validation(
        Y_raw=namespace["Y_p"], Y_raw_uncertainty=namespace["Y_p_unc"],
        Y_cfsc=namespace["Y_p_cali"],
        validation_csv=DATA_DIR / "simulation_selected_validation_points.csv",
        Y_train=namespace["Y"], save_dir=output_dir,
    )
    auto = namespace["auto"]
    return {
        "result_source": "fresh surrogate training and optimization; supplied historical simulation truth",
        "executed_cell_indices": [cell["cell_index"] for cell in cells],
        "training_rows": len(namespace["X"]),
        "selected_model": auto.info(),
        "evaluation_metrics": namespace["metrics"],
        "calibration": auto.calibration,
        "pareto_counts": {"raw": len(namespace["Y_p"]), "raw_uncertainty": len(namespace["Y_p_unc"]),
                          "cfsc": len(namespace["Y_p_cali"])},
        "simulation_records": len(validation),
        "external_truth_boundary": "Real T/Real P and their batch predictions remain supplied historical records; new selected designs have no new simulation truth.",
    }


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (("figures", "Recreate the final figure from archived numerical results"),
                               ("train", "Retrain and optimize using the selected original notebook cells")):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--output-dir", type=Path, help="Override the separate figures/train output directory")
        if command == "train":
            child.add_argument("--smoke", action="store_true", help="Reduced check: 20 rows, no bags, RBF only, K=2, population=12, generations=2")
    args = parser.parse_args(argv)
    smoke = getattr(args, "smoke", False)
    default_output = ROOT / "outputs" / "inhouse" / ("figures" if args.command == "figures" else "train")
    if args.command == "train":
        default_output = default_output / ("smoke" if smoke else "full")
    output = (args.output_dir or default_output).expanduser().resolve()
    for protected in (DATA_DIR, REFERENCE_DIR, HERE / "source"):
        if output == protected or protected in output.parents:
            parser.error("Output directory must not overwrite packaged data, references or source snapshots")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")
    metadata = {
        "workflow": "inhouse", "command": args.command, "smoke": smoke,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "smoke_overrides": ({"sample_n": 20, "bags": 0, "kernel_grid": [["RBF", None]],
                             "calibration_K": 2, "pop_size": 12, "n_gen": 2, "torch_threads": 1} if smoke else {}),
        "authoritative_source": "CaseStudy_battery_sim.ipynb; provenance/inhouse.json",
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        result = figures(output) if args.command == "figures" else train(output, smoke)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    metadata.update(result)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    print(f"In-house {args.command} {'SMOKE ' if smoke else ''}outputs: {output}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

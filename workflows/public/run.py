"""Execute the preserved public-data workflows without the original workspace."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "source"
DATA = HERE / "data"
REFERENCE = HERE / "reference"
PLOTS = HERE / "plots"
DATASETS = {
    "li": ("battery_data_Li.xlsx", ["V", "TD"]),
    "verma": ("battery_data_Verma_modify.xlsx", ["T", "P", "TD"]),
}
NESTED_CELLS = {
    ("li", "nested-surrogate"): [13, 14],
    ("verma", "nested-surrogate"): [16, 17],
    ("li", "nested-calibration"): [22, 23],
    ("verma", "nested-calibration"): [25, 26],
}
PARETO_CELLS = {
    "li": [2, 4, 22, 29, 31, 43, 44],
    "verma": [2, 4, 22, 30, 32, 48, 49],
}


def output_directory(value: str | None) -> Path:
    output = Path(value).expanduser().resolve() if value else ROOT / "outputs" / "public"
    # A custom output may never overwrite archived inputs or source code.
    for protected in (HERE, ROOT / "legacy", ROOT / "provenance", ROOT / "docs"):
        if output == protected or protected in output.parents:
            raise ValueError(f"Output is inside a preserved source/reference directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


@contextmanager
def plotting_environment():
    previous = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_reliability_plot():
    """Load the exact original plotting definition without model dependencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from typing import Sequence

    path = SOURCE / "nested_cell_20.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
             and node.name == "_save_calibration_reliability_diagrams"]
    namespace = {"pd": pd, "plt": plt, "Path": Path, "Sequence": Sequence}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    return namespace["_save_calibration_reliability_diagrams"]


def figures(args):
    import pandas as pd

    output = output_directory(args.output_dir)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    with plotting_environment():
        reliability = original_reliability_plot()
        for dataset in datasets:
            destination = output / dataset / "figures"
            destination.mkdir(parents=True, exist_ok=True)
            metadata = {
                "dataset": dataset, "stage": "figures", "smoke": False,
                "source": "archived reference results; no model fitting",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            record = destination / "run_metadata.json"
            record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            observed = DATA / DATASETS[dataset][0]
            pareto = REFERENCE / "pareto" / f"{dataset}_pf_store.pkl"
            script = "plot_li_pf_three_frameworks.py" if dataset == "li" else "plot_verma_pf_framework_comparison.py"
            subprocess.run([sys.executable, str(PLOTS / script), "--pf-store", str(pareto),
                            "--observed", str(observed), "--output-dir", str(destination)], check=True)
            summary = REFERENCE / "nested" / dataset / "calibration" / "reliability_summary.csv"
            reliability(pd.read_csv(summary), DATASETS[dataset][1], destination)
            if dataset == "verma":
                subprocess.run([sys.executable, str(PLOTS / "plot_verma_pf_local_support_3x3.py"),
                                "--output", str(destination / "verma_pf_local_support_3x3.pdf")], check=True)
                combined = load_module(PLOTS / "plot_combined_reliability.py", "public_combined_reliability")
                combined.OUTPUT = destination / "reliability_diagrams"
                combined.OUTPUT.mkdir(exist_ok=True)
                shutil.copyfile(summary, destination / "reliability_summary.csv")
                combined.main()
            metadata["status"] = "completed"
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Recreated {dataset} figures from archived numerical results: {destination}")


class InvocationPaths(ast.NodeTransformer):
    """Only adapt file constants and explicitly requested smoke invocations."""

    def __init__(self, destination: Path, smoke: bool):
        self.destination = destination
        self.smoke = smoke

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value in ("../battery_data_Li.xlsx", "../battery_data_Verma_modify.xlsx"):
                return ast.copy_location(ast.Constant(str(DATA / Path(node.value).name)), node)
            if node.value == "pareto_results" or node.value.startswith("results/"):
                return ast.copy_location(ast.Constant(str(self.destination)), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not self.smoke or not isinstance(node.func, ast.Name):
            return node
        if node.func.id in ("run_gp_nested_repeated_oof_benchmark",
                            "run_model_selection_bagging_nested_repeated_calibration"):
            overrides = {
                "repeat_seeds": "range(1)", "outer_folds": "2", "inner_folds": "2",
                "calibration_folds": "2", "bagging_bags": "1", "bags": "1",
                "bootstrap_reps": "20", "kernel_grid": "[('Matern', 2.5)]",
            }
        elif node.func.id == "multi_seed_pf_compare":
            overrides = {"seeds": "range(1)", "K": "2", "pop_size": "16", "n_gen": "2"}
        elif node.func.id == "AutoConfig":
            overrides = {"bags": "1"}
        elif node.func.id == "save_pf_store":
            # The original save cell repeats full-run settings in a dictionary.
            # Keep smoke metadata honest without changing full-run source values.
            for keyword in node.keywords:
                if keyword.arg == "metadata" and isinstance(keyword.value, ast.Dict):
                    values = {"seeds": "range(1)", "pop_size": "16", "n_gen": "2"}
                    for index, key in enumerate(keyword.value.keys):
                        if isinstance(key, ast.Constant) and key.value in values:
                            keyword.value.values[index] = ast.parse(values[key.value], mode="eval").body
                    keyword.value.keys.append(ast.Constant("smoke"))
                    keyword.value.values.append(ast.Constant(True))
            return node
        else:
            return node
        for keyword in node.keywords:
            if keyword.arg in overrides:
                keyword.value = ast.parse(overrides[keyword.arg], mode="eval").body
        return node


def execute_cell(prefix, index, namespace, destination, smoke=False, invocation=False):
    path = SOURCE / f"{prefix}_cell_{index:02d}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if invocation:
        tree = InvocationPaths(destination, smoke).visit(tree)
        ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec", dont_inherit=True), namespace)


def train(args):
    sys.path.insert(0, str(ROOT / "legacy"))
    import torch

    output = output_directory(args.output_dir)
    destination = output / args.dataset / args.stage / ("smoke" if args.smoke else "full")
    destination.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        torch.set_num_threads(1)
    module = ModuleType("__public_preserved_notebook__")
    module.__file__ = str(__file__)
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    metadata = {
        "dataset": args.dataset, "stage": args.stage, "smoke": args.smoke,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "source": "provenance/public.json",
        "smoke_overrides": ({
            "nested": "1 repeat; 2 outer/inner/calibration folds; 1 bootstrap GP; 20 CI resamples; Matern 2.5 kernel candidate",
            "pareto": "1 seed; 2 calibration folds; 1 bootstrap GP; population 16; 2 generations",
        } if args.smoke else {}),
    }
    manifest = destination / "run_metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with plotting_environment():
            if args.stage.startswith("nested-"):
                for index in [2, 5, 8, 10]:
                    execute_cell("nested", index, namespace, destination)
                if args.stage == "nested-calibration":
                    execute_cell("nested", 20, namespace, destination)
                for index in NESTED_CELLS[(args.dataset, args.stage)]:
                    execute_cell("nested", index, namespace, destination, args.smoke, invocation=True)
            else:
                invocation_cells = {4, 31, 44} if args.dataset == "li" else {4, 32, 49}
                for index in PARETO_CELLS[args.dataset]:
                    execute_cell(args.dataset, index, namespace, destination, args.smoke,
                                 invocation=index in invocation_cells)
                namespace["df_metrics"].to_csv(destination / "pareto_metrics_by_seed.csv", index=False)
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.dataset} {args.stage} {'SMOKE' if args.smoke else 'full'} results: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    figure_parser = subparsers.add_parser("figures", help="Recreate final figures from preserved results; no training")
    figure_parser.add_argument("--dataset", choices=["all", "li", "verma"], default="all")
    figure_parser.add_argument("--output-dir")
    figure_parser.set_defaults(func=figures)
    train_parser = subparsers.add_parser("train", help="Run the exact selected notebook pipeline")
    train_parser.add_argument("--dataset", choices=["li", "verma"], required=True)
    train_parser.add_argument("--stage", choices=["nested-surrogate", "nested-calibration", "pareto"], required=True)
    train_parser.add_argument("--smoke", action="store_true", help="Use documented reduced invocation settings; not historical results")
    train_parser.add_argument("--output-dir")
    train_parser.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

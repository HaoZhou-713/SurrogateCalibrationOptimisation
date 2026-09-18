"""Portable runners around the preserved benchmark notebook cells.

The final manuscript tables are archived inputs, assembled historically from
several experiments. Training commands produce fresh stage outputs separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = HERE / "reference"
sys.path.insert(0, str(ROOT / "legacy"))
os.environ.setdefault("MPLBACKEND", "Agg")


def load_source(name):
    path = HERE / "source" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figures(output):
    module = load_source("generate_paper_artifacts")
    module.OUT = output
    module.REVISION = REFERENCE
    module.FON_SUMMARY = REFERENCE / "fon_final.csv"
    module.DTLZ2_SUMMARY = REFERENCE / "dtlz2_final.csv"
    module.DTLZ2_SAMPLE_SIZE_XLSX = REFERENCE / "sensitivity/ece95.xlsx"
    module.main()
    cost = load_source("generate_benchmark_cost_table")
    cost.OUT = output
    cost.REVISION = REFERENCE
    cost.SOURCES = {b: REFERENCE / (b.lower() + "_five_methods.csv") for b in ["FON", "DTLZ2"]}
    cost.main()
    # The original generator's README contains historical source paths. Keep
    # those as provenance and add the standalone mapping next to the outputs.
    (output / "standalone_sources.json").write_text(json.dumps({
        "mode": "redraw_archived_final_tables",
        "source_mapping": {
            "FON": "workflows/benchmark/reference/fon_final.csv",
            "DTLZ2": "workflows/benchmark/reference/dtlz2_final.csv",
            "sample_size": "workflows/benchmark/reference/sensitivity/ece95.xlsx",
        },
        "lineage_note": "See docs/benchmark.md; retraining stages do not overwrite these historical final tables.",
    }, indent=2) + "\n", encoding="utf-8")
    print("Benchmark figures:", output)


class CellRunner:
    def __init__(self, key, output, smoke):
        self.key, self.output, self.smoke = key, output, smoke
        self.namespace = {"__name__": "__benchmark_cells__", "display": lambda *x: None}
        self.records = []
        self.trial_frames = []

    def run(self, index, replacements=()):
        path = HERE / "source_cells" / self.key / f"cell_{index:03d}.py"
        original = path.read_text(encoding="utf-8")
        source = original
        changes = []
        for old, new in replacements:
            if old not in source:
                raise ValueError(f"Expected source text absent in {self.key} cell {index}: {old!r}")
            source = source.replace(old, new)
            changes.append({"from": old, "to": new})
        self.records.append({"cell_index": index, "sha256_source": hashlib.sha256(original.encode()).hexdigest(), "replacements": changes})
        self.save_audit("running")
        exec(compile(source, str(path), "exec"), self.namespace)

    def save_audit(self, status):
        (self.output / "execution.json").write_text(json.dumps({
            "stage_source": self.key,
            "smoke": self.smoke,
            "status": status,
            "cells": self.records,
            "note": "Original code cells are unchanged on disk; listed substitutions adapt paths and explicitly selected run settings.",
        }, indent=2) + "\n", encoding="utf-8")


def five_methods(runner, method):
    runner.run(1)
    changes = [("Path(\"results/benchmark_FON_5methods_20seeds\")" if runner.key.startswith("fon_") else "Path(\"results/benchmark_DTLZ2_5methods_10seeds\")", "Path('results')")]
    if runner.smoke:
        changes += [("N_POOL = 30000", "N_POOL = 512"), ("SEEDS = range(20)", "SEEDS = range(1)"),
                    ("N_EVAL_METRIC = 3000", "N_EVAL_METRIC = 128"), ("BAGS = 20", "BAGS = 2"),
                    ("NSGA2_POP_SIZE = 96", "NSGA2_POP_SIZE = 24"), ("NSGA2_N_GEN = 60", "NSGA2_N_GEN = 3")]
    runner.run(2, changes)
    choices = {"vanilla": 0, "selected": 1, "bagging": 2, "split": 3, "cv": 4}
    if method != "all":
        runner.namespace["METHOD_CONFIGS"] = [runner.namespace["METHOD_CONFIGS"][choices[method]]]
    for index in range(3, 12):
        runner.run(index)


def vanilla(runner):
    fon = runner.key.startswith("fon_")
    # DTLZ2 cell 2 references this before cell 4 assigns it in the notebook.
    runner.namespace["BENCHMARK_NAME"] = "FON" if fon else "DTLZ2"
    for index in (range(1, 8) if fon else range(1, 7)):
        changes = []
        if index == (5 if fon else 2):
            changes += [("Path(\"results\") / BENCHMARK_NAME / \"vanilla_gp\"", "Path('results')")]
        if runner.smoke and index == 4:
            changes += [("SEEDS = list(range(20))", "SEEDS = list(range(1))"), ("N_POOL = 30000", "N_POOL = 512"),
                        ("N_EVAL_METRIC = 5000", "N_EVAL_METRIC = 128"), ("N_CANDIDATES = 30000", "N_CANDIDATES = 256"),
                        ("N_REF_PF = 1000", "N_REF_PF = 128"), ("TRAIN_ITERS = 120" if fon else "TRAIN_ITERS = 140", "TRAIN_ITERS = 8")]
        runner.run(index, changes)


def cv_metrics(runner, sensitivity=False):
    fon = runner.key.startswith("fon_")
    if fon:
        for index in [2, 4, 5, 26, 27, 28, 29, 31, 32, 34, 37, 39, 40, 42]:
            changes = []
            if runner.smoke:
                if index == 28:
                    changes.append(("bags=10", "bags=2"))
                if index == 34:
                    changes.append(("N_pool=30000", "N_pool=512"))
                if index == 39:
                    changes.append(("n_eval_metric: int = 5000", "n_eval_metric: int = 128"))
                if index == 40:
                    changes += [("n_candidates=100000", "n_candidates=256"), ("seeds=range(20)", "seeds=range(1)")]
            runner.run(index, changes)
    else:
        for index in range(1, 7):
            changes = []
            if index == 6:
                # The notebook reassigns df_all on every sensitivity batch.
                # Retain all returned frames so an early failed batch cannot
                # be hidden by a later successful batch.
                original_seed_runner = runner.namespace["run_benchmark_seeds"]

                def capture_batch(*args, **kwargs):
                    result = original_seed_runner(*args, **kwargs)
                    runner.trial_frames.append(result[0])
                    return result

                runner.namespace["run_benchmark_seeds"] = capture_batch
                changes.append(('output_dir="results/dtlz2_samples_sensitivity"', 'output_dir="results"'))
                if not sensitivity:
                    # Archived CV metric file records 70 training observations;
                    # the saved notebook later changed this loop for sensitivity.
                    changes.append(("for n in [30, 50, 90, 110]:", "for n in [70]:"))
            if runner.smoke:
                if index == 4:
                    changes.append(("bags=10", "bags=2"))
                if index == 6:
                    changes += [("N_pool=30000", "N_pool=512"), ("seeds=range(20)", "seeds=range(1)"),
                                ("n_eval_metric=5000", "n_eval_metric=128"), ("pop_size=128", "pop_size=24"), ("n_gen=100", "n_gen=3")]
                    if sensitivity:
                        changes.append(("for n in [30, 50, 90, 110]:", "for n in [30]:"))
            runner.run(index, changes)


def legacy_pareto(runner, method):
    fon = runner.key.startswith("fon_")
    indices = [2, 4, 5, 26, 27, 28, 29, 30, 32, 35, 37, 38, 40] if fon else [2, 3, 6, 7, 29, 30, 31, 32, 33, 34, 36, 41, 42, 48, 50, 52]
    for index in indices:
        changes = []
        if not fon and index == 48:
            # The current saved trial returns (metrics, model); the historical
            # seed loop expects metrics. Unpack only the return container.
            changes.append(("out = evaluate_one_trial(", "out, _trial_model = evaluate_one_trial("))
        if index == (38 if fon else 50) and method == "raw":
            changes.append(("flag_calibration = True", "flag_calibration = False"))
        if index == (40 if fon else 52):
            old = 'out_path = "New_method_results.csv"' if fon else 'out_path = "normal_cali_results.csv"'
            changes.append((old, f'out_path = "legacy_{method}_results.csv"'))
        if runner.smoke:
            if index == (28 if fon else 31):
                changes.append(("bags=10", "bags=2"))
            if index == (32 if fon else 36):
                changes.append(("N_pool=30000", "N_pool=512"))
            if not fon and index == 34:
                changes += [("pop_size=256", "pop_size=24"), ("n_gen=100", "n_gen=3")]
            if index == (38 if fon else 50):
                changes += [("n_candidates=100000", "n_candidates=256"), ("seeds=range(20)", "seeds=range(1)")]
        runner.run(index, changes)


def check_training_frames(runner):
    import pandas as pd
    frames = runner.trial_frames + [runner.namespace.get(name) for name in ["df", "df_all"]]
    frames = [x for x in frames if isinstance(x, pd.DataFrame) and ("seed" in x.columns)]
    if not frames:
        raise RuntimeError("Training produced no seed-level table")
    for frame in frames:
        if "failed" in frame and frame["failed"].fillna(False).any():
            raise RuntimeError("One or more notebook trials failed; inspect saved seed table and console output")
        if "RMSE_mean" in frame and not frame["RMSE_mean"].notna().all():
            raise RuntimeError("Training table has missing accuracy metrics")


def train(args, output):
    stage = args.stage.replace("-", "_")
    key = f"{args.benchmark}_{'cv_metrics' if stage == 'sensitivity' else stage}"
    runner = CellRunner(key, output, args.smoke)
    previous = Path.cwd()
    os.chdir(output)
    try:
        if stage == "five_methods":
            five_methods(runner, args.method)
        elif stage == "vanilla":
            vanilla(runner)
        elif stage in ["cv_metrics", "sensitivity"]:
            cv_metrics(runner, stage == "sensitivity")
        elif stage == "legacy_pareto":
            legacy_pareto(runner, args.method)
        check_training_frames(runner)
        runner.save_audit("completed")
    except BaseException:
        runner.save_audit("failed")
        raise
    finally:
        os.chdir(previous)
    print("Fresh benchmark training outputs:", output)


def check_lineage(output):
    """Check identified numerical joins without manufacturing missing history."""
    import numpy as np
    import pandas as pd
    records = []
    for benchmark in ["fon", "dtlz2"]:
        final = pd.read_csv(REFERENCE / f"{benchmark}_final.csv").set_index("Method")
        files = {
            "Selected + Bagging": "basic_results.csv",
            "Selected + Bagging + Split CP": "calibrated_results.csv" if benchmark == "fon" else "normal_cali_results.csv",
            "Selected + Bagging + CV": "New_method_results.csv" if benchmark == "fon" else "cv+_results.csv",
        }
        for method, name in files.items():
            path = REFERENCE / "legacy_pareto" / benchmark.upper() / name
            source = pd.read_csv(path)
            for metric in ["GD", "IGD", "Hausdorff", "Recall", "HV_ratio", "HV_pred", "HV_gt", "n_pf_gt", "n_pareto_x"]:
                for stat, operation in [("median", "median"), ("q25", lambda x: x.quantile(.25)), ("q75", lambda x: x.quantile(.75)), ("mean", "mean"), ("std", "std")]:
                    column = f"{metric}_{stat}"
                    if metric not in source or column not in final:
                        continue
                    actual = float(source[metric].agg(operation))
                    expected = float(final.loc[method, column])
                    records.append({"benchmark": benchmark, "method": method, "column": column, "computed": actual,
                                    "final": expected, "matches": bool(np.isclose(actual, expected, atol=1e-8, rtol=1e-5)),
                                    "source": path.relative_to(ROOT).as_posix()})
    data = pd.DataFrame(records)
    data.to_csv(output / "legacy_pareto_lineage_check.csv", index=False)
    if not data["matches"].all():
        raise RuntimeError("A previously identified final Pareto/source aggregate changed")
    print(f"Verified {len(data)} final summary values against archived legacy Pareto runs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["figures", "check-lineage", "train"]:
        item = sub.add_parser(name)
        item.add_argument("--output-dir", type=Path)
        if name == "train":
            item.add_argument("--benchmark", choices=["fon", "dtlz2"], default="fon")
            item.add_argument("--stage", choices=["five-methods", "vanilla", "cv-metrics", "legacy-pareto", "sensitivity"], default="five-methods")
            item.add_argument("--method", default="all", help="five-methods: all/vanilla/selected/bagging/split/cv; legacy-pareto: cv/raw")
            item.add_argument("--smoke", action="store_true", help="Explicit reduced workload; never a reproduction of final numerical results")
    args = parser.parse_args()
    if args.command == "train":
        if args.stage == "sensitivity" and args.benchmark != "dtlz2":
            parser.error("sensitivity is recorded only for DTLZ2")
        valid = ["all", "vanilla", "selected", "bagging", "split", "cv"] if args.stage == "five-methods" else ["all", "cv", "raw"] if args.stage == "legacy-pareto" else ["all"]
        if args.method not in valid:
            parser.error(f"Invalid method for {args.stage}; choices: {', '.join(valid)}")
        if args.stage == "legacy-pareto" and args.method == "all":
            args.method = "cv"
    default = ROOT / "outputs/benchmark" / ("figures" if args.command == "figures" else "lineage" if args.command == "check-lineage" else f"{'smoke_' if args.smoke else ''}{args.benchmark}_{args.stage}_{args.method}")
    output = (args.output_dir or default).resolve()
    # User-selectable output must never overwrite the reference/source snapshots.
    if output == ROOT:
        parser.error("Output directory cannot be the repository root")
    for protected in [ROOT / name for name in ["workflows", "legacy", "provenance", "docs", "tests", ".git"]]:
        if output == protected or protected in output.parents:
            parser.error("Output directory overlaps protected source/reference files")
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "figures":
        figures(output)
    elif args.command == "check-lineage":
        check_lineage(output)
    else:
        train(args, output)


if __name__ == "__main__":
    main()

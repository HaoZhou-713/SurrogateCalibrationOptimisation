# Public battery datasets: preserved validation and final Pareto results

The latest nested repeated validation and the final saved Pareto results use two separate original pipelines. This package preserves both, including their different calibration definitions. It does not replace either pipeline with the other or change its scientific algorithms.

Run these commands from the repository root. The entry point also works from other working directories when addressed by its absolute path.

```powershell
python workflows/public/run.py figures
python workflows/public/run.py figures --dataset li
python workflows/public/run.py train --dataset li --stage nested-surrogate
python workflows/public/run.py train --dataset li --stage nested-calibration
python workflows/public/run.py train --dataset li --stage pareto
python workflows/public/run.py train --dataset verma --stage nested-surrogate
python workflows/public/run.py train --dataset verma --stage nested-calibration
python workflows/public/run.py train --dataset verma --stage pareto
```

`figures` performs no model fitting. It recreates the saved Li three-framework Pareto plot, the Verma main and supplementary framework plots, the final Verma local-support plot, and the latest nested calibration reliability plots. Verma numerical framework summaries are recomputed from the preserved Pareto coordinates. All generated files go to `outputs/public`, or the directory passed with `--output-dir`. Training outputs have separate dataset, stage, and `full`/`smoke` subdirectories. Archived `reference/` files are never overwritten.

For a bounded execution check, add `--smoke` to any training command. This is explicitly a reduced run, not reproduction of the published numerical values:

```powershell
python workflows/public/run.py train --dataset li --stage nested-surrogate --smoke
python workflows/public/run.py train --dataset li --stage nested-calibration --smoke
python workflows/public/run.py train --dataset li --stage pareto --smoke
```

Smoke nested validation uses one repeat, two outer/inner/calibration folds, one bootstrap GP member, 20 observation-bootstrap resamples, and a single Matern-2.5 kernel candidate. Smoke Pareto training uses one seed, two calibration folds, one bootstrap GP member, population 16, and two generations. It preserves the model fitting and optimization implementations. Full runs keep the original settings, including all data, all seeds and all kernel candidates. The original dataset-specific train/validation fractions remain unchanged in both modes. Each run writes `run_metadata.json`, including failures if fitting raises an exception.

## Source and result map

| Result | Exact original source | Selected zero-based cells |
|---|---|---|
| Nested model definitions and surrogate validation | `revision/ablation_battery_nested_repeated_oof_robust_gp_pf.ipynb` | 2, 5, 8, 10 |
| Li nested surrogate ablation | Same notebook | 13, 14 |
| Verma nested surrogate ablation | Same notebook | 16, 17 |
| Nested calibration definitions | Same notebook | 20 |
| Li nested calibration | Same notebook | 22, 23 |
| Verma nested calibration | Same notebook | 25, 26 |
| Li final multi-seed Pareto generation | `Surrogate_models_for_small-size_datasets/CaseStudy_battery_Li.ipynb` | 2, 4, 22, 29, 31, 43, 44 |
| Verma final multi-seed Pareto generation | `Surrogate_models_for_small-size_datasets/CaseStudy_battery_Ver.ipynb` | 2, 4, 22, 30, 32, 48, 49 |
| Final Verma local-support plot functions | Same Verma notebook | 31 |

The latest `_robust_gp_pf` notebook's cells 0–26 have identical source to `ablation_battery_nested_repeated_oof_robust_gp.ipynb`. Its stored outputs show completed Li and Verma nested validation. The appended Pareto section is a different experiment: its Verma run failed allocating approximately 4.95 GB, so it is not selected as the final Pareto pipeline.

`source/*_cell_*.py` contains exact joined cell source, without refactoring. The runner only adjusts data/output paths and applies explicit smoke invocation overrides. The nested source defines its own robust GP classes. The Pareto source imports the unchanged `legacy/surrogate_pipeline.py` and `legacy/surrogate_pareto_botorch.py`. The latter contains the original custom NSGA-II implementation; this public route does not require pymoo.

The two raw spreadsheets are `data/battery_data_Li.xlsx` (50 observations; features d1, d2, d3, d4, v; targets V, TD) and `data/battery_data_Verma_modify.xlsx` (100 observations; features V, T_in, AR, IA; targets T, P, TD). The original unmodified Verma spreadsheet is not the input to these final workflows.

Nested historical outputs are retained in `reference/nested/{li,verma}/{surrogate_ablation,calibration}`. Full nested settings are ten partition seeds 0–9, five outer folds, five inner folds, twenty bootstrap GP members plus one base model, and 2,000 bootstrap resamples of original observation IDs. Calibration uses alpha 0.1, five calibration folds, and the original 20% holdout comparator.

Final Pareto snapshots and figure/table artifacts are in `reference/pareto`, originating from `Surrogate_models_for_small-size_datasets/pareto_results`. Both saved stores were written on 2026-09-04 and include ten seeds, NSGA-II population 256, 150 generations, all-minimization objectives, and `use_hull=False`. Li uses train/validation fractions 0.1/0.1; Verma uses 0.15/0.15. Li saved keys are `raw`, `raw_uncertainty`, `cal`; Verma keys are `raw`, `raw_uncertainty`, `cfsc`, although its original descriptive metadata lists `cal`. This historical naming is retained.

## Reproducibility limits

The cell source and selected reference files are hash-tracked in `provenance/public.json`. The full historical model-fitting runs were not rerun as part of packaging. The notebook's recorded environment is Python 3.10.15 / `arbo`; use the repository's captured environment to reduce numerical drift. GP optimization can still vary with numerical libraries and hardware.

Stored Pareto `Y_pareto` arrays are predictive means. The uncertainty-adjusted objective values used for membership and the predictive standard deviations were not saved. Figures and support statistics can be reproduced from the snapshots, but uncertainty-aware nondominance cannot be independently revalidated from those pickles alone. No retrospective filtering is performed.

The old `generate_section6_public_btms_figures.py` reads pre-nested validation CSVs, so it is deliberately excluded from the latest-validation figure path. Preserved historical JSON provenance may contain original absolute paths as evidence; executable code never reads those locations.

## Packaging checks performed

Using the recorded `arbo` environment, all final public figures were regenerated, all five original Verma numerical tests passed, and the four regenerated Verma CSV tables matched the archived tables to `1e-12`. Li figure generation also succeeded from an external working directory. CPU smoke runs completed for Li nested surrogate validation, Li nested calibration, Li Pareto training, and Verma Pareto training. The Verma main figure was visually inspected. All 108 copied/extracted source and reference artifacts passed their documented hash checks. Verma nested surrogate and calibration smoke runs also completed through the root `reproduce.py` entry point; all reported per-repeat numeric metrics were finite. Full training at the original settings was not executed during packaging.

# Benchmark: preserved final figures and training stages

The final paper figures are reproduced from their exact saved numerical inputs.
The final five-method tables were historically assembled from several runs;
running one notebook does not regenerate all their entries. This repository
preserves that distinction and does not change the original methodology.

From the repository root:

```powershell
python workflows/benchmark/run.py figures
python workflows/benchmark/run.py check-lineage
python workflows/benchmark/run.py train --benchmark fon --stage vanilla --smoke
python workflows/benchmark/run.py train --benchmark dtlz2 --stage five-methods --method selected --smoke
```

`figures` creates figures 5.1–5.4, the controlled benchmark summary table, and the
benchmark timing table in `outputs/benchmark/figures`. It uses the original plot
scripts, changing only their input/output path globals. Both PDF and PNG are
generated. The inputs and original PDFs are under
`workflows/benchmark/reference`; reruns cannot overwrite that directory.
The repository root and source/documentation/test directories are also rejected
as output destinations; `outputs/` descendants and external directories work.

The standalone mappings are:

| Repository input | Original input below the old project root | Used for |
| --- | --- | --- |
| `reference/fon_final.csv` | `revision/results/FON/FON_5methods_20seeds_summary_final.csv` | FON final figure/table values |
| `reference/dtlz2_final.csv` | `revision/results/DTLZ2/DTLZ2_5methods_20seeds_summary_final.csv` | DTLZ2 final figure/table values |
| `reference/sensitivity/ece95.xlsx` | `revision/results/dtlz2_samples_sensitivity/DTLZ2_sample_size_sensitivity_full_data_ECE95.xlsx` | Figure 5.3, sheet `Key_Metrics_ECE95` |
| `reference/fon_five_methods.csv` | `revision/results/benchmark_FON_5methods_20seeds/fon_5methods_20seeds_with_surrogate_calibration_metrics.csv` | Original accuracy, timing source |
| `reference/dtlz2_five_methods.csv` | `revision/results/benchmark_DTLZ2_5methods_10seeds/dtlz2_5methods_10seeds_with_surrogate_calibration_metrics.csv` | Original accuracy, timing source |

Paths in that table beginning with `reference/` are relative to
`workflows/benchmark/`. The original generator's README retains its original
paths for provenance; `standalone_sources.json` next to generated outputs gives
the portable mappings.

## What the final table combines

The original scripts `generate_paper_artifacts.py` and
`generate_benchmark_cost_table.py` are copied byte for byte in `source/`.
The first reads the final summary tables. The second reads the earlier raw
five-method CSVs, so its Vanilla GP timing is from that earlier runner.
The current saved cost generator adds a `Runs` column to the formatted table
and LaTeX output. The archived formatted CSV has six columns; a fresh export
has seven. Its six historical columns and every numeric timing statistic match
the archive. Source-path strings in the timing and sensitivity CSVs use the
portable paths. These are recorded export differences, not modified metrics.

For both benchmarks, the final `Selected GP/ICM` accuracy/Pareto entries match
the revision five-method run, apart from the later calibration summary fields.
The final `Vanilla GP` row uses the separate true-vanilla baseline notebook and
CSV. The final Bagging and Split CP surrogate-accuracy entries come from the
revision five-method runs. Their final Pareto entries, and those for CV, come
from these older saved runs:

| Final method | FON source under `reference/legacy_pareto/FON` | DTLZ2 source under `reference/legacy_pareto/DTLZ2` |
| --- | --- | --- |
| Selected + Bagging | `basic_results.csv` | `basic_results.csv` |
| Selected + Bagging + Split CP | `calibrated_results.csv` | `normal_cali_results.csv` |
| Selected + Bagging + CV | `New_method_results.csv` | `cv+_results.csv` |

`check-lineage` reaggregates median, q25, q75, mean and sample standard deviation
for nine Pareto fields, and verifies **270 values** against the final summaries
using `atol=1e-8, rtol=1e-5`. It writes the evidence table rather than replacing
the final inputs.

The final CV accuracy entries use the later metric runs in `reference/fon_cv_metrics.csv`
and `reference/dtlz2_cv_metrics.csv` where matching values are recoverable.
The FON CSV also has appended summary rows: these are preserved verbatim,
not silently treated as additional seeds. Its final coverage differs from the
saved seed-level coverage. DTLZ2 CV metrics use 70 training observations, while
the five-method accuracy runner uses 50. Both legacy DTLZ2 Pareto CSVs and the
CV Pareto CSV use 70; FON uses 30.

The final summaries record `calibration_target=0.95` and modified ECE summary
values; the saved training cells use target coverage 0.90. No surviving final
assembly script was identified. The numerical final inputs are therefore
authoritative archived artifacts. The runner does not invent missing coverage
or ECE processing or overwrite these values with a newly chosen definition.
The plot script's original handling of nonmonotonic stored intervals is retained.

## Training commands

Remove `--smoke` to use the preserved full settings. Every run writes its actual
cell substitutions and status to `execution.json`. `--output-dir PATH` changes
the output location. Relative paths are resolved before the process changes
working directory, so the commands work from outside the repository too.

```powershell
python workflows/benchmark/run.py train --benchmark fon --stage five-methods
python workflows/benchmark/run.py train --benchmark dtlz2 --stage five-methods
python workflows/benchmark/run.py train --benchmark fon --stage vanilla
python workflows/benchmark/run.py train --benchmark dtlz2 --stage vanilla
python workflows/benchmark/run.py train --benchmark fon --stage cv-metrics
python workflows/benchmark/run.py train --benchmark dtlz2 --stage cv-metrics
python workflows/benchmark/run.py train --benchmark dtlz2 --stage sensitivity
python workflows/benchmark/run.py train --benchmark fon --stage legacy-pareto --method cv
python workflows/benchmark/run.py train --benchmark dtlz2 --stage legacy-pareto --method cv
```

`five-methods --method` can select `vanilla`, `selected`, `bagging`, `split`, or
`cv`; default `all` executes all five entries from the original notebook.
The separate `vanilla` stage is the true independent RBF-GP baseline used by
the final table. The earlier five-method notebook's configurable Vanilla label
is preserved as written: its attached selection flags are not read by the
preserved shared module. No model-selection behavior has been altered.

`legacy-pareto --method raw` selects the notebook's existing
`flag_calibration=False` setting. The saved calibration-on legacy source uses
the CV function. The historical Split CP source version is not present: its
reference CSV is bundled, but the runner does not claim that the current CV
cell reproduces that earlier split-calibrated run.

| Stage | Preserved full settings |
| --- | --- |
| `five-methods`, both | CPU float64; 20 seeds (0–19); 30,000 Sobol pool; 3,000 metric points; FON d=4, N=30, [-1,1]; DTLZ2 d=6, N=50, [0,1]; 20 bags; 5 folds; split fraction .20; target .90; NSGA-II 96 × 60; full refit with frozen scalers |
| `vanilla`, both | Independent single-output RBF GPs; 20 seeds; pool/candidates 30,000; 5,000 metric points; reference PF 1,000; Adam lr .08; FON 120 iterations/N=30, DTLZ2 140/N=50; target .90 |
| `cv-metrics`, FON | 20 seeds; N=30; 10 bags; pool 30,000; metric points 5,000; 100,000 Sobol candidates; target .90; full refit with refitted scalers |
| `cv-metrics`, DTLZ2 | 20 seeds; N=70 from saved CSV; 10 bags; pool 30,000; metric points 5,000; NSGA-II 128 × 100; target .90; frozen scalers |
| `sensitivity`, DTLZ2 | Same current fixed-CV notebook, N=[30,50,90,110] |
| `legacy-pareto`, FON | 20 seeds; N=30; 10 bags; pool 30,000; 100,000 Sobol candidates; full refit with refitted scalers |
| `legacy-pareto`, DTLZ2 | 20 seeds; N=70; 10 bags; pool 30,000; NSGA-II 256 × 100; full refit with refitted scalers |

Smoke runs are explicit execution checks. They use one seed, a 512-point pool,
128 evaluation points where applicable, two bags, and NSGA-II 24 × 3 or 256
Sobol candidates. Vanilla smoke uses eight optimizer iterations and a
128-point reference PF. Sensitivity smoke selects N=30 only. These reductions
are written to `execution.json`; smoke metrics are not final paper results.
The final sensitivity workbook also includes the N=70 reference from the
separate CV run; its five plotted training sizes are 30, 50, 70, 90, and 110.

## Source extraction and execution-only adaptations

Each file in `source_cells/<profile>/cell_NNN.py` contains the exact original
notebook code cell, indexed from zero. The original notebook hash and extracted
source hash are recorded in `provenance/benchmark.json`. Unrelated exploration,
embedded notebook figures, and copied notebooks are excluded.

| Profile | Original notebook | Selected cells (zero based) |
| --- | --- | --- |
| `fon_five_methods` | `revision/CaseStudy_benchmark_FON_clean_10seeds_5methods_with_metrics.ipynb` | 1–11 |
| `dtlz2_five_methods` | `revision/CaseStudy_benchmark_DTLZ2_clean_20seeds_5methods_with_metrics.ipynb` | 1–11 |
| `fon_vanilla` | `revision/FON_true_vanilla_GP_complete_q25_q75.ipynb` | 1–7 |
| `dtlz2_vanilla` | `revision/DTLZ2_true_vanilla_GP_complete_q25_q75.ipynb` | 1–6 |
| `fon_cv_metrics` | `revision/CaseStudy_benchmark_FON_add_surrogate_calibration_metrics.ipynb` | 2,4,5,26,27,28,29,31,32,34,37,39,40,42 |
| `dtlz2_cv_metrics` | `revision/CaseStudy_benchmark_DTLZ2_cvplus_metrics_fixed.ipynb` | 1–6 |
| `fon_legacy_pareto` | `Surrogate_models_for_small-size_datasets/CaseStudy_benchmark_FON.ipynb` | 2,4,5,26,27,28,29,30,32,35,37,38,40 |
| `dtlz2_legacy_pareto` | `Surrogate_models_for_small-size_datasets/CaseStudy_benchmark_DTLZ2.ipynb` | 2,3,6,7,29,30,31,32,33,34,36,41,42,48,50,52 |

The DTLZ2 true-vanilla notebook cell 2 uses `BENCHMARK_NAME` before cell 4
assigns it. The wrapper supplies the identical value before running cell 2.
The current legacy DTLZ2 trial returns `(metrics, model)`, while its seed loop
expects a metrics dictionary. The wrapper unpacks that tuple and discards the
returned model; the trial calculations remain unchanged. Notebook `display`
calls are suppressed on the command line. Output paths are redirected.

For DTLZ2 `cv-metrics`, the current sensitivity loop is restricted to N=70,
following the archived CV metric file; `sensitivity` runs the saved list.
The current legacy DTLZ2 export filename misleadingly says `normal_cali` despite
calling CV internally; output filenames identify the selected method instead.

`Surrogate_models_for_small-size_datasets/CaseStudy_benchmark_DTLZ2 - Copy.ipynb`
actually contains a Branin–Currin case, so it is not a DTLZ2 source. Both clean
notebooks use 20 seeds despite stale 10-seed names/comments. Shared modules in
`legacy/` are byte-identical between the original case-study folder and
`revision/`. The unrelated `surrogate_conformal_rebuild` implementation is not
used.

Dependencies are the repository's recorded Python environment: NumPy, pandas,
PyTorch, GPyTorch, BoTorch, SciPy, scikit-learn, Matplotlib, and openpyxl/Jinja2
for figure/table export. The preserved NSGA-II is implemented in the shared
module and does not require pymoo.

## Validation performed

The original figure scripts completed with the packaged data. The final main
summary numeric/formatted tables and calibration diagonal match their archived
CSV values. Sensitivity/timing values also match; portable source paths and the
current cost script's added `Runs` column are the export differences noted above.
All 270 identified legacy Pareto aggregate values pass the lineage check.

Eight reduced training checks completed: FON/DTLZ2 true vanilla; DTLZ2 selected
GP from the five-method runner; FON CV from the five-method runner; FON/DTLZ2 CV
metrics; DTLZ2 legacy CV Pareto; and DTLZ2 sensitivity at N=30. Three regression
checks pass for multi-batch failure reporting and repository-root protection:

```powershell
python workflows/benchmark/test_orchestration.py
```

All 126 archived source/reference files match their recorded hashes, including
84 code cells matching the exact UTF-8 source bytes of their original notebooks.
Full 20-seed training was not executed during packaging.

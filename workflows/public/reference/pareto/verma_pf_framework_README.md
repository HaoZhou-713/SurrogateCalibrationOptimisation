# Verma PF framework comparison

Run from the repository root:

```powershell
python 'Surrogate_models_for_small-size_datasets/pareto_results/plot_verma_pf_framework_comparison.py'
```

The script resolves its default inputs relative to itself, so it also runs from
other working directories. Options: `--pf-store`, `--observed`, `--output-dir`,
`--dpi` (at least 600), and `--width` (inches, default 7.2). Dependencies: NumPy,
pandas, SciPy, scikit-learn, Matplotlib, and openpyxl. No optimisation or model
fitting is invoked.

## Inputs and source logic

- `verma_pf_store.pkl`: `pf_store[method]` lists `(seed, X_pareto, Y_pareto)`.
  Methods are `raw`, `raw_uncertainty`, `cfsc`; each contains seeds 0–9 and
  256 solutions per seed. `X_pareto` columns are V, T_in, AR, IA;
  `Y_pareto` columns are T (K), P (Pa), TD (K).
- `../../battery_data_Verma_modify.xlsx`: 100 complete observed rows, with
  T, P, TD selected in that order. This is the data file read by the source
  notebook; no observations were dropped or imputed.
- `../CaseStudy_battery_Ver.ipynb`, zero-based cells 30–36: stored-front
  generation, support plotting, and quantitative convex-hull membership.
- `../surrogate_pareto_botorch.py`: final Pareto extraction is performed on
  the full three-objective framework-specific values. The saved `Y_pareto`
  coordinates are the corresponding predictive means.

The uncertainty-adjusted objective values and predictive standard deviations
are not in the pickle. Original uncertainty-aware nondominance therefore
cannot be independently revalidated from this store alone. Every original
recommendation is retained; no 2D or 3D Pareto filtering is applied here.

## Numerical definitions

Normalize each objective with its minimum and maximum over **all saved PF
solutions from every method and run**. The observed data do not determine
these ranges; the same affine transformation is applied to them for distances.
The precise ranges are recorded in `verma_pf_framework_provenance.json`.

For normalized point sets A and B, use symmetric Hausdorff distance:

```text
H(A,B) = max(max_{a in A} min_{b in B} ||a-b||_2,
             max_{b in B} min_{a in A} ||b-a||_2).
```

The medoid minimizes the mean distance to all other runs of its method.
Exact ties select the smallest seed ID. Dispersion is the mean over unordered
pairs of distinct runs. Both calculations use actual saved 3D point sets.
One selected run is reused in all three projections. No coordinates are
averaged and no solutions are thinned, deduplicated or connected by arbitrary
lines. A single run would give an undefined (NaN) dispersion.

Quantitative support is the **full 3D convex hull of the observed objectives**,
matching `compute_nonphysical_rate(..., method="convex_hull")` in the notebook.
The original half-space rule with tolerance 1e-10 is preserved and checked for
consistency after normalization. Distance is zero inside; outside, it is the
minimum Euclidean distance to the normalized hull's triangular faces,
including their edges and vertices. It is not nearest-observation distance.
Outside-support status indicates objective-space extrapolation, not a proof
of physical infeasibility.

`negative_pressure_rate` uses P < 0 Pa. Both rate columns are arithmetic means
of per-run fractions, matching the notebook. Each current run has equal size,
so they also equal the pooled fractions. Distance summaries pool all 2,560
saved solutions per method, including zero distances and duplicate rows;
the 95th percentile uses NumPy's linear interpolation. All optional requested
metrics are available from these inputs.

## Figures and outputs

- `verma_pf_framework_comparison.pdf` and `.png`: main 1×3 figure,
  (a) T–P, (b) T–TD, (c) P–TD. Faint markers (alpha 0.10) show all runs;
  stronger markers show each medoid. All 100 observed points and their
  projected convex hull form a shared background.
- `verma_pf_all_runs_supplementary.pdf` and `.png`: 3×3 diagnostic, with
  the same methods as columns and projections as rows. Preserves the original
  **method-specific union of k=5 nearest observed neighbours selected in
  unnormalized 3D space**, and its projected hull. All saved PF runs are shown
  with a fixed method colour. The original diagnostic files are untouched.
- `verma_pf_framework_summary.csv`: requested three-row method summary.
- `verma_pf_run_summary.csv`: all run scores, rates, and medoid indicators.
- `verma_pf_pairwise_hausdorff.csv`: all 135 within-method run-pair distances.
- `verma_pf_point_support_metrics.csv`: all 7,680 saved points, original row
  indices, and support/pressure metrics for audit.
- `verma_pf_framework_comparison_caption.tex`: suggested LaTeX caption.
- `verma_pf_framework_provenance.json`: definitions, metadata and SHA-256
  checksums confirming the inputs and four original figure files are unchanged.

PDFs contain vector elements and embedded TrueType fonts. PNGs use 600 dpi.
The main figure is 7.2 × 3.05 inches. Charcoal Raw and purple CFSC preserve
`revision/results/For paper/generate_section6_public_btms_figures.py`'s PF
palette; orange is added for conventional GP uncertainty. Shapes also
distinguish methods: circles, triangles and squares, with grey crosses for
observations.

## Validation and interpretation

```powershell
python -m unittest discover -s 'Surrogate_models_for_small-size_datasets/pareto_results' -p 'test_verma_pf_framework_comparison.py'
```

Numerical tests cover Hausdorff symmetry and its maximum-distance definition,
medoid selection and dispersion denominators, exact hull distances at faces,
edges and vertices, the single-run case, and preservation of points that
appear dominated in projection. Additional generation QA compared every
Hausdorff pair with a dense distance calculation and 50 outside-support
distances with independent constrained quadratic minimisation (maximum absolute
discrepancy below 4e-15). All saved coordinate rows were checked against the
pickle; image resolution, vector PDF content, and original-file hashes passed.

The medoid seeds are Raw 2, Raw + GP uncertainty 2, CFSC 8. PF dispersions are
approximately 0.319321, 0.325102, and 0.369770, respectively. CFSC has lower
negative-pressure and outside-support rates but **higher Hausdorff dispersion**
in these saved runs. This figure and table should not be cited as evidence that
CFSC reduces run-to-run PF dispersion.

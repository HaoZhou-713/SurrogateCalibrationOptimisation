# Reproducibility repository design and execution plan

## Scope
Create a standalone Git repository inside physics_test that reproduces the final benchmark, public-data validation/Pareto, and in-house validation results. Preserve selected original scientific code, parameters and datasets. Changes are limited to execution order, explicit paths, command-line entry points, dependency capture, and provenance. Original project files are read-only inputs.

## Architecture
- `workflows/benchmark`, `workflows/public`, `workflows/inhouse`: independent runnable workflows with selected source snapshots, local data/reference results, and a `run.py` CLI.
- Each workflow supports `figures` from saved numerical artifacts and `train` from original data, with explicitly labelled `--smoke` for reduced runtime. Separate reference results from generated outputs.
- `reproduce.py`: front door for listing workflows, verifying manifest hashes, drawing all final figures, and dispatching a selected workflow.
- `provenance`: relative original paths, SHA-256 hashes, notebook cell indices, exact extraction/path changes, and documented limits on historical result matching.
- `requirements.txt` and `environment.yml`: versions from the notebook's arbo environment. No machine-specific paths in runnable code.
- Outputs live under an ignored `outputs/` directory; archived inputs and reference artifacts are never overwritten.

## Deliverables and checks
- [x] Benchmark: identify final table lineage, retain exact plot code and reference tables, expose original training runners and final aggregation.
- [x] Public datasets: preserve nested repeated validation separately from final Li/Verma Pareto generation and plotting.
- [x] In-house: preserve final three-framework training/optimization and recreate validation figure using supplied simulation truth.
- [x] Shared repository: portable CLI, pinned environment, Chinese quickstart and result/source map, provenance verification.
- [x] Validation: compile scripts; verify source hashes; recreate final plots/tables; run bounded model/training smoke checks; verify execution from a different working directory; initialize local Git repository.

## Acceptance and limits
The repo must function without the parent physics_test directory. Archived reference results must be clearly distinguished from freshly retrained results. Full repeated GP training can be computationally expensive; document which jobs were actually executed and do not claim full numerical reproduction from smoke runs. Externally generated simulation truth is an input, unless the original simulator is present. No algorithmic audit or optimization.

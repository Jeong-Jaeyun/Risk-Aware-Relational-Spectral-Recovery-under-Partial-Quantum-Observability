# biqmn

`biqmn` contains the simulation and analysis code for the relational spectral
recovery experiments reported in the manuscript. The main reproducibility target
is the C3R robust-policy analysis over clean, partial-syndrome, noisy-syndrome,
partial-plus-noisy, and ambiguity-plus-measurement regimes.

## Environment

The project was developed and run with the `QEC` conda environment.

```powershell
cd biqmn
conda activate QEC
python -m pip install -e ".[dev]"
```

Core runtime dependencies are declared in `pyproject.toml`:

- `numpy`
- `scipy`
- `PyYAML`
- `qiskit`
- `qiskit-aer`
- `matplotlib`

## Package Layout

```text
biqmn/core/             Clock conditioning, graph/Laplacian maps, trajectories,
                        admissibility, recovery objectives, and noise models
biqmn/baselines/        Syndrome-recovery baseline utilities
biqmn/experiments/      Experiment runners and result-table builders
configs/                YAML fragments for states, noise models, and experiments
tests/                  Unit and smoke tests for theory alignment and experiments
results/raw/            Raw JSON payloads used to rebuild reported tables
results/tables/         CSV/Markdown derived tables
results/plots/          Generated diagnostic figures
scripts/                Reproduction helpers
```

## Basic Validation

Run the test suite from the package root:

```powershell
python -m pytest
```

For a quick import/config check:

```powershell
python -c "import biqmn; print(biqmn.__version__)"
```

## Reproducing the Reported C3R Results

The full C3R batch can be run with:

```powershell
.\scripts\run_c3r_seed10.ps1
```

Default settings:

- conda environment: `QEC`
- seeds: `11,12,13,14,15,16,17,18,19,20`
- output location: `results/raw`, `results/tables`, and `results/plots`
- output suffix: `260427_seed10`

The script runs these modules in sequence:

```text
biqmn.experiments.run_hybrid_c123_regime_map
biqmn.experiments.run_partial_syndrome_baseline
biqmn.experiments.run_noisy_syndrome_baseline
biqmn.experiments.run_partial_noisy_syndrome_regime_map
biqmn.experiments.run_ambiguity_measurement_syndrome_regime_map
```

The same modules can be run manually. For example:

```powershell
python -m biqmn.experiments.run_partial_syndrome_baseline `
  --seeds 11,12,13,14,15,16,17,18,19,20 `
  --output-stem partial_syndrome_c3r_260427_seed10 `
  --plot-prefix partial_syndrome_c3r_260427_seed10
```

## Rebuilding Tables From Existing Raw Results

If the raw JSON files already exist, derived CSV and Markdown tables can be
rebuilt without rerunning simulations:

```powershell
.\scripts\rebuild_c3r_results.ps1
```

This calls:

```powershell
python -m biqmn.experiments.rebuild_c3r_results
```

By default, it rebuilds the five reported C3R stems:

```text
hybrid_c123_regime_map_c3r_260427_seed10
partial_syndrome_c3r_260427_seed10
noisy_syndrome_c3r_260427_seed10
partial_noisy_syndrome_c3r_260427_seed10
ambiguity_measurement_c3r_260427_seed10
```

## Result Files Used by the Manuscript

The current manuscript tables are based on:

```text
results/raw/*.json
results/tables/*.csv
results/tables/*.md
```

`result_c3r_260525/` is a preserved copy of a previous C3R result snapshot. The
active manuscript-facing result tree is `results/`.

## Notes for Reviewers

The code is deterministic for a fixed seed list and configuration set. Runtime
depends on the local BLAS/Qiskit installation; the full ten-seed C3R batch is a
long run, while rebuilding tables from raw JSON is fast.

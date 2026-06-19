# Risk-Aware Relational Spectral Recovery under Partial Quantum Observability

Simulation and analysis code for relational spectral recovery under partial
quantum observability. This release includes the canonical stabilizer-code
extension reported in **Uncertainty-Gated Structural Recovery under Syndrome
Ambiguity across Canonical Stabilizer Codes**.

The Phase-2 study evaluates the same fixed recovery-policy geometry on the
perfect `[[5,1,3]]`, Steane `[[7,1,3]]`, and Shor `[[9,1,3]]` codes without
code-specific policy retuning.

## Public artifacts

- Complete 64,800-case dataset:
  <https://huggingface.co/datasets/nogalee/uncertainty-gated-structural-recovery-64800-cases>
- Archived Phase-1 release:
  <https://doi.org/10.5281/zenodo.19928607>

The large case-level tables are hosted on Hugging Face rather than duplicated
in Git history. The dataset includes the five raw regime tables, a combined
Parquet table, aggregate audit tables, configurations, and SHA-256 manifest.

## Environment

The Phase-2 experiments were developed and run with the `Qml` conda environment.

```powershell
cd biqmn
conda activate Qml
python -m pip install -e ".[dev]"
```

Core runtime dependencies are declared in `pyproject.toml`:

- `numpy`
- `scipy`
- `PyYAML`
- `qiskit`
- `qiskit-aer`
- `matplotlib`
- `tqdm`

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

Top-level analysis helpers:

```text
analyze_recovery_failure_linkage.py  Mechanism and failure-linkage audits
make_paper_figures.py                Manuscript figure generation
prepare_hf_dataset.py                Dataset packaging and SHA-256 manifest
```

## Basic Validation

Run the test suite from the package root:

```powershell
python -m pytest
```

The Phase-2 theory-alignment suite was validated with 33 passing tests and 9
passing subtests in the declared Qiskit Aer environment.

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

- conda environment: `Qml`
- seeds: `11,12,13,14,15,16,17,18,19,20`
- output location: `results/raw`, `results/tables`, and `results/plots`
- output suffix: `phase2_seed10`
- workers: `8`

The complete Phase-2 grid contains:

1. clean syndrome information (3,600 cases);
2. partial syndrome observation (14,400 cases);
3. noisy syndrome observation (18,000 cases);
4. partial plus noisy syndrome observation (10,800 cases);
5. ambiguity plus measurement/reset stress (18,000 cases).

Total: **64,800 cases**. The runner stores per-case checkpoints and resumes an
incomplete grid when invoked again with the same output suffix. Pass `-Force`
to recompute an existing output.

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
  --output-stem partial_syndrome_c3r_phase2_seed10 `
  --plot-prefix partial_syndrome_c3r_phase2_seed10 `
  --workers 8
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
hybrid_c123_regime_map_c3r_phase2_seed10
partial_syndrome_c3r_phase2_seed10
noisy_syndrome_c3r_phase2_seed10
partial_noisy_syndrome_c3r_phase2_seed10
ambiguity_measurement_c3r_phase2_seed10
```

## Published result files

The complete case-level and aggregate result files are published at:

<https://huggingface.co/datasets/nogalee/uncertainty-gated-structural-recovery-64800-cases>

To build the Hugging Face package from the five final raw CSV tables, run:

```powershell
python prepare_hf_dataset.py
```

The packaging script validates all five regime row counts, writes the combined
Parquet table, copies aggregate tables/configurations, and records SHA-256
checksums in `dataset_manifest.json`.

## Notes for Reviewers

The code is deterministic for a fixed seed list and configuration set. Runtime
depends on the local BLAS/Qiskit installation; the full ten-seed C3R batch is a
long run, while rebuilding tables from raw JSON is fast.

## Authors

- Jaeyun Jeong ([ORCID: 0009-0005-3281-5858](https://orcid.org/0009-0005-3281-5858))
- HwaYoung Jeong ([ORCID: 0000-0002-5017-934X](https://orcid.org/0000-0002-5017-934X))

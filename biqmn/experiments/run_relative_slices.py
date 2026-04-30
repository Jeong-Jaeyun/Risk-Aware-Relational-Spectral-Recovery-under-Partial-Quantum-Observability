from __future__ import annotations

import json

import numpy as np

from ..core.clock import clock_overlap
from ..core.relative_state import adjacent_fidelity, check_slice_sanity
from ..core.trajectory import spectral_response, trajectory_smoothness_penalty
from .common import (
    build_parser,
    build_pipeline,
    load_config,
    resolve_output_stem,
    write_json_result,
)


def run(config: dict) -> dict:
    bundle = build_pipeline(config, with_noise=False)
    family = bundle["relative_family"]
    tau_grid = bundle["tau_grid"]
    Hc = bundle["Hc"]
    psi0_clock = bundle["psi0_clock"]
    traj = bundle["trajectory"]

    slice_checks = [check_slice_sanity(rho) for rho in family]
    adjacent_slice_fidelity = adjacent_fidelity(family).tolist()
    adjacent_clock_overlap_abs = [
        float(abs(clock_overlap(Hc, tau_grid[idx], tau_grid[idx + 1], psi0_clock)))
        for idx in range(len(tau_grid) - 1)
    ]
    spectral_steps = [
        spectral_response(traj.spectra[idx], traj.spectra[idx + 1])
        for idx in range(len(traj) - 1)
    ]

    summary = {
        "n_slices": len(family),
        "all_valid": bool(all(item["is_valid"] for item in slice_checks)),
        "trace_min": float(min(item["trace"] for item in slice_checks)),
        "trace_max": float(max(item["trace"] for item in slice_checks)),
        "min_eig_min": float(min(item["min_eig"] for item in slice_checks)),
        "adjacent_fidelity_mean": float(np.mean(adjacent_slice_fidelity)),
        "adjacent_clock_overlap_mean_abs": float(np.mean(adjacent_clock_overlap_abs)),
        "trajectory_smoothness": float(trajectory_smoothness_penalty(traj)),
        "constraint_residual": float(bundle["meta"]["constraint_residual"]),
        "nullspace_dim": int(bundle["meta"]["nullspace_dim"]),
    }
    return {
        "summary": summary,
        "tau_grid": tau_grid,
        "slice_checks": slice_checks,
        "adjacent_slice_fidelity": adjacent_slice_fidelity,
        "adjacent_clock_overlap_abs": adjacent_clock_overlap_abs,
        "spectral_steps": spectral_steps,
        "spectra": traj.spectra,
        "meta": bundle["meta"],
    }


def main() -> None:
    parser = build_parser(
        description="Run the relative slicing sanity probe.",
        default_experiment="experiment/trajectory_probe.yaml",
        default_state="states/null_dynamic.yaml",
    )
    args = parser.parse_args()
    config = load_config(
        experiment_config=args.config,
        state_config=args.state_config,
    )
    result = run(config)
    stem = resolve_output_stem(config, "relative_slices", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

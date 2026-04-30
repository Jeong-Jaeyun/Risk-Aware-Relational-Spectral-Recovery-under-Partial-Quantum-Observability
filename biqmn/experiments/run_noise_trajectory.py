from __future__ import annotations

import json

import numpy as np

from ..core.laplacian import weyl_shift_estimate
from ..core.metrics import trace_distance
from ..core.trajectory import spectral_response, trajectory_distance
from .common import (
    build_parser,
    build_pipeline,
    load_config,
    resolve_output_stem,
    write_json_result,
)


def run(config: dict) -> dict:
    clean = build_pipeline(config, with_noise=False)
    noisy = build_pipeline(config, with_noise=True)
    clean_traj = clean["trajectory"]
    noisy_traj = noisy["trajectory"]

    slice_responses = [
        spectral_response(clean_traj.spectra[idx], noisy_traj.spectra[idx])
        for idx in range(len(clean_traj))
    ]
    slice_trace_distances = [
        trace_distance(clean["relative_family"][idx], noisy["relative_family"][idx])
        for idx in range(len(clean["relative_family"]))
    ]
    weyl = [
        weyl_shift_estimate(clean_traj.laplacians[idx], noisy_traj.laplacians[idx])
        for idx in range(len(clean_traj))
    ]

    summary = {
        "trajectory_distance": float(trajectory_distance(clean_traj, noisy_traj)),
        "mean_slice_response": float(np.mean(slice_responses)),
        "max_slice_response": float(np.max(slice_responses)),
        "mean_state_trace_distance": float(np.mean(slice_trace_distances)),
        "mean_weyl_frobenius": float(np.mean([item["frobenius"] for item in weyl])),
        "noise_steps": len(noisy["meta"]["applied_noise"]),
    }
    return {
        "summary": summary,
        "tau_grid": clean["tau_grid"],
        "slice_responses": slice_responses,
        "slice_trace_distances": slice_trace_distances,
        "weyl_estimates": weyl,
        "clean_spectra": clean_traj.spectra,
        "noisy_spectra": noisy_traj.spectra,
        "meta": {
            "clean": clean["meta"],
            "noisy": noisy["meta"],
        },
    }


def main() -> None:
    parser = build_parser(
        description="Compare clean and noisy spectral trajectories.",
        default_experiment="experiment/trajectory_probe.yaml",
        default_state="states/null_dynamic.yaml",
        default_noise="noise/dephasing.yaml",
    )
    args = parser.parse_args()
    config = load_config(
        experiment_config=args.config,
        state_config=args.state_config,
        noise_config=args.noise_config,
    )
    result = run(config)
    stem = resolve_output_stem(config, "noise_trajectory", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

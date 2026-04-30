from __future__ import annotations

import json

from ..core.clock_consistency import calibrate_clock_eta, clock_penalty
from .common import (
    build_parser,
    build_pipeline,
    build_reference_bank,
    load_config,
    resolve_output_stem,
    write_json_result,
)


def run(config: dict) -> dict:
    reference_bundles = build_reference_bank(config)
    reference_trajs = [bundle["trajectory"] for bundle in reference_bundles]

    target_clean = build_pipeline(config, with_noise=False)
    target_noisy = build_pipeline(config, with_noise=True)

    calibration = calibrate_clock_eta(
        reference_trajs,
        target_clean["Hc"],
        target_clean["psi0_clock"],
        kappa=float(config.get("clock_consistency", {}).get("kappa", 3.0)),
    )
    clean_penalty = clock_penalty(
        target_clean["trajectory"],
        target_clean["Hc"],
        target_clean["psi0_clock"],
    )
    noisy_penalty = clock_penalty(
        target_noisy["trajectory"],
        target_noisy["Hc"],
        target_noisy["psi0_clock"],
    )

    summary = {
        "reference_count": len(reference_bundles),
        "eta_threshold": float(calibration["eta"]),
        "phi_clean": float(clean_penalty["phi"]),
        "phi_noisy": float(noisy_penalty["phi"]),
        "phi_threshold": float(calibration["phi_threshold"]),
    }
    return {
        "summary": summary,
        "reference_labels": [bundle["label"] for bundle in reference_bundles],
        "calibration": calibration,
        "clean_penalty": clean_penalty,
        "noisy_penalty": noisy_penalty,
    }


def main() -> None:
    parser = build_parser(
        description="Calibrate eta and evaluate the clock consistency penalty.",
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
    stem = resolve_output_stem(config, "clock_consistency", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

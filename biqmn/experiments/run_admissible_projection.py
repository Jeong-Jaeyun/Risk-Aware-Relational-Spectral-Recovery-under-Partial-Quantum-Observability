from __future__ import annotations

import json

from ..core.admissibility import admissibility_report, calibrate_thresholds
from ..core.clock_consistency import calibrate_clock_eta
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

    clock_cal = calibrate_clock_eta(
        reference_trajs,
        target_clean["Hc"],
        target_clean["psi0_clock"],
        kappa=float(config.get("clock_consistency", {}).get("kappa", 3.0)),
    )
    threshold_report = calibrate_thresholds(
        reference_trajs,
        target_clean["Hc"],
        target_clean["psi0_clock"],
        kappa=float(config.get("admissibility", {}).get("kappa", 3.0)),
    )
    thresholds = {
        key: value["threshold"]
        for key, value in threshold_report.items()
        if key.startswith("phi_")
    }
    clean_report = admissibility_report(
        target_clean["trajectory"],
        reference_trajs,
        target_clean["Hc"],
        target_clean["psi0_clock"],
        thresholds,
    )
    noisy_report = admissibility_report(
        target_noisy["trajectory"],
        reference_trajs,
        target_noisy["Hc"],
        target_noisy["psi0_clock"],
        thresholds,
    )

    summary = {
        "reference_count": len(reference_bundles),
        "clean_admissible": bool(clean_report["admissible"]),
        "noisy_admissible": bool(noisy_report["admissible"]),
        "phi_ref_threshold": float(thresholds["phi_ref"]),
        "phi_clock_threshold": float(thresholds["phi_clock"]),
    }
    return {
        "summary": summary,
        "reference_labels": [bundle["label"] for bundle in reference_bundles],
        "clock_calibration": clock_cal,
        "threshold_report": threshold_report,
        "thresholds": thresholds,
        "clean_report": clean_report,
        "noisy_report": noisy_report,
    }


def main() -> None:
    parser = build_parser(
        description="Calibrate admissibility thresholds and score clean/noisy trajectories.",
        default_experiment="experiment/recovery_eval.yaml",
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
    stem = resolve_output_stem(config, "admissible_projection", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

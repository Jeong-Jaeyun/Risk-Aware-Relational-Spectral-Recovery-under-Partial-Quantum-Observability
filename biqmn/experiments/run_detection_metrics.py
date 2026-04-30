from __future__ import annotations

import json

import numpy as np

from ..core.spectral_density import (
    auto_bandwidth,
    auto_grid,
    hybrid_detect_score,
    kde_spectral_density,
)
from ..core.trajectory import trajectory_distance
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

    detection_cfg = config.get("detection", {})
    grid = auto_grid(clean_traj.spectra + noisy_traj.spectra, n_grid=int(detection_cfg.get("n_grid", 256)))
    sigma_cfg = detection_cfg.get("sigma", "auto")
    if sigma_cfg == "auto":
        sigma = max(auto_bandwidth(spec) for spec in clean_traj.spectra + noisy_traj.spectra)
    else:
        sigma = float(sigma_cfg)

    per_slice = []
    for idx in range(len(clean_traj)):
        p = kde_spectral_density(clean_traj.spectra[idx], grid, sigma)
        q = kde_spectral_density(noisy_traj.spectra[idx], grid, sigma)
        metrics = hybrid_detect_score(
            p,
            q,
            grid,
            alpha=float(detection_cfg.get("alpha", 0.5)),
            kl_scale=float(detection_cfg.get("kl_scale", 1.0)),
            w1_scale=float(detection_cfg.get("w1_scale", 1.0)),
        )
        metrics["index"] = idx
        per_slice.append(metrics)

    summary = {
        "trajectory_distance": float(trajectory_distance(clean_traj, noisy_traj)),
        "sigma": float(sigma),
        "kl_mean": float(np.mean([item["kl"] for item in per_slice])),
        "js_mean": float(np.mean([item["js"] for item in per_slice])),
        "w1_mean": float(np.mean([item["w1"] for item in per_slice])),
        "score_mean": float(np.mean([item["score"] for item in per_slice])),
        "score_max": float(np.max([item["score"] for item in per_slice])),
    }
    return {
        "summary": summary,
        "grid": grid,
        "per_slice": per_slice,
        "meta": {
            "clean": clean["meta"],
            "noisy": noisy["meta"],
        },
    }


def main() -> None:
    parser = build_parser(
        description="Compute KL/JSD/W1 detection scores on clean vs noisy trajectories.",
        default_experiment="experiment/detection_eval.yaml",
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
    stem = resolve_output_stem(config, "detection_metrics", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

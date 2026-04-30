from __future__ import annotations

import json

from ..core.admissibility import admissibility_report, calibrate_thresholds
from ..core.clock_consistency import calibrate_clock_eta
from ..core.recovery import two_stage_admissible_recovery
from ..core.trajectory import trajectory_distance
from .common import (
    build_parser,
    build_pipeline,
    build_reference_bank,
    load_config,
    resolve_output_stem,
    write_json_result,
)


def _term_delta(new_terms: dict | None, old_terms: dict | None) -> dict[str, float] | None:
    if new_terms is None or old_terms is None:
        return None
    keys = set(new_terms) | set(old_terms)
    return {
        key: float(new_terms.get(key, 0.0) - old_terms.get(key, 0.0))
        for key in sorted(keys)
    }


def _weighted_terms(terms: dict | None, weights: dict) -> dict[str, float] | None:
    if terms is None:
        return None
    return {
        key: float(weights.get(key, 0.0) * value)
        for key, value in terms.items()
    }


def run(config: dict) -> dict:
    reference_bundles = build_reference_bank(config)
    reference_trajs = [bundle["trajectory"] for bundle in reference_bundles]
    reference_labels = [bundle["label"] for bundle in reference_bundles]

    clean = build_pipeline(config, with_noise=False)
    observed = build_pipeline(config, with_noise=True)

    eta = calibrate_clock_eta(
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        kappa=float(config.get("clock_consistency", {}).get("kappa", 3.0)),
    )["eta"]
    threshold_report = calibrate_thresholds(
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        kappa=float(config.get("admissibility", {}).get("kappa", 3.0)),
        eta=eta,
    )
    thresholds = {
        key: value["threshold"]
        for key, value in threshold_report.items()
        if key.startswith("phi_")
    }
    recovery_cfg = config.get("recovery", {})
    weights = recovery_cfg.get("weights", recovery_cfg.get("betas", {}))
    stage2_cfg = recovery_cfg.get("stage2", {})

    recovery = two_stage_admissible_recovery(
        observed["trajectory"],
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        stage2_enabled=bool(stage2_cfg.get("enabled", True)),
        weights=weights,
        ref_bank=reference_trajs,
        eta=eta,
        thresholds=thresholds,
        stage2_local_count=int(stage2_cfg.get("local_count", 3)),
        stage2_grid_steps=int(stage2_cfg.get("grid_steps", 10)),
        stage2_require_interior=bool(stage2_cfg.get("require_interior", True)),
        stage2_apply_rule=str(stage2_cfg.get("apply_rule", "diagnostic_only")),
    )
    stage1 = recovery["stage1"]
    stage2 = recovery["stage2"]
    stage1_recovered = stage1["recovered"]
    reference_anchor = stage1["reference_anchor"]
    stage2_candidate = stage2["best_admissible_recovered"]
    recovered = recovery["final_recovered"]
    observed_report = admissibility_report(
        observed["trajectory"],
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        thresholds,
        eta=eta,
    )
    stage1_report = admissibility_report(
        stage1_recovered,
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        thresholds,
        eta=eta,
    )
    recovered_report = admissibility_report(
        recovered,
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        thresholds,
        eta=eta,
    )
    stage2_candidate_report = (
        None
        if stage2_candidate is None
        else admissibility_report(
            stage2_candidate,
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            thresholds,
            eta=eta,
        )
    )
    stage2_candidate_clean_distance = (
        None
        if stage2_candidate is None
        else float(trajectory_distance(clean["trajectory"], stage2_candidate))
    )
    stage2_candidate_observed_distance = (
        None
        if stage2_candidate is None
        else float(trajectory_distance(observed["trajectory"], stage2_candidate))
    )
    clean_to_stage1_distance = float(
        trajectory_distance(clean["trajectory"], stage1_recovered)
    )
    clean_to_observed_distance = float(
        trajectory_distance(clean["trajectory"], observed["trajectory"])
    )
    clean_to_recovered_distance = float(
        trajectory_distance(clean["trajectory"], recovered)
    )
    stage2_candidate_clean_delta = (
        None
        if stage2_candidate_clean_distance is None
        else float(stage2_candidate_clean_distance - clean_to_stage1_distance)
    )
    final_clean_delta = float(clean_to_recovered_distance - clean_to_stage1_distance)
    observed_to_stage1_distance = float(
        trajectory_distance(observed["trajectory"], stage1_recovered)
    )
    observed_to_recovered_distance = float(
        trajectory_distance(observed["trajectory"], recovered)
    )
    summary = {
        "candidate_count": len(reference_trajs),
        "stage1_best_index": int(stage1["best_index"]),
        "stage1_best_label": reference_labels[stage1["best_index"]],
        "reference_anchor_index": int(stage1["reference_anchor_index"]),
        "reference_anchor_label": reference_labels[stage1["reference_anchor_index"]],
        "reference_anchor_distance_squared": float(stage1["reference_anchor_distance_squared"]),
        "stage1_objective": float(stage1["best_objective"]),
        "stage1_recovered_admissible": bool(stage1_report["admissible"]),
        "stage2_applied": bool(recovery["stage2_applied"]),
        "stage2_apply_rule": recovery["stage2_apply_rule"],
        "stage2_apply_reason": recovery["stage2_apply_reason"],
        "stage2_feasible_count": int(stage2["feasible_count"]),
        "stage2_interior_feasible_count": int(stage2["interior_feasible_count"]),
        "stage2_best_admissible_objective": (
            None
            if stage2["best_admissible_objective"] is None
            else float(stage2["best_admissible_objective"])
        ),
        "stage2_best_admissible_is_interior": (
            None
            if stage2["best_admissible_is_interior"] is None
            else bool(stage2["best_admissible_is_interior"])
        ),
        "stage2_candidate_objective_gain_vs_stage1": float(
            recovery["stage2_candidate_objective_gain"]
        ),
        "stage2_objective_gain_vs_stage1": float(recovery["stage2_objective_gain"]),
        "final_stage": recovery["final_stage"],
        "recovered_objective": float(recovery["final_objective"]),
        "objective_gain_vs_reference_anchor": float(
            stage1["objective_gain_vs_reference_anchor"]
        ),
        "observed_admissible": bool(observed_report["admissible"]),
        "recovered_admissible": bool(recovered_report["admissible"]),
        "clean_to_stage1_distance": clean_to_stage1_distance,
        "clean_to_observed_distance": clean_to_observed_distance,
        "clean_to_recovered_distance": clean_to_recovered_distance,
        "clean_to_stage2_candidate_distance": stage2_candidate_clean_distance,
        "stage2_candidate_clean_distance_delta_vs_stage1": stage2_candidate_clean_delta,
        "stage2_clean_distance_delta_vs_stage1": stage2_candidate_clean_delta,
        "final_clean_distance_delta_vs_stage1": final_clean_delta,
        "observed_to_stage1_distance": observed_to_stage1_distance,
        "observed_to_stage2_candidate_distance": stage2_candidate_observed_distance,
        "observed_to_recovered_distance": observed_to_recovered_distance,
        "clean_to_reference_anchor_distance": float(
            trajectory_distance(clean["trajectory"], reference_anchor)
        ),
    }
    return {
        "summary": summary,
        "reference_labels": reference_labels,
        "threshold_report": threshold_report,
        "thresholds": thresholds,
        "observed_report": observed_report,
        "stage1_report": stage1_report,
        "stage2_candidate_report": stage2_candidate_report,
        "recovered_report": recovered_report,
        "term_decomposition": {
            "observed_terms": stage1["observed_terms"],
            "stage1_terms": stage1["best_terms"],
            "stage2_candidate_terms": stage2["best_admissible_terms"],
            "final_terms": (
                stage2["best_admissible_terms"]
                if recovery["final_stage"] == "stage2"
                else stage1["best_terms"]
            ),
            "reference_anchor_terms": stage1["reference_anchor_terms"],
            "observed_weighted_terms": _weighted_terms(stage1["observed_terms"], weights),
            "reference_anchor_weighted_terms": _weighted_terms(stage1["reference_anchor_terms"], weights),
            "stage1_weighted_terms": _weighted_terms(stage1["best_terms"], weights),
            "stage2_candidate_weighted_terms": _weighted_terms(stage2["best_admissible_terms"], weights),
            "final_weighted_terms": _weighted_terms(
                stage2["best_admissible_terms"] if recovery["final_stage"] == "stage2" else stage1["best_terms"],
                weights,
            ),
            "delta_stage1_vs_observed": _term_delta(stage1["best_terms"], stage1["observed_terms"]),
            "delta_stage1_weighted_vs_observed": _term_delta(
                _weighted_terms(stage1["best_terms"], weights),
                _weighted_terms(stage1["observed_terms"], weights),
            ),
            "delta_stage2_candidate_vs_stage1": _term_delta(
                stage2["best_admissible_terms"], stage1["best_terms"]
            ),
            "delta_stage2_candidate_weighted_vs_stage1": _term_delta(
                _weighted_terms(stage2["best_admissible_terms"], weights),
                _weighted_terms(stage1["best_terms"], weights),
            ),
            "delta_final_vs_stage1": _term_delta(
                stage2["best_admissible_terms"] if recovery["final_stage"] == "stage2" else stage1["best_terms"],
                stage1["best_terms"],
            ),
            "delta_final_weighted_vs_stage1": _term_delta(
                _weighted_terms(
                    stage2["best_admissible_terms"] if recovery["final_stage"] == "stage2" else stage1["best_terms"],
                    weights,
                ),
                _weighted_terms(stage1["best_terms"], weights),
            ),
        },
        "stage1_projection": {
            key: value
            for key, value in stage1.items()
            if key not in {"recovered", "reference_anchor"}
        },
        "stage2_refinement": {
            **{
                key: value
                for key, value in stage2.items()
                if key not in {"best_recovered", "best_admissible_recovered"}
            },
            "local_labels": [
                reference_labels[idx]
                for idx in stage2["local_indices"]
            ],
        },
        "diagnostics": {
            "observed_self_objective": float(stage1["observed_objective"]),
            "observed_self_terms": stage1["observed_terms"],
            "reference_anchor_objective": float(stage1["reference_anchor_objective"]),
            "reference_anchor_terms": stage1["reference_anchor_terms"],
        },
    }


def main() -> None:
    parser = build_parser(
        description="Recover a noisy trajectory via nearest admissible projection.",
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
    stem = resolve_output_stem(config, "recovery_objective", args.output_stem)
    output_path = write_json_result(result, stem)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

"""Encoded [[3,1,1]] repetition-code baseline: syndrome, relational, and hybrid.

For every sampled CPTP noise schedule and every selected code (bitflip /
phaseflip) the runner builds a clean and a noisy pipeline, then evaluates:

    Baseline A  : circuit-level syndrome extraction + Pauli correction
                  (:mod:`biqmn.core.encoding` + :mod:`biqmn.baselines.syndrome_recovery`).
    Baseline B  : relational / admissible two-stage trajectory recovery (the
                  existing :func:`two_stage_admissible_recovery` pathway).
    Baseline C  : syndrome-first hybrid. Keep the syndrome recovery by default,
                  but veto it when it is trajectory-inadmissible, or use the
                  relational candidate as a tie-breaker when the syndrome looks
                  benign yet the relational objective is strictly better.

Logical fidelities, trajectory distances, and the
"syndrome-consistent-but-trajectory-inconsistent" rate are aggregated across
all samples and written to a JSON + CSV + markdown report so the two baselines
can be read side-by-side.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from ..baselines.syndrome_recovery import (
    apply_syndrome_recovery_to_trajectory,
    collect_syndrome_statistics,
    logical_fidelity,
    trajectory_logical_fidelity,
)
from ..core.admissibility import admissibility_report, calibrate_thresholds
from ..core.clock_consistency import calibrate_clock_eta
from ..core.recovery import recovery_objective, two_stage_admissible_recovery
from ..core.trajectory import trajectory_distance
from .common import (
    RESULT_ROOT,
    build_pipeline,
    build_reference_bank,
    load_config,
    resolve_output_stem,
    to_serializable,
    write_json_result,
)
from .run_random_noise_baseline import _sample_schedule, _schedule_signature


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


def _distribution_stats(values: Sequence[float], stem: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{stem}_std": None,
            f"{stem}_median": None,
            f"{stem}_p25": None,
            f"{stem}_p75": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{stem}_std": float(arr.std()),
        f"{stem}_median": float(np.median(arr)),
        f"{stem}_p25": float(np.percentile(arr, 25.0)),
        f"{stem}_p75": float(np.percentile(arr, 75.0)),
    }


def _syndrome_label(raw: Any) -> str:
    bits = [ch for ch in str(raw) if ch in "01"]
    if len(bits) >= 2:
        return "".join(bits[:2])
    return str(raw)


def _syndrome_summary(stats: dict[str, Any]) -> dict[str, Any]:
    labels = []
    per_slice = []
    for item in stats.get("per_slice", []):
        probs = dict(item.get("probabilities", {}))
        if probs:
            dominant_key, dominant_prob = max(probs.items(), key=lambda pair: pair[1])
            dominant_label = _syndrome_label(dominant_key)
        else:
            dominant_label = "??"
            dominant_prob = 0.0
        labels.append(dominant_label)
        per_slice.append(
            {
                "tau_index": int(item.get("tau_index", len(per_slice))),
                "dominant": dominant_label,
                "dominant_probability": float(dominant_prob),
            }
        )
    dominant_label, dominant_rate = _mode_label(labels)
    return {
        "dominant": dominant_label,
        "dominant_rate": dominant_rate,
        "per_slice": per_slice,
        "mean_no_error_probability": float(stats["mean_no_error_probability"]),
        "min_no_error_probability": float(stats["min_no_error_probability"]),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _logical_ket_per_slice(clean_traj) -> list[np.ndarray]:
    kets = []
    for rho in clean_traj.densities:
        rho_h = 0.5 * (np.asarray(rho) + np.asarray(rho).conj().T)
        vals, vecs = np.linalg.eigh(rho_h)
        kets.append(vecs[:, int(np.argmax(vals.real))])
    return kets


def _mean_logical_fidelity(traj, clean_kets: list[np.ndarray]) -> float:
    vals = [logical_fidelity(rho, psi) for rho, psi in zip(traj.densities, clean_kets)]
    return float(np.mean(vals)) if vals else 0.0


def _candidate_objective_and_report(candidate, tr_bundle: dict) -> tuple[float, dict, dict]:
    obj, terms = recovery_objective(
        candidate,
        tr_bundle["observed"]["trajectory"],
        tr_bundle["reference_anchor"],
        tr_bundle["Hc"],
        tr_bundle["psi0_clock"],
        weights=tr_bundle["weights"],
        ref_bank=tr_bundle["reference_trajs"],
        eta=tr_bundle["eta"],
    )
    report = admissibility_report(
        candidate,
        tr_bundle["reference_trajs"],
        tr_bundle["Hc"],
        tr_bundle["psi0_clock"],
        tr_bundle["thresholds"],
        eta=tr_bundle["eta"],
    )
    return float(obj), terms, report


def _hybrid_decision(
    *,
    recovered_A,
    recovered_B,
    tr_bundle: dict,
    syndrome_consistent: bool,
    objective_tol: float,
    tie_break_requires_syndrome_consistent: bool,
) -> dict:
    obj_A, terms_A, report_A = _candidate_objective_and_report(recovered_A, tr_bundle)
    obj_B, terms_B, report_B = _candidate_objective_and_report(recovered_B, tr_bundle)

    use_b = False
    reason = "keep_syndrome"
    if (not bool(report_A["admissible"])) and bool(report_B["admissible"]):
        use_b = True
        reason = "veto_nonadmissible_A"
    else:
        tie_break_ready = bool(report_B["admissible"]) and (obj_B < obj_A - float(objective_tol))
        if tie_break_requires_syndrome_consistent:
            tie_break_ready = tie_break_ready and syndrome_consistent
        if tie_break_ready:
            use_b = True
            reason = "tie_break_objective"

    selected = recovered_B if use_b else recovered_A
    selected_label = "B" if use_b else "A"
    selected_obj = obj_B if use_b else obj_A
    selected_terms = terms_B if use_b else terms_A
    selected_report = report_B if use_b else report_A
    return {
        "selected_baseline": selected_label,
        "selected_recovered": selected,
        "selected_objective": float(selected_obj),
        "selected_terms": selected_terms,
        "selected_report": selected_report,
        "use_relational": bool(use_b),
        "reason": reason,
        "objective_A": float(obj_A),
        "objective_B": float(obj_B),
        "terms_A": terms_A,
        "terms_B": terms_B,
        "report_A": report_A,
        "report_B": report_B,
        "objective_gain_B_vs_A": float(obj_A - obj_B),
    }


def _run_trajectory_recovery(config: dict) -> dict:
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
    recovered_report = admissibility_report(
        recovery["final_recovered"],
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        thresholds,
        eta=eta,
    )
    observed_report = admissibility_report(
        observed["trajectory"],
        reference_trajs,
        clean["Hc"],
        clean["psi0_clock"],
        thresholds,
        eta=eta,
    )
    return {
        "clean": clean,
        "observed": observed,
        "recovered": recovery["final_recovered"],
        "final_stage": recovery["final_stage"],
        "stage1_best_label": reference_labels[recovery["stage1"]["best_index"]],
        "observed_admissible": bool(observed_report["admissible"]),
        "recovered_admissible": bool(recovered_report["admissible"]),
        "reference_trajs": reference_trajs,
        "thresholds": thresholds,
        "eta": eta,
        "weights": weights,
        "Hc": clean["Hc"],
        "psi0_clock": clean["psi0_clock"],
        "reference_anchor": recovery["stage1"]["reference_anchor"],
        "recovered_report": recovered_report,
    }


def _sample_row(
    *,
    code: str,
    base_seed: int,
    sample_index: int,
    schedule: list[dict[str, Any]],
    state_config_path: str,
    base_config: dict,
    trajectory_inconsistency_threshold: float,
    syndrome_consistent_threshold: float,
    hybrid_objective_tol: float,
    tie_break_requires_syndrome_consistent: bool,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("noise", {})["schedule"] = list(schedule)

    tr_bundle = _run_trajectory_recovery(config)
    clean_traj = tr_bundle["clean"]["trajectory"]
    observed_traj = tr_bundle["observed"]["trajectory"]
    recovered_B = tr_bundle["recovered"]

    recovered_A = apply_syndrome_recovery_to_trajectory(observed_traj, code)
    syndrome_stats = collect_syndrome_statistics(observed_traj, code)
    syndrome = _syndrome_summary(syndrome_stats)

    clean_kets = _logical_ket_per_slice(clean_traj)
    fid_observed = _mean_logical_fidelity(observed_traj, clean_kets)
    fid_recovered_A = _mean_logical_fidelity(recovered_A, clean_kets)
    fid_recovered_B = _mean_logical_fidelity(recovered_B, clean_kets)

    traj_logical_observed = trajectory_logical_fidelity(observed_traj, clean_traj)
    traj_logical_recovered_A = trajectory_logical_fidelity(recovered_A, clean_traj)
    traj_logical_recovered_B = trajectory_logical_fidelity(recovered_B, clean_traj)

    d_clean_observed = float(trajectory_distance(clean_traj, observed_traj))
    d_clean_recovered_A = float(trajectory_distance(clean_traj, recovered_A))
    d_clean_recovered_B = float(trajectory_distance(clean_traj, recovered_B))

    syndrome_consistent = bool(
        syndrome_stats["mean_no_error_probability"] >= float(syndrome_consistent_threshold)
    )
    trajectory_inconsistent = bool(
        d_clean_observed > float(trajectory_inconsistency_threshold)
    )
    hybrid = _hybrid_decision(
        recovered_A=recovered_A,
        recovered_B=recovered_B,
        tr_bundle=tr_bundle,
        syndrome_consistent=syndrome_consistent,
        objective_tol=float(hybrid_objective_tol),
        tie_break_requires_syndrome_consistent=bool(tie_break_requires_syndrome_consistent),
    )
    recovered_C = hybrid["selected_recovered"]
    fid_recovered_C = _mean_logical_fidelity(recovered_C, clean_kets)
    d_clean_recovered_C = float(trajectory_distance(clean_traj, recovered_C))
    noise_family = _schedule_signature(schedule)
    noise_depth = len(schedule)
    total_strength = float(sum(float(step["p"]) for step in schedule))
    state_label = Path(state_config_path).stem
    experiment_id = f"{code}-seed{int(base_seed)}-sample{int(sample_index):03d}"

    return {
        "experiment_id": experiment_id,
        "seed": int(base_seed),
        "sample_index": int(sample_index),
        "code": code,
        "code_type": code,
        "schedule_signature": noise_family,
        "noise_family": noise_family,
        "schedule_length": noise_depth,
        "noise_depth": noise_depth,
        "total_strength": total_strength,
        "noise_strength": total_strength,
        "max_strength": float(max(float(step["p"]) for step in schedule)),
        "schedule": schedule,
        "simulation_backend": str(tr_bundle["observed"]["meta"].get("simulation_backend", "unknown")),
        "backend": str(tr_bundle["observed"]["meta"].get("simulation_backend", "unknown")),
        "state_config": state_config_path,
        "clean_state_label": state_label,
        "candidate_A": f"syndrome_recovery[{code}]",
        "candidate_B": str(tr_bundle["stage1_best_label"]),
        "chosen_C": str(hybrid["selected_baseline"]),
        "reason_C": str(hybrid["reason"]),
        "syndrome": syndrome["dominant"],
        "syndrome_summary": syndrome,
        "fid_observed_mean": fid_observed,
        "fidelity_before": fid_observed,
        "fid_recovered_A_mean": fid_recovered_A,
        "fidelity_after_A": fid_recovered_A,
        "fid_recovered_B_mean": fid_recovered_B,
        "fidelity_after_B": fid_recovered_B,
        "fid_recovered_C_mean": fid_recovered_C,
        "fidelity_after_C": fid_recovered_C,
        "fid_A_gain": float(fid_recovered_A - fid_observed),
        "gain_A": float(fid_recovered_A - fid_observed),
        "fid_B_gain": float(fid_recovered_B - fid_observed),
        "gain_B": float(fid_recovered_B - fid_observed),
        "fid_C_gain": float(fid_recovered_C - fid_observed),
        "gain_C": float(fid_recovered_C - fid_observed),
        "fid_A_minus_B": float(fid_recovered_A - fid_recovered_B),
        "fid_C_minus_A": float(fid_recovered_C - fid_recovered_A),
        "fid_C_minus_B": float(fid_recovered_C - fid_recovered_B),
        "clean_to_observed_distance": d_clean_observed,
        "clean_observed_distance": d_clean_observed,
        "clean_to_recovered_A_distance": d_clean_recovered_A,
        "traj_dist_A": d_clean_recovered_A,
        "clean_to_recovered_B_distance": d_clean_recovered_B,
        "traj_dist_B": d_clean_recovered_B,
        "clean_to_recovered_C_distance": d_clean_recovered_C,
        "A_recovery_nonworsen": bool(d_clean_recovered_A <= d_clean_observed + 1.0e-12),
        "B_recovery_nonworsen": bool(d_clean_recovered_B <= d_clean_observed + 1.0e-12),
        "C_recovery_nonworsen": bool(d_clean_recovered_C <= d_clean_observed + 1.0e-12),
        "syndrome_mean_no_error": float(syndrome_stats["mean_no_error_probability"]),
        "syndrome_min_no_error": float(syndrome_stats["min_no_error_probability"]),
        "syndrome_consistent": syndrome_consistent,
        "trajectory_inconsistent": trajectory_inconsistent,
        "syndrome_consistent_but_trajectory_inconsistent": bool(
            syndrome_consistent and trajectory_inconsistent
        ),
        "traj_logical_observed_mean": float(traj_logical_observed["mean"]),
        "traj_logical_recovered_A_mean": float(traj_logical_recovered_A["mean"]),
        "traj_logical_recovered_B_mean": float(traj_logical_recovered_B["mean"]),
        "traj_logical_recovered_C_mean": float(trajectory_logical_fidelity(recovered_C, clean_traj)["mean"]),
        "trajectory_recovery_final_stage": tr_bundle["final_stage"],
        "trajectory_recovery_stage1_label": tr_bundle["stage1_best_label"],
        "observed_admissible": tr_bundle["observed_admissible"],
        "recovered_B_admissible": tr_bundle["recovered_admissible"],
        "admissible_B": tr_bundle["recovered_admissible"],
        "recovered_A_admissible": bool(hybrid["report_A"]["admissible"]),
        "admissible_A": bool(hybrid["report_A"]["admissible"]),
        "recovered_C_admissible": bool(hybrid["selected_report"]["admissible"]),
        "hybrid_selected_baseline": str(hybrid["selected_baseline"]),
        "hybrid_use_relational": bool(hybrid["use_relational"]),
        "hybrid_reason": str(hybrid["reason"]),
        "hybrid_objective_A": float(hybrid["objective_A"]),
        "objective_A": float(hybrid["objective_A"]),
        "hybrid_objective_B": float(hybrid["objective_B"]),
        "objective_B": float(hybrid["objective_B"]),
        "hybrid_objective_C": float(hybrid["selected_objective"]),
        "objective_C": float(hybrid["selected_objective"]),
        "hybrid_objective_gain_B_vs_A": float(hybrid["objective_gain_B_vs_A"]),
    }


def _mode_label(values: Sequence[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    label, count = counts.most_common(1)[0]
    return label, float(count / len(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    backend_label, _ = _mode_label([str(row["simulation_backend"]) for row in rows])
    codes_evaluated = sorted({str(row["code"]) for row in rows})
    hybrid_label, hybrid_rate = _mode_label([str(row["hybrid_selected_baseline"]) for row in rows])
    summary = {
        "cases": len(rows),
        "codes_evaluated": codes_evaluated,
        "code_count": len(codes_evaluated),
        "simulation_backend": backend_label,
        "mean_schedule_length": _mean([float(row["schedule_length"]) for row in rows]),
        "mean_total_strength": _mean([float(row["total_strength"]) for row in rows]),
        "mixed_schedule_rate": float(
            sum(1.0 if str(row["schedule_signature"]) == "mixed" else 0.0 for row in rows) / len(rows)
        ),
        "fid_observed_mean": _mean([float(row["fid_observed_mean"]) for row in rows]),
        "fid_recovered_A_mean": _mean([float(row["fid_recovered_A_mean"]) for row in rows]),
        "fid_recovered_B_mean": _mean([float(row["fid_recovered_B_mean"]) for row in rows]),
        "fid_recovered_C_mean": _mean([float(row["fid_recovered_C_mean"]) for row in rows]),
        "fid_A_gain_mean": _mean([float(row["fid_A_gain"]) for row in rows]),
        "fid_B_gain_mean": _mean([float(row["fid_B_gain"]) for row in rows]),
        "fid_C_gain_mean": _mean([float(row["fid_C_gain"]) for row in rows]),
        "fid_A_minus_B_mean": _mean([float(row["fid_A_minus_B"]) for row in rows]),
        "fid_C_minus_A_mean": _mean([float(row["fid_C_minus_A"]) for row in rows]),
        "fid_C_minus_B_mean": _mean([float(row["fid_C_minus_B"]) for row in rows]),
        "clean_to_observed_distance_mean": _mean([float(row["clean_to_observed_distance"]) for row in rows]),
        "clean_to_recovered_A_distance_mean": _mean([float(row["clean_to_recovered_A_distance"]) for row in rows]),
        "clean_to_recovered_B_distance_mean": _mean([float(row["clean_to_recovered_B_distance"]) for row in rows]),
        "clean_to_recovered_C_distance_mean": _mean([float(row["clean_to_recovered_C_distance"]) for row in rows]),
        "A_recovery_nonworsen_rate": float(sum(1.0 if bool(row["A_recovery_nonworsen"]) else 0.0 for row in rows) / len(rows)),
        "B_recovery_nonworsen_rate": float(sum(1.0 if bool(row["B_recovery_nonworsen"]) else 0.0 for row in rows) / len(rows)),
        "C_recovery_nonworsen_rate": float(sum(1.0 if bool(row["C_recovery_nonworsen"]) else 0.0 for row in rows) / len(rows)),
        "syndrome_mean_no_error_mean": _mean([float(row["syndrome_mean_no_error"]) for row in rows]),
        "syndrome_consistent_rate": float(sum(1.0 if bool(row["syndrome_consistent"]) else 0.0 for row in rows) / len(rows)),
        "trajectory_inconsistent_rate": float(sum(1.0 if bool(row["trajectory_inconsistent"]) else 0.0 for row in rows) / len(rows)),
        "syndrome_consistent_but_trajectory_inconsistent_rate": float(
            sum(1.0 if bool(row["syndrome_consistent_but_trajectory_inconsistent"]) else 0.0 for row in rows) / len(rows)
        ),
        "hybrid_selected_baseline": hybrid_label,
        "hybrid_selected_rate": hybrid_rate,
        "hybrid_uses_relational_rate": float(sum(1.0 if bool(row["hybrid_use_relational"]) else 0.0 for row in rows) / len(rows)),
        "hybrid_veto_rate": float(sum(1.0 if str(row["hybrid_reason"]) == "veto_nonadmissible_A" else 0.0 for row in rows) / len(rows)),
        "hybrid_tie_break_rate": float(sum(1.0 if str(row["hybrid_reason"]) == "tie_break_objective" else 0.0 for row in rows) / len(rows)),
        "observed_admissible_rate": float(sum(1.0 if bool(row["observed_admissible"]) else 0.0 for row in rows) / len(rows)),
        "recovered_A_admissible_rate": float(sum(1.0 if bool(row["recovered_A_admissible"]) else 0.0 for row in rows) / len(rows)),
        "recovered_B_admissible_rate": float(sum(1.0 if bool(row["recovered_B_admissible"]) else 0.0 for row in rows) / len(rows)),
        "recovered_C_admissible_rate": float(sum(1.0 if bool(row["recovered_C_admissible"]) else 0.0 for row in rows) / len(rows)),
    }
    summary.update(_distribution_stats([float(row["fid_A_gain"]) for row in rows], "fid_A_gain"))
    summary.update(_distribution_stats([float(row["fid_B_gain"]) for row in rows], "fid_B_gain"))
    summary.update(_distribution_stats([float(row["fid_C_gain"]) for row in rows], "fid_C_gain"))
    summary.update(_distribution_stats([float(row["clean_to_observed_distance"]) for row in rows], "clean_to_observed_distance"))
    summary.update(_distribution_stats([float(row["hybrid_objective_gain_B_vs_A"]) for row in rows], "hybrid_objective_gain_B_vs_A"))
    return summary


def _group_rows(rows: list[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    table = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_rows(grouped[group_key]))
        table.append(entry)
    return table


def _build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Encoded [[3,1,1]] QEC Baseline",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "simulation_backend",
            "mean_schedule_length",
            "mean_total_strength",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "fid_A_gain_mean",
            "fid_B_gain_mean",
            "fid_C_gain_mean",
            "A_recovery_nonworsen_rate",
            "B_recovery_nonworsen_rate",
            "C_recovery_nonworsen_rate",
            "hybrid_selected_baseline",
            "hybrid_selected_rate",
            "hybrid_uses_relational_rate",
            "syndrome_consistent_rate",
            "trajectory_inconsistent_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
        "",
        "## By Code",
        "",
        _markdown_table(result["tables"]["by_code"], [
            "code",
            "cases",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "fid_A_minus_B_mean",
            "fid_C_minus_A_mean",
            "fid_C_minus_B_mean",
            "A_recovery_nonworsen_rate",
            "B_recovery_nonworsen_rate",
            "C_recovery_nonworsen_rate",
            "hybrid_uses_relational_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
        "",
        "## By Noise Family",
        "",
        _markdown_table(result["tables"]["by_noise_family"], [
            "noise_family",
            "cases",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "hybrid_uses_relational_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
        "",
        "## By Noise Depth",
        "",
        _markdown_table(result["tables"]["by_noise_depth"], [
            "noise_depth",
            "cases",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "hybrid_uses_relational_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
        "",
        "## By Code and Schedule Signature",
        "",
        _markdown_table(result["tables"]["by_code_and_signature"], [
            "code",
            "schedule_signature",
            "cases",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "syndrome_mean_no_error_mean",
            "hybrid_uses_relational_rate",
            "trajectory_inconsistent_rate",
        ]),
        "",
        "## By Code and Schedule Length",
        "",
        _markdown_table(result["tables"]["by_code_and_length"], [
            "code",
            "schedule_length",
            "cases",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "hybrid_uses_relational_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
    ]
    return "\n".join(lines) + "\n"


def run(
    *,
    codes: Sequence[str],
    state_configs: dict[str, str],
    kinds_by_code: dict[str, Sequence[str]] | None,
    n_samples: int,
    seed: int,
    min_steps: int,
    max_steps: int,
    kinds: Sequence[str],
    p_min: float,
    p_max: float,
    trajectory_inconsistency_threshold: float = 0.05,
    syndrome_consistent_threshold: float = 0.9,
    hybrid_objective_tol: float = 1.0e-9,
    tie_break_requires_syndrome_consistent: bool = True,
    experiment_config: str = "experiment/encoded_qec_baseline.yaml",
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for sample_index in range(int(n_samples)):
        for code in codes:
            code_kinds = (
                list(kinds_by_code.get(code, kinds))
                if kinds_by_code is not None
                else list(kinds)
            )
            schedule = _sample_schedule(
                rng,
                n_system=3,
                kinds=code_kinds,
                min_steps=int(min_steps),
                max_steps=int(max_steps),
                p_min=float(p_min),
                p_max=float(p_max),
            )
            state_path = state_configs[code]
            base_config = load_config(
                experiment_config=experiment_config,
                state_config=state_path,
            )
            row = _sample_row(
                code=code,
                base_seed=int(seed),
                sample_index=sample_index,
                schedule=schedule,
                state_config_path=state_path,
                base_config=base_config,
                trajectory_inconsistency_threshold=trajectory_inconsistency_threshold,
                syndrome_consistent_threshold=syndrome_consistent_threshold,
                hybrid_objective_tol=hybrid_objective_tol,
                tie_break_requires_syndrome_consistent=tie_break_requires_syndrome_consistent,
            )
            row["schedule_kind_space"] = list(code_kinds)
            rows.append(row)

    tables = {
        "by_code": _group_rows(rows, keys=("code",)),
        "by_noise_family": _group_rows(rows, keys=("noise_family",)),
        "by_noise_depth": _group_rows(rows, keys=("noise_depth",)),
        "by_code_and_signature": _group_rows(rows, keys=("code", "schedule_signature")),
        "by_code_and_length": _group_rows(rows, keys=("code", "schedule_length")),
    }
    result = {
        "grid": {
            "codes": [str(c) for c in codes],
            "kinds_by_code": (
                {
                    str(key): [str(item) for item in value]
                    for key, value in kinds_by_code.items()
                }
                if kinds_by_code is not None
                else None
            ),
            "n_samples": int(n_samples),
            "seed": int(seed),
            "min_steps": int(min_steps),
            "max_steps": int(max_steps),
            "kinds": [str(k) for k in kinds],
            "p_min": float(p_min),
            "p_max": float(p_max),
            "trajectory_inconsistency_threshold": float(trajectory_inconsistency_threshold),
            "syndrome_consistent_threshold": float(syndrome_consistent_threshold),
            "hybrid_objective_tol": float(hybrid_objective_tol),
            "tie_break_requires_syndrome_consistent": bool(tie_break_requires_syndrome_consistent),
        },
        "overall": _aggregate_rows(rows),
        "tables": tables,
        "rows": rows,
    }
    result["markdown"] = _build_markdown_report(result)
    return result


def _resolve_state_configs(raw: Any, fallback: dict[str, str]) -> dict[str, str]:
    if not raw:
        return dict(fallback)
    resolved = dict(fallback)
    if isinstance(raw, dict):
        resolved.update({str(key): str(value) for key, value in raw.items()})
    return resolved


def _resolve_kinds_by_code(raw: Any) -> dict[str, list[str]] | None:
    if not raw or not isinstance(raw, dict):
        return None
    resolved: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            resolved[str(key)] = [str(item) for item in value]
    return resolved or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the [[3,1,1]] encoded QEC baseline (syndrome vs trajectory recovery).",
    )
    parser.add_argument("--config", default="experiment/encoded_qec_baseline.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--codes", default=None, help="Comma-separated codes")
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(experiment_config=args.config)
    cfg = dict(base_config.get("encoded_qec", {}))
    default_codes = list(cfg.get("codes", ["bitflip", "phaseflip"]))
    if args.codes:
        codes = [item.strip() for item in args.codes.split(",") if item.strip()]
    else:
        codes = default_codes
    default_state_configs = {
        "bitflip": "states/repetition_bitflip.yaml",
        "phaseflip": "states/repetition_phaseflip.yaml",
    }
    state_configs = _resolve_state_configs(cfg.get("state_configs"), default_state_configs)
    kinds_by_code = _resolve_kinds_by_code(cfg.get("kinds_by_code"))
    hybrid_cfg = dict(cfg.get("hybrid", {}))
    result = run(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        n_samples=int(args.n_samples or cfg.get("n_samples", 12)),
        seed=int(args.seed or cfg.get("seed", 41)),
        min_steps=int(args.min_steps or cfg.get("min_steps", 1)),
        max_steps=int(args.max_steps or cfg.get("max_steps", 3)),
        kinds=list(
            cfg.get(
                "kinds",
                [
                    "bitflip",
                    "phaseflip",
                    "dephasing",
                    "depolarizing",
                    "amplitude_damping",
                    "coherent_x",
                    "coherent_z",
                ],
            )
        ),
        p_min=float(args.p_min or cfg.get("p_min", 0.02)),
        p_max=float(args.p_max or cfg.get("p_max", 0.16)),
        trajectory_inconsistency_threshold=float(cfg.get("trajectory_inconsistency_threshold", 0.05)),
        syndrome_consistent_threshold=float(cfg.get("syndrome_consistent_threshold", 0.9)),
        hybrid_objective_tol=float(hybrid_cfg.get("objective_tol", 1.0e-9)),
        tie_break_requires_syndrome_consistent=bool(
            hybrid_cfg.get("tie_break_requires_syndrome_consistent", True)
        ),
        experiment_config=args.config,
    )
    stem = resolve_output_stem(base_config, "encoded_qec_baseline", args.output_stem)
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_rows.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

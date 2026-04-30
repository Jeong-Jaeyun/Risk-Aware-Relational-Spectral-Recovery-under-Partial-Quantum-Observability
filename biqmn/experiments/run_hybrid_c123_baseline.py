"""Baseline comparison for hybrid C1/C2/C3/C3R recovery policies.

The policies all choose between the same two recovered trajectories:

* A: syndrome recovery
* B: relational trajectory recovery

The difference lies only in the policy used to pick between them.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from ..baselines.syndrome_recovery import (
    apply_syndrome_recovery_to_trajectory,
    collect_syndrome_statistics,
)
from ..core.trajectory import trajectory_distance
from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .syndrome_observation import SyndromeObservationConfig, observe_syndrome_statistics
from .run_encoded_qec_baseline import (
    _candidate_objective_and_report,
    _distribution_stats,
    _hybrid_decision,
    _logical_ket_per_slice,
    _mean_logical_fidelity,
    _run_trajectory_recovery,
    _schedule_signature,
    _syndrome_summary,
    _write_csv,
    _markdown_table,
)
from .run_encoded_regime_map import _schedule_for_family


DEFAULT_CODES = ("bitflip", "phaseflip")


@dataclass(frozen=True)
class PolicyScoreConfig:
    lambda_s: float
    lambda_t: float
    lambda_i: float
    lambda_o: float


@dataclass(frozen=True)
class C3RPolicyConfig:
    score_margin_min: float = 0.0
    admissibility_gap_min: float = 0.0
    b_violation_tolerance: float = 0.0
    uncertainty_max: float = 0.99


def _score_utility(
    *,
    syn_score: float,
    traj_distance_to_observed: float,
    inadmissible: bool,
    objective: float,
    cfg: PolicyScoreConfig,
) -> float:
    # Lower recovery objective is better, so convert it into a bounded utility.
    objective_utility = 1.0 / (1.0 + float(objective))
    return (
        float(cfg.lambda_s) * float(syn_score)
        - float(cfg.lambda_t) * float(traj_distance_to_observed)
        - float(cfg.lambda_i) * (1.0 if bool(inadmissible) else 0.0)
        + float(cfg.lambda_o) * float(objective_utility)
    )


def _mode_label(values: Sequence[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    label = max(counts, key=counts.get)
    return label, float(counts[label] / len(values))


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


def _rate(rows: Sequence[dict[str, Any]], predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return float(sum(1.0 if predicate(row) else 0.0 for row in rows) / len(rows))


def _rate_given(
    rows: Sequence[dict[str, Any]],
    condition_fn,
    value_fn,
) -> float:
    subset = [row for row in rows if condition_fn(row)]
    if not subset:
        return 0.0
    return _rate(subset, value_fn)


def _sum(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values))


def _q(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), float(percentile)))


def _lower_cvar(values: Sequence[float], percentile: float = 5.0) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    threshold = float(np.percentile(arr, float(percentile)))
    tail = arr[arr <= threshold + 1.0e-12]
    if tail.size == 0:
        return None
    return float(np.mean(tail))


def _upper_cvar(values: Sequence[float], percentile: float = 95.0) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    threshold = float(np.percentile(arr, float(percentile)))
    tail = arr[arr >= threshold - 1.0e-12]
    if tail.size == 0:
        return None
    return float(np.mean(tail))


def _candidate_violation(row: dict[str, Any], candidate_label: str) -> float:
    return float(row["c3r_violation_B"] if str(candidate_label) == "B" else row["c3r_violation_A"])


def enrich_c3r_row(row: dict[str, Any], *, fidelity_margin: float = 0.01) -> dict[str, Any]:
    """Populate derived C3R intervention metrics on an existing raw row."""
    item = dict(row)
    c2_B = str(item["candidate_C2"]) == "B"
    block = bool(c2_B and str(item["candidate_C3R"]) == "A")
    harmful = bool(
        c2_B
        and float(item["fid_gain_A"]) > float(item["fid_gain_B"]) + float(fidelity_margin)
    )
    beneficial = bool(
        c2_B
        and float(item["fid_gain_B"]) > float(item["fid_gain_A"]) + float(fidelity_margin)
    )
    prevented_loss = (
        max(0.0, float(item["fid_gain_A"]) - float(item["fid_gain_B"]))
        if block
        else 0.0
    )
    missed_gain = (
        max(0.0, float(item["fid_gain_B"]) - float(item["fid_gain_A"]))
        if block
        else 0.0
    )
    oracle_gain = max(float(item["fid_gain_A"]), float(item["fid_gain_B"]))
    item["c3r_blocks_c2_switch_flag"] = bool(block)
    item["c3r_harmful_c2_switch_flag"] = bool(harmful)
    item["c3r_beneficial_c2_switch_flag"] = bool(beneficial)
    item["c3r_prevented_harmful_switch_flag"] = bool(block and harmful)
    item["c3r_missed_beneficial_switch_flag"] = bool(block and beneficial)
    item["c3r_intervention_gain"] = float(item["fid_gain_C3R"]) - float(item["fid_gain_C2"])
    item["c3r_prevented_loss"] = float(prevented_loss)
    item["c3r_missed_gain"] = float(missed_gain)
    item["c3r_net_intervention_gain"] = float(prevented_loss - missed_gain)
    item["oracle_gain"] = float(oracle_gain)
    for mode in ("C1", "C2", "C3", "C3R"):
        item[f"oracle_regret_{mode}"] = float(oracle_gain - float(item[f"fid_gain_{mode}"]))
        item[f"violation_{mode}"] = _candidate_violation(item, str(item[f"candidate_{mode}"]))
    for mode in ("A", "B", "C1", "C2", "C3", "C3R"):
        observed_key = f"observed_failure_boundary_flag_{mode}"
        if observed_key not in item:
            item[observed_key] = bool(item.get(f"failure_boundary_flag_{mode}", False))
        item[f"true_failure_boundary_flag_{mode}"] = bool(
            item.get("true_syndrome_consistent", False)
            and not bool(item[f"logical_success_{mode}"])
        )
    return item


def _code_name(code_family: str) -> str:
    return {
        "bitflip": "3q_bitflip_repetition",
        "phaseflip": "3q_phaseflip_repetition",
    }.get(str(code_family), str(code_family))


def _normalize_family(family: str) -> str:
    return "mixed" if str(family) == "mixed_pauli" else str(family)


def _candidate_row(
    *,
    candidate_label: str,
    candidate_name: str,
    recovered,
    observed_traj,
    clean_traj,
    clean_kets: list[np.ndarray],
    syndrome_mean_no_error: float,
    objective: float,
    admissible: bool,
) -> dict[str, Any]:
    fidelity_after = _mean_logical_fidelity(recovered, clean_kets)
    return {
        "candidate_label": str(candidate_label),
        "candidate_name": str(candidate_name),
        "admissible": bool(admissible),
        "objective": float(objective),
        "traj_distance_to_observed": float(trajectory_distance(observed_traj, recovered)),
        "traj_distance_to_clean": float(trajectory_distance(clean_traj, recovered)),
        "fidelity_after": float(fidelity_after),
        "fid_gain": None,  # filled later
        "logical_success": None,  # filled later
        "syndrome_score": float(syndrome_mean_no_error if candidate_label == "A" else 0.0),
    }


def _decision_disagreement_ab(
    *,
    candidate_A: dict[str, Any],
    candidate_B: dict[str, Any],
    fidelity_margin: float,
) -> bool:
    return bool(
        bool(candidate_A["admissible"]) != bool(candidate_B["admissible"])
        or abs(float(candidate_B["fidelity_after"]) - float(candidate_A["fidelity_after"])) > float(fidelity_margin)
    )


def _choose_score_policy(
    *,
    candidate_A: dict[str, Any],
    candidate_B: dict[str, Any],
    cfg: PolicyScoreConfig,
    reason_A: str,
    reason_B: str,
) -> dict[str, Any]:
    score_A = _score_utility(
        syn_score=float(candidate_A["syndrome_score"]),
        traj_distance_to_observed=float(candidate_A["traj_distance_to_observed"]),
        inadmissible=not bool(candidate_A["admissible"]),
        objective=float(candidate_A["objective"]),
        cfg=cfg,
    )
    score_B = _score_utility(
        syn_score=float(candidate_B["syndrome_score"]),
        traj_distance_to_observed=float(candidate_B["traj_distance_to_observed"]),
        inadmissible=not bool(candidate_B["admissible"]),
        objective=float(candidate_B["objective"]),
        cfg=cfg,
    )
    if bool(candidate_A["admissible"]) != bool(candidate_B["admissible"]):
        if bool(candidate_A["admissible"]) and score_A >= score_B:
            chosen = candidate_A
        elif bool(candidate_B["admissible"]) and score_B >= score_A:
            chosen = candidate_B
        else:
            chosen = candidate_B if score_B > score_A else candidate_A
        reason = "inadmissibility_penalty_triggered"
    else:
        chosen = candidate_B if score_B > score_A else candidate_A
        reason = reason_B if chosen is candidate_B else reason_A
    return {
        "chosen": chosen,
        "reason": reason,
        "score_A": float(score_A),
        "score_B": float(score_B),
    }


def _choose_safety_policy(
    *,
    candidate_A: dict[str, Any],
    candidate_B: dict[str, Any],
    cfg: PolicyScoreConfig,
) -> dict[str, Any]:
    if bool(candidate_A["admissible"]) != bool(candidate_B["admissible"]):
        chosen = candidate_A if bool(candidate_A["admissible"]) else candidate_B
        return {
            "chosen": chosen,
            "reason": "hard_inadmissibility_block",
            "score_A": None,
            "score_B": None,
        }
    score_A = _score_utility(
        syn_score=float(candidate_A["syndrome_score"]),
        traj_distance_to_observed=float(candidate_A["traj_distance_to_observed"]),
        inadmissible=False,
        objective=float(candidate_A["objective"]),
        cfg=cfg,
    )
    score_B = _score_utility(
        syn_score=float(candidate_B["syndrome_score"]),
        traj_distance_to_observed=float(candidate_B["traj_distance_to_observed"]),
        inadmissible=False,
        objective=float(candidate_B["objective"]),
        cfg=cfg,
    )
    chosen = candidate_B if score_B > score_A else candidate_A
    return {
        "chosen": chosen,
        "reason": "safety_prefers_B" if chosen is candidate_B else "safety_prefers_A",
        "score_A": float(score_A),
        "score_B": float(score_B),
    }


def normalized_admissibility_violation(report: dict[str, Any]) -> float:
    values = []
    penalties = dict(report.get("penalties", {}))
    thresholds = dict(report.get("thresholds", {}))
    for key, penalty in penalties.items():
        threshold = float(thresholds[key])
        denom = max(abs(threshold), 1.0e-12)
        values.append(max(0.0, (float(penalty) - threshold) / denom))
    return float(max(values)) if values else 0.0


def _choose_c3r_policy(
    *,
    candidate_A: dict[str, Any],
    candidate_B: dict[str, Any],
    c2_policy: dict[str, Any],
    score_A: float,
    score_B: float,
    report_A: dict[str, Any],
    report_B: dict[str, Any],
    syndrome_observation_ratio: float,
    syndrome_noise_prob: float,
    syndrome_ambiguity_level: float,
    measurement_error_prob: float,
    reset_error_prob: float,
    syndrome_information_loss: float,
    cfg: C3RPolicyConfig,
) -> dict[str, Any]:
    violation_A = normalized_admissibility_violation(report_A)
    violation_B = normalized_admissibility_violation(report_B)
    score_margin = float(score_B - score_A)
    structural_margin = float(violation_A - violation_B)
    raw_syndrome_uncertainty = float(
        (1.0 - float(syndrome_observation_ratio))
        + float(syndrome_noise_prob)
        + float(syndrome_ambiguity_level)
        + float(measurement_error_prob)
        + float(reset_error_prob)
        + float(syndrome_information_loss)
    )
    syndrome_uncertainty = float(np.clip(raw_syndrome_uncertainty, 0.0, 1.0))

    gate_c2_switch = str(c2_policy["candidate"]) == "B"
    gate_score_margin = score_margin >= float(cfg.score_margin_min)
    gate_leave_A = violation_A >= float(cfg.admissibility_gap_min)
    gate_B_safe = bool(candidate_B["admissible"]) and violation_B <= float(cfg.b_violation_tolerance)
    gate_uncertainty = syndrome_uncertainty <= float(cfg.uncertainty_max)

    allow_B = bool(
        gate_c2_switch
        and gate_score_margin
        and gate_leave_A
        and gate_B_safe
        and gate_uncertainty
    )
    if allow_B:
        chosen = candidate_B
        reason = "c3r_all_gates_pass_switch_to_B"
    else:
        chosen = candidate_A
        if not gate_c2_switch:
            reason = "c3r_c2_preserves_A"
        elif not gate_score_margin:
            reason = "c3r_blocks_low_score_margin"
        elif not gate_leave_A:
            reason = "c3r_blocks_insufficient_A_risk"
        elif not gate_B_safe:
            reason = "c3r_blocks_unsafe_B"
        elif not gate_uncertainty:
            reason = "c3r_blocks_high_syndrome_uncertainty"
        else:
            reason = "c3r_blocks_unknown"

    return {
        "candidate": str(chosen["candidate_label"]),
        "candidate_label": str(chosen["candidate_label"]),
        "admissible": bool(chosen["admissible"]),
        "objective": float(chosen["objective"]),
        "traj_distance": float(chosen["traj_distance_to_observed"]),
        "traj_distance_to_clean": float(chosen["traj_distance_to_clean"]),
        "fidelity_after": float(chosen["fidelity_after"]),
        "fid_gain": float(chosen["fid_gain"]),
        "logical_success": bool(chosen["logical_success"]),
        "reason": str(reason),
        "decision_reason": str(reason),
        "score_A": float(score_A),
        "score_B": float(score_B),
        "score_margin": float(score_margin),
        "structural_margin": float(structural_margin),
        "violation_A": float(violation_A),
        "violation_B": float(violation_B),
        "syndrome_uncertainty": float(syndrome_uncertainty),
        "raw_syndrome_uncertainty": float(raw_syndrome_uncertainty),
        "gate_c2_switch": bool(gate_c2_switch),
        "gate_score_margin": bool(gate_score_margin),
        "gate_leave_A": bool(gate_leave_A),
        "gate_B_safe": bool(gate_B_safe),
        "gate_uncertainty": bool(gate_uncertainty),
        "allow_B": bool(allow_B),
    }


def _policy_projection(
    *,
    chosen: dict[str, Any],
    reason: str,
    score_A: float | None = None,
    score_B: float | None = None,
) -> dict[str, Any]:
    return {
        "candidate": str(chosen["candidate_label"]),
        "admissible": bool(chosen["admissible"]),
        "objective": float(chosen["objective"]),
        "traj_distance": float(chosen["traj_distance_to_observed"]),
        "traj_distance_to_clean": float(chosen["traj_distance_to_clean"]),
        "fidelity_after": float(chosen["fidelity_after"]),
        "fid_gain": float(chosen["fid_gain"]),
        "logical_success": bool(chosen["logical_success"]),
        "reason": str(reason),
        "score_A": None if score_A is None else float(score_A),
        "score_B": None if score_B is None else float(score_B),
    }


def _sample_row(
    *,
    run_id: str,
    timestamp: str,
    config_name: str,
    code_family: str,
    state_config_path: str,
    base_config: dict[str, Any],
    seed: int,
    sample_index: int,
    noise_family: str,
    noise_strength: float,
    noise_depth: int,
    logical_success_threshold: float,
    fidelity_margin: float,
    c2_cfg: PolicyScoreConfig,
    c3_cfg: PolicyScoreConfig,
    c3r_cfg: C3RPolicyConfig | None = None,
    syndrome_obs_cfg: SyndromeObservationConfig | None = None,
    c1_objective_tol: float,
    c1_tie_break_requires_syndrome_consistent: bool,
) -> dict[str, Any]:
    syndrome_obs_cfg = syndrome_obs_cfg or SyndromeObservationConfig()
    c3r_cfg = c3r_cfg or C3RPolicyConfig()
    rng = np.random.default_rng(int(seed))
    n_system = int(base_config.get("system", {}).get("n_qubits", 3))
    kinds_by_code = {
        str(key): [str(item) for item in value]
        for key, value in dict(base_config.get("hybrid_c123", {}).get("kinds_by_code", {})).items()
        if isinstance(value, (list, tuple))
    }
    schedule = _schedule_for_family(
        rng,
        code=str(code_family),
        family=_normalize_family(str(noise_family)),
        strength=float(noise_strength),
        depth=int(noise_depth),
        n_system=n_system,
        kinds_by_code=kinds_by_code,
    )
    config = deepcopy(base_config)
    config.setdefault("noise", {})["schedule"] = list(schedule)

    tr_bundle = _run_trajectory_recovery(config)
    clean_traj = tr_bundle["clean"]["trajectory"]
    observed_traj = tr_bundle["observed"]["trajectory"]
    recovered_B = tr_bundle["recovered"]
    # NOTE:
    # recovered_A is computed from the quantum trajectory itself, not directly
    # from the corrupted / partial syndrome observation. This experiment
    # therefore probes policy behavior under degraded syndrome information,
    # rather than the physical fidelity of a decoder that must recover using
    # corrupted syndrome bits.
    recovered_A = apply_syndrome_recovery_to_trajectory(observed_traj, str(code_family))

    syndrome_stats = collect_syndrome_statistics(observed_traj, str(code_family))
    true_syndrome = _syndrome_summary(syndrome_stats)
    observed_syndrome = observe_syndrome_statistics(
        syndrome_stats,
        cfg=syndrome_obs_cfg,
        rng=rng,
    )
    syndrome_consistent = bool(
        float(observed_syndrome["mean_no_error_probability"])
        >= float(syndrome_obs_cfg.consistency_threshold)
    )
    true_syndrome_consistent = bool(
        float(syndrome_stats["mean_no_error_probability"])
        >= float(syndrome_obs_cfg.consistency_threshold)
    )

    clean_kets = _logical_ket_per_slice(clean_traj)
    fidelity_before = _mean_logical_fidelity(observed_traj, clean_kets)
    objective_A, _, report_A = _candidate_objective_and_report(recovered_A, tr_bundle)
    objective_B, _, report_B = _candidate_objective_and_report(recovered_B, tr_bundle)

    candidate_A = _candidate_row(
        candidate_label="A",
        candidate_name=f"syndrome_recovery[{code_family}]",
        recovered=recovered_A,
        observed_traj=observed_traj,
        clean_traj=clean_traj,
        clean_kets=clean_kets,
        syndrome_mean_no_error=float(observed_syndrome["mean_no_error_probability"]),
        objective=float(objective_A),
        admissible=bool(report_A["admissible"]),
    )
    candidate_B = _candidate_row(
        candidate_label="B",
        candidate_name=str(tr_bundle["stage1_best_label"]),
        recovered=recovered_B,
        observed_traj=observed_traj,
        clean_traj=clean_traj,
        clean_kets=clean_kets,
        syndrome_mean_no_error=float(observed_syndrome["mean_no_error_probability"]),
        objective=float(objective_B),
        admissible=bool(report_B["admissible"]),
    )
    for candidate in (candidate_A, candidate_B):
        candidate["fid_gain"] = float(candidate["fidelity_after"] - float(fidelity_before))
        candidate["logical_success"] = bool(candidate["fidelity_after"] >= float(logical_success_threshold))

    c1 = _hybrid_decision(
        recovered_A=recovered_A,
        recovered_B=recovered_B,
        tr_bundle=tr_bundle,
        syndrome_consistent=syndrome_consistent,
        objective_tol=float(c1_objective_tol),
        tie_break_requires_syndrome_consistent=bool(c1_tie_break_requires_syndrome_consistent),
    )
    chosen_c1 = candidate_B if str(c1["selected_baseline"]) == "B" else candidate_A
    c1_policy = _policy_projection(chosen=chosen_c1, reason=str(c1["reason"]))

    c2 = _choose_score_policy(
        candidate_A=candidate_A,
        candidate_B=candidate_B,
        cfg=c2_cfg,
        reason_A="score_prefers_A",
        reason_B="score_prefers_B",
    )
    c2_policy = _policy_projection(
        chosen=c2["chosen"],
        reason=str(c2["reason"]),
        score_A=c2["score_A"],
        score_B=c2["score_B"],
    )

    c3 = _choose_safety_policy(
        candidate_A=candidate_A,
        candidate_B=candidate_B,
        cfg=c3_cfg,
    )
    c3_policy = _policy_projection(
        chosen=c3["chosen"],
        reason=str(c3["reason"]),
        score_A=c3["score_A"],
        score_B=c3["score_B"],
    )

    c3r_policy = _choose_c3r_policy(
        candidate_A=candidate_A,
        candidate_B=candidate_B,
        c2_policy=c2_policy,
        score_A=float(c2["score_A"]),
        score_B=float(c2["score_B"]),
        report_A=report_A,
        report_B=report_B,
        syndrome_observation_ratio=float(observed_syndrome["observation_ratio"]),
        syndrome_noise_prob=float(observed_syndrome["noise_prob"]),
        syndrome_ambiguity_level=float(observed_syndrome["ambiguity_level"]),
        measurement_error_prob=float(observed_syndrome["measurement_error_prob"]),
        reset_error_prob=float(observed_syndrome["reset_error_prob"]),
        syndrome_information_loss=float(observed_syndrome["syndrome_information_loss"]),
        cfg=c3r_cfg,
    )
    violation_A = float(c3r_policy["violation_A"])
    violation_B = float(c3r_policy["violation_B"])

    def false_safe(policy: dict[str, Any]) -> bool:
        return bool(syndrome_consistent and not bool(policy["admissible"]))

    def false_safe_fidelity(policy: dict[str, Any]) -> bool:
        return bool(float(policy["fid_gain"]) < -float(fidelity_margin))

    def logical_success(policy: dict[str, Any]) -> bool:
        return bool(policy["logical_success"])

    c3r_blocks_c2_switch = bool(c2_policy["candidate"] == "B" and c3r_policy["candidate"] == "A")
    c3r_harmful_c2_switch = bool(
        c2_policy["candidate"] == "B"
        and float(candidate_A["fid_gain"]) > float(candidate_B["fid_gain"]) + float(fidelity_margin)
    )
    c3r_beneficial_c2_switch = bool(
        c2_policy["candidate"] == "B"
        and float(candidate_B["fid_gain"]) > float(candidate_A["fid_gain"]) + float(fidelity_margin)
    )
    c3r_prevented_harmful_switch = bool(
        c3r_blocks_c2_switch
        and c3r_harmful_c2_switch
    )
    c3r_missed_beneficial_switch = bool(
        c3r_blocks_c2_switch
        and c3r_beneficial_c2_switch
    )
    c3r_prevented_loss = (
        max(0.0, float(candidate_A["fid_gain"]) - float(candidate_B["fid_gain"]))
        if c3r_blocks_c2_switch
        else 0.0
    )
    c3r_missed_gain = (
        max(0.0, float(candidate_B["fid_gain"]) - float(candidate_A["fid_gain"]))
        if c3r_blocks_c2_switch
        else 0.0
    )
    c3r_intervention_gain = float(c3r_policy["fid_gain"]) - float(c2_policy["fid_gain"])
    c3r_net_intervention_gain = float(c3r_prevented_loss - c3r_missed_gain)
    oracle_gain = max(float(candidate_A["fid_gain"]), float(candidate_B["fid_gain"]))

    experiment_id = (
        f"{code_family}-{noise_family}-p{float(noise_strength):.2f}-d{int(noise_depth)}-seed{int(seed):03d}-"
        f"sample{int(sample_index):03d}"
    )

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": str(tr_bundle["observed"]["meta"].get("simulation_backend", "unknown")),
        "config_name": str(config_name),
        "code_family": str(code_family),
        "code_name": _code_name(str(code_family)),
        "noise_family": str(noise_family),
        "noise_strength": float(noise_strength),
        "noise_depth": int(noise_depth),
        "seed": int(seed),
        "syndrome_label": str(observed_syndrome["dominant"]),
        "true_syndrome_label": str(true_syndrome["dominant"]),
        "observed_syndrome_label": str(observed_syndrome["dominant"]),
        "syndrome_mean_no_error": float(observed_syndrome["mean_no_error_probability"]),
        "true_syndrome_mean_no_error": float(syndrome_stats["mean_no_error_probability"]),
        "observed_syndrome_mean_no_error": float(observed_syndrome["mean_no_error_probability"]),
        "syndrome_min_no_error": float(observed_syndrome["min_no_error_probability"]),
        "true_syndrome_min_no_error": float(syndrome_stats["min_no_error_probability"]),
        "observed_syndrome_min_no_error": float(observed_syndrome["min_no_error_probability"]),
        "syndrome_observation_ratio": float(observed_syndrome["observation_ratio"]),
        "syndrome_noise_prob": float(observed_syndrome["noise_prob"]),
        "syndrome_ambiguity_level": float(observed_syndrome["ambiguity_level"]),
        "measurement_error_prob": float(observed_syndrome["measurement_error_prob"]),
        "reset_error_prob": float(observed_syndrome["reset_error_prob"]),
        "syndrome_corruption_rate": float(observed_syndrome["syndrome_corruption_rate"]),
        "syndrome_information_loss": float(observed_syndrome["syndrome_information_loss"]),
        "is_syndrome_partial": bool(observed_syndrome["is_syndrome_partial"]),
        "is_syndrome_corrupted": bool(observed_syndrome["is_syndrome_corrupted"]),
        "is_syndrome_ambiguous": bool(observed_syndrome["is_syndrome_ambiguous"]),
        "true_syndrome_consistent": bool(true_syndrome_consistent),
        "observed_syndrome_consistent": bool(syndrome_consistent),
        "candidate_A": str(candidate_A["candidate_name"]),
        "candidate_B": str(candidate_B["candidate_name"]),
        "candidate_C1": str(c1_policy["candidate"]),
        "candidate_C2": str(c2_policy["candidate"]),
        "candidate_C3": str(c3_policy["candidate"]),
        "candidate_C3R": str(c3r_policy["candidate"]),
        "admissible_A": bool(candidate_A["admissible"]),
        "admissible_B": bool(candidate_B["admissible"]),
        "admissible_C1": bool(c1_policy["admissible"]),
        "admissible_C2": bool(c2_policy["admissible"]),
        "admissible_C3": bool(c3_policy["admissible"]),
        "admissible_C3R": bool(c3r_policy["admissible"]),
        "objective_A": float(candidate_A["objective"]),
        "objective_B": float(candidate_B["objective"]),
        "objective_C1": float(c1_policy["objective"]),
        "objective_C2": float(c2_policy["objective"]),
        "objective_C3": float(c3_policy["objective"]),
        "objective_C3R": float(c3r_policy["objective"]),
        "traj_distance_A": float(candidate_A["traj_distance_to_observed"]),
        "traj_distance_B": float(candidate_B["traj_distance_to_observed"]),
        "traj_distance_C1": float(c1_policy["traj_distance"]),
        "traj_distance_C2": float(c2_policy["traj_distance"]),
        "traj_distance_C3": float(c3_policy["traj_distance"]),
        "traj_distance_C3R": float(c3r_policy["traj_distance"]),
        "traj_distance_to_clean_A": float(candidate_A["traj_distance_to_clean"]),
        "traj_distance_to_clean_B": float(candidate_B["traj_distance_to_clean"]),
        "traj_distance_to_clean_C1": float(c1_policy["traj_distance_to_clean"]),
        "traj_distance_to_clean_C2": float(c2_policy["traj_distance_to_clean"]),
        "traj_distance_to_clean_C3": float(c3_policy["traj_distance_to_clean"]),
        "traj_distance_to_clean_C3R": float(c3r_policy["traj_distance_to_clean"]),
        "fidelity_before": float(fidelity_before),
        "fidelity_after_A": float(candidate_A["fidelity_after"]),
        "fidelity_after_B": float(candidate_B["fidelity_after"]),
        "fidelity_after_C1": float(c1_policy["fidelity_after"]),
        "fidelity_after_C2": float(c2_policy["fidelity_after"]),
        "fidelity_after_C3": float(c3_policy["fidelity_after"]),
        "fidelity_after_C3R": float(c3r_policy["fidelity_after"]),
        "fid_gain_A": float(candidate_A["fid_gain"]),
        "fid_gain_B": float(candidate_B["fid_gain"]),
        "fid_gain_C1": float(c1_policy["fid_gain"]),
        "fid_gain_C2": float(c2_policy["fid_gain"]),
        "fid_gain_C3": float(c3_policy["fid_gain"]),
        "fid_gain_C3R": float(c3r_policy["fid_gain"]),
        "logical_success_A": bool(candidate_A["logical_success"]),
        "logical_success_B": bool(candidate_B["logical_success"]),
        "logical_success_C1": bool(c1_policy["logical_success"]),
        "logical_success_C2": bool(c2_policy["logical_success"]),
        "logical_success_C3": bool(c3_policy["logical_success"]),
        "logical_success_C3R": bool(c3r_policy["logical_success"]),
        "decision_reason_C1": str(c1_policy["reason"]),
        "decision_reason_C2": str(c2_policy["reason"]),
        "decision_reason_C3": str(c3_policy["reason"]),
        "decision_reason_C3R": str(c3r_policy["reason"]),
        "false_safe_flag_A": false_safe(candidate_A),
        "false_safe_flag_B": false_safe(candidate_B),
        "false_safe_flag_C1": false_safe(c1_policy),
        "false_safe_flag_C2": false_safe(c2_policy),
        "false_safe_flag_C3": false_safe(c3_policy),
        "false_safe_flag_C3R": false_safe(c3r_policy),
        "false_safe_fidelity_flag_A": false_safe_fidelity(candidate_A),
        "false_safe_fidelity_flag_B": false_safe_fidelity(candidate_B),
        "false_safe_fidelity_flag_C1": false_safe_fidelity(c1_policy),
        "false_safe_fidelity_flag_C2": false_safe_fidelity(c2_policy),
        "false_safe_fidelity_flag_C3": false_safe_fidelity(c3_policy),
        "false_safe_fidelity_flag_C3R": false_safe_fidelity(c3r_policy),
        "nonworsen_A": bool(candidate_A["fid_gain"] >= -1.0e-12),
        "nonworsen_B": bool(candidate_B["fid_gain"] >= -1.0e-12),
        "nonworsen_C1": bool(c1_policy["fid_gain"] >= -1.0e-12),
        "nonworsen_C2": bool(c2_policy["fid_gain"] >= -1.0e-12),
        "nonworsen_C3": bool(c3_policy["fid_gain"] >= -1.0e-12),
        "nonworsen_C3R": bool(c3r_policy["fid_gain"] >= -1.0e-12),
        "failure_boundary_flag_A": bool(syndrome_consistent and not logical_success(candidate_A)),
        "failure_boundary_flag_B": bool(syndrome_consistent and not logical_success(candidate_B)),
        "failure_boundary_flag_C1": bool(syndrome_consistent and not logical_success(c1_policy)),
        "failure_boundary_flag_C2": bool(syndrome_consistent and not logical_success(c2_policy)),
        "failure_boundary_flag_C3": bool(syndrome_consistent and not logical_success(c3_policy)),
        "failure_boundary_flag_C3R": bool(syndrome_consistent and not logical_success(c3r_policy)),
        "observed_failure_boundary_flag_A": bool(syndrome_consistent and not logical_success(candidate_A)),
        "observed_failure_boundary_flag_B": bool(syndrome_consistent and not logical_success(candidate_B)),
        "observed_failure_boundary_flag_C1": bool(syndrome_consistent and not logical_success(c1_policy)),
        "observed_failure_boundary_flag_C2": bool(syndrome_consistent and not logical_success(c2_policy)),
        "observed_failure_boundary_flag_C3": bool(syndrome_consistent and not logical_success(c3_policy)),
        "observed_failure_boundary_flag_C3R": bool(syndrome_consistent and not logical_success(c3r_policy)),
        "true_failure_boundary_flag_A": bool(true_syndrome_consistent and not logical_success(candidate_A)),
        "true_failure_boundary_flag_B": bool(true_syndrome_consistent and not logical_success(candidate_B)),
        "true_failure_boundary_flag_C1": bool(true_syndrome_consistent and not logical_success(c1_policy)),
        "true_failure_boundary_flag_C2": bool(true_syndrome_consistent and not logical_success(c2_policy)),
        "true_failure_boundary_flag_C3": bool(true_syndrome_consistent and not logical_success(c3_policy)),
        "true_failure_boundary_flag_C3R": bool(true_syndrome_consistent and not logical_success(c3r_policy)),
        "decision_disagreement_rate_AB_flag": _decision_disagreement_ab(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            fidelity_margin=float(fidelity_margin),
        ),
        "decision_disagreement_rate_C1A_flag": bool(c1_policy["candidate"] != "A"),
        "decision_disagreement_rate_C2A_flag": bool(c2_policy["candidate"] != "A"),
        "decision_disagreement_rate_C3A_flag": bool(c3_policy["candidate"] != "A"),
        "decision_disagreement_rate_C3RA_flag": bool(c3r_policy["candidate"] != "A"),
        "decision_disagreement_C3R_vs_C2_flag": bool(
            c3r_policy["candidate"] != c2_policy["candidate"]
        ),
        "risky_case_flag": bool(
            false_safe(candidate_A) or float(candidate_B["fidelity_after"]) > float(candidate_A["fidelity_after"]) + float(fidelity_margin)
        ),
        "safe_case_flag": bool(
            not (
                false_safe(candidate_A)
                or float(candidate_B["fidelity_after"]) > float(candidate_A["fidelity_after"]) + float(fidelity_margin)
            )
        ),
        "score_C2_A": c2_policy["score_A"],
        "score_C2_B": c2_policy["score_B"],
        "score_C3_A": c3_policy["score_A"],
        "score_C3_B": c3_policy["score_B"],
        "score_C3R_A": c3r_policy["score_A"],
        "score_C3R_B": c3r_policy["score_B"],
        "c3r_score_margin": float(c3r_policy["score_margin"]),
        "c3r_structural_margin": float(c3r_policy["structural_margin"]),
        "c3r_violation_A": float(c3r_policy["violation_A"]),
        "c3r_violation_B": float(c3r_policy["violation_B"]),
        "c3r_syndrome_uncertainty": float(c3r_policy["syndrome_uncertainty"]),
        "c3r_raw_syndrome_uncertainty": float(c3r_policy["raw_syndrome_uncertainty"]),
        "violation_C1": float(violation_B if str(c1_policy["candidate"]) == "B" else violation_A),
        "violation_C2": float(violation_B if str(c2_policy["candidate"]) == "B" else violation_A),
        "violation_C3": float(violation_B if str(c3_policy["candidate"]) == "B" else violation_A),
        "violation_C3R": float(violation_B if str(c3r_policy["candidate"]) == "B" else violation_A),
        "c3r_gate_c2_switch": bool(c3r_policy["gate_c2_switch"]),
        "c3r_gate_score_margin": bool(c3r_policy["gate_score_margin"]),
        "c3r_gate_leave_A": bool(c3r_policy["gate_leave_A"]),
        "c3r_gate_B_safe": bool(c3r_policy["gate_B_safe"]),
        "c3r_gate_uncertainty": bool(c3r_policy["gate_uncertainty"]),
        "c3r_allow_B": bool(c3r_policy["allow_B"]),
        "c3r_blocks_c2_switch_flag": bool(c3r_blocks_c2_switch),
        "c3r_harmful_c2_switch_flag": bool(c3r_harmful_c2_switch),
        "c3r_beneficial_c2_switch_flag": bool(c3r_beneficial_c2_switch),
        "c3r_prevented_harmful_switch_flag": bool(c3r_prevented_harmful_switch),
        "c3r_missed_beneficial_switch_flag": bool(c3r_missed_beneficial_switch),
        "c3r_intervention_gain": float(c3r_intervention_gain),
        "c3r_prevented_loss": float(c3r_prevented_loss),
        "c3r_missed_gain": float(c3r_missed_gain),
        "c3r_net_intervention_gain": float(c3r_net_intervention_gain),
        "oracle_gain": float(oracle_gain),
        "oracle_regret_C1": float(oracle_gain - float(c1_policy["fid_gain"])),
        "oracle_regret_C2": float(oracle_gain - float(c2_policy["fid_gain"])),
        "oracle_regret_C3": float(oracle_gain - float(c3_policy["fid_gain"])),
        "oracle_regret_C3R": float(oracle_gain - float(c3r_policy["fid_gain"])),
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"cases": 0}

    def observed_failure_boundary(row: dict[str, Any], mode: str) -> bool:
        key = f"observed_failure_boundary_flag_{mode}"
        if key in row:
            return bool(row[key])
        return bool(row.get(f"failure_boundary_flag_{mode}", False))

    def true_failure_boundary(row: dict[str, Any], mode: str) -> bool:
        key = f"true_failure_boundary_flag_{mode}"
        if key in row:
            return bool(row[key])
        return bool(row.get("true_syndrome_consistent", False) and not bool(row[f"logical_success_{mode}"]))

    def c3r_block(row: dict[str, Any]) -> bool:
        return bool(row.get("c3r_blocks_c2_switch_flag", False))

    def c2_switch_to_B(row: dict[str, Any]) -> bool:
        return str(row["candidate_C2"]) == "B"

    def harmful_c2_switch(row: dict[str, Any]) -> bool:
        if "c3r_harmful_c2_switch_flag" in row:
            return bool(row["c3r_harmful_c2_switch_flag"])
        return bool(
            c2_switch_to_B(row)
            and float(row["fid_gain_A"]) > float(row["fid_gain_B"]) + float(row.get("fidelity_margin", 0.01))
        )

    def beneficial_c2_switch(row: dict[str, Any]) -> bool:
        if "c3r_beneficial_c2_switch_flag" in row:
            return bool(row["c3r_beneficial_c2_switch_flag"])
        return bool(
            c2_switch_to_B(row)
            and float(row["fid_gain_B"]) > float(row["fid_gain_A"]) + float(row.get("fidelity_margin", 0.01))
        )

    def prevented_loss(row: dict[str, Any]) -> float:
        if "c3r_prevented_loss" in row:
            return float(row["c3r_prevented_loss"])
        return max(0.0, float(row["fid_gain_A"]) - float(row["fid_gain_B"])) if c3r_block(row) else 0.0

    def missed_gain(row: dict[str, Any]) -> float:
        if "c3r_missed_gain" in row:
            return float(row["c3r_missed_gain"])
        return max(0.0, float(row["fid_gain_B"]) - float(row["fid_gain_A"])) if c3r_block(row) else 0.0

    def chosen_violation(row: dict[str, Any], mode: str) -> float:
        key = f"violation_{mode}"
        if key in row:
            return float(row[key])
        return _candidate_violation(row, str(row[f"candidate_{mode}"]))

    summary = {
        "cases": len(rows),
        "backend": str(rows[0]["backend"]),
        "code_count": len(sorted({str(row["code_family"]) for row in rows})),
        "noise_family_count": len(sorted({str(row["noise_family"]) for row in rows})),
    }
    summary["true_syndrome_mean_no_error_mean"] = _mean(
        [float(row["true_syndrome_mean_no_error"]) for row in rows]
    )
    summary["observed_syndrome_mean_no_error_mean"] = _mean(
        [float(row["observed_syndrome_mean_no_error"]) for row in rows]
    )
    summary["syndrome_corruption_rate"] = _mean(
        [float(row["syndrome_corruption_rate"]) for row in rows]
    )
    summary["syndrome_information_loss_mean"] = _mean(
        [float(row["syndrome_information_loss"]) for row in rows]
    )
    summary["observed_syndrome_consistent_rate"] = _rate(
        rows, lambda row: bool(row["observed_syndrome_consistent"])
    )
    summary["true_syndrome_consistent_rate"] = _rate(
        rows, lambda row: bool(row["true_syndrome_consistent"])
    )
    summary["partial_syndrome_case_rate"] = _rate(
        rows, lambda row: bool(row["is_syndrome_partial"])
    )
    summary["ambiguous_syndrome_case_rate"] = _rate(
        rows, lambda row: bool(row["is_syndrome_ambiguous"])
    )
    for mode in ("A", "B", "C1", "C2", "C3", "C3R"):
        fid_gains = [float(row[f"fid_gain_{mode}"]) for row in rows]
        summary[f"fid_gain_{mode}_mean"] = _mean(fid_gains)
        summary.update(_distribution_stats(fid_gains, f"fid_gain_{mode}"))
        summary[f"fid_gain_q05_{mode}"] = _q(fid_gains, 5.0)
        summary[f"fid_gain_cvar05_{mode}"] = _lower_cvar(fid_gains, 5.0)
        summary[f"nonworsen_rate_{mode}"] = _rate(rows, lambda row, m=mode: bool(row[f"nonworsen_{m}"]))
        summary[f"admissible_rate_{mode}"] = _rate(rows, lambda row, m=mode: bool(row[f"admissible_{m}"]))
        summary[f"logical_success_rate_{mode}"] = _rate(rows, lambda row, m=mode: bool(row[f"logical_success_{m}"]))
        summary[f"false_safe_rate_{mode}"] = _rate(rows, lambda row, m=mode: bool(row[f"false_safe_flag_{m}"]))
        summary[f"false_safe_fidelity_rate_{mode}"] = _rate(
            rows, lambda row, m=mode: bool(row[f"false_safe_fidelity_flag_{m}"])
        )
        summary[f"failure_boundary_rate_{mode}"] = _rate(rows, lambda row, m=mode: bool(row[f"failure_boundary_flag_{m}"]))
        summary[f"observed_failure_boundary_rate_{mode}"] = _rate(
            rows, lambda row, m=mode: observed_failure_boundary(row, m)
        )
        summary[f"true_failure_boundary_rate_{mode}"] = _rate(
            rows, lambda row, m=mode: true_failure_boundary(row, m)
        )
        if mode in ("C1", "C2", "C3", "C3R"):
            summary[f"chosen_candidate_violation_mean_{mode}"] = _mean(
                [chosen_violation(row, mode) for row in rows]
            )
            regrets = [
                float(row.get(f"oracle_regret_{mode}", max(float(row["fid_gain_A"]), float(row["fid_gain_B"])) - float(row[f"fid_gain_{mode}"])))
                for row in rows
            ]
            summary[f"oracle_regret_mean_{mode}"] = _mean(regrets)
            summary[f"oracle_regret_sum_{mode}"] = _sum(regrets)
            summary[f"oracle_regret_q95_{mode}"] = _q(regrets, 95.0)
            summary[f"oracle_regret_cvar95_{mode}"] = _upper_cvar(regrets, 95.0)
    summary["decision_disagreement_rate_AB"] = _rate(rows, lambda row: bool(row["decision_disagreement_rate_AB_flag"]))
    summary["decision_disagreement_rate_C1A"] = _rate(rows, lambda row: bool(row["decision_disagreement_rate_C1A_flag"]))
    summary["decision_disagreement_rate_C2A"] = _rate(rows, lambda row: bool(row["decision_disagreement_rate_C2A_flag"]))
    summary["decision_disagreement_rate_C3A"] = _rate(rows, lambda row: bool(row["decision_disagreement_rate_C3A_flag"]))
    summary["decision_disagreement_rate_C3RA"] = _rate(rows, lambda row: bool(row["decision_disagreement_rate_C3RA_flag"]))
    summary["decision_disagreement_rate_C3R_vs_C2"] = _rate(
        rows, lambda row: bool(row["decision_disagreement_C3R_vs_C2_flag"])
    )
    for mode in ("C1", "C2", "C3", "C3R"):
        summary[f"chosen_B_rate_{mode}"] = _rate(rows, lambda row, m=mode: str(row[f"candidate_{m}"]) == "B")
        summary[f"safe_case_retention_rate_{mode}"] = _rate(
            [row for row in rows if bool(row["safe_case_flag"])],
            lambda row, m=mode: str(row[f"candidate_{m}"]) == "A",
        )
        summary[f"risky_case_capture_rate_{mode}"] = _rate(
            [row for row in rows if bool(row["risky_case_flag"])],
            lambda row, m=mode: str(row[f"candidate_{m}"]) == "B",
        )
    summary["veto_rate_C1"] = _rate(rows, lambda row: str(row["decision_reason_C1"]) == "veto_nonadmissible_A")
    summary["tie_break_rate_C1"] = _rate(rows, lambda row: str(row["decision_reason_C1"]) == "tie_break_objective")
    summary["weighted_score_switch_rate_C2"] = _rate(rows, lambda row: str(row["candidate_C2"]) == "B")
    summary["weighted_score_switch_rate_C3"] = _rate(rows, lambda row: str(row["candidate_C3"]) == "B")
    summary["weighted_score_switch_rate_C3R"] = _rate(rows, lambda row: str(row["candidate_C3R"]) == "B")
    summary["veto_rate_C2"] = _rate(rows, lambda row: str(row["decision_reason_C2"]) == "inadmissibility_penalty_triggered")
    summary["veto_rate_C3"] = _rate(rows, lambda row: str(row["decision_reason_C3"]) == "hard_inadmissibility_block")
    summary["veto_rate_C3R"] = _rate(rows, lambda row: str(row["candidate_C2"]) == "B" and str(row["candidate_C3R"]) == "A")
    summary["tie_break_rate_C2"] = 0.0
    summary["tie_break_rate_C3"] = 0.0
    summary["tie_break_rate_C3R"] = 0.0
    summary["c3r_gate_c2_switch_rate"] = _rate(rows, lambda row: bool(row["c3r_gate_c2_switch"]))
    summary["c3r_gate_score_margin_rate"] = _rate(rows, lambda row: bool(row["c3r_gate_score_margin"]))
    summary["c3r_gate_leave_A_rate"] = _rate(rows, lambda row: bool(row["c3r_gate_leave_A"]))
    summary["c3r_gate_B_safe_rate"] = _rate(rows, lambda row: bool(row["c3r_gate_B_safe"]))
    summary["c3r_gate_uncertainty_rate"] = _rate(rows, lambda row: bool(row["c3r_gate_uncertainty"]))
    summary["c3r_allow_B_rate"] = _rate(rows, lambda row: bool(row["c3r_allow_B"]))
    c2_selected_B = lambda row: str(row["candidate_C2"]) == "B"
    summary["c3r_gate_c2_switch_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_gate_c2_switch"])
    )
    summary["c3r_gate_score_margin_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_gate_score_margin"])
    )
    summary["c3r_gate_leave_A_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_gate_leave_A"])
    )
    summary["c3r_gate_B_safe_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_gate_B_safe"])
    )
    summary["c3r_gate_uncertainty_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_gate_uncertainty"])
    )
    summary["c3r_allow_B_rate_given_c2_B"] = _rate_given(
        rows, c2_selected_B, lambda row: bool(row["c3r_allow_B"])
    )
    summary["c2_B_count"] = sum(1 for row in rows if c2_switch_to_B(row))
    summary["c3r_block_count"] = sum(1 for row in rows if c3r_block(row))
    summary["c3r_blocks_c2_switch_rate"] = _rate(rows, c3r_block)
    summary["c3r_block_rate_given_c2_B"] = _rate_given(rows, c2_switch_to_B, c3r_block)
    summary["c3r_harmful_c2_switch_rate"] = _rate(rows, harmful_c2_switch)
    summary["c3r_beneficial_c2_switch_rate"] = _rate(rows, beneficial_c2_switch)
    summary["c3r_prevented_harmful_switch_rate"] = _rate(
        rows, lambda row: bool(c3r_block(row) and harmful_c2_switch(row))
    )
    summary["c3r_missed_beneficial_switch_rate"] = _rate(
        rows, lambda row: bool(c3r_block(row) and beneficial_c2_switch(row))
    )
    summary["c3r_prevented_harmful_switch_rate_given_block"] = _rate_given(
        rows,
        c3r_block,
        harmful_c2_switch,
    )
    summary["c3r_missed_beneficial_switch_rate_given_block"] = _rate_given(
        rows,
        c3r_block,
        beneficial_c2_switch,
    )
    summary["c3r_harmful_block_precision"] = summary["c3r_prevented_harmful_switch_rate_given_block"]
    summary["c3r_harmful_switch_recall"] = _rate_given(rows, harmful_c2_switch, c3r_block)
    summary["c3r_beneficial_switch_block_rate"] = _rate_given(rows, beneficial_c2_switch, c3r_block)
    summary["c3r_beneficial_switch_retention"] = 1.0 - float(summary["c3r_beneficial_switch_block_rate"])
    summary["c3r_prevented_loss_sum"] = _sum([prevented_loss(row) for row in rows])
    summary["c3r_missed_gain_sum"] = _sum([missed_gain(row) for row in rows])
    summary["c3r_net_intervention_gain"] = float(
        summary["c3r_prevented_loss_sum"] - summary["c3r_missed_gain_sum"]
    )
    intervention_gains = [
        float(row.get("c3r_intervention_gain", float(row["fid_gain_C3R"]) - float(row["fid_gain_C2"])))
        for row in rows
    ]
    summary["c3r_intervention_gain_mean"] = _mean(intervention_gains)
    summary["c3r_intervention_gain_sum"] = _sum(intervention_gains)
    summary["c3r_score_margin_mean"] = _mean([float(row["c3r_score_margin"]) for row in rows])
    summary["c3r_structural_margin_mean"] = _mean([float(row["c3r_structural_margin"]) for row in rows])
    summary["c3r_violation_A_mean"] = _mean([float(row["c3r_violation_A"]) for row in rows])
    summary["c3r_violation_B_mean"] = _mean([float(row["c3r_violation_B"]) for row in rows])
    summary["c3r_syndrome_uncertainty_mean"] = _mean([float(row["c3r_syndrome_uncertainty"]) for row in rows])
    summary["c3r_raw_syndrome_uncertainty_mean"] = _mean(
        [float(row["c3r_raw_syndrome_uncertainty"]) for row in rows]
    )
    summary["c3r_blocked_by_score_margin_count"] = sum(
        1 for row in rows if str(row["decision_reason_C3R"]) == "c3r_blocks_low_score_margin"
    )
    summary["c3r_blocked_by_leave_A_count"] = sum(
        1 for row in rows if str(row["decision_reason_C3R"]) == "c3r_blocks_insufficient_A_risk"
    )
    summary["c3r_blocked_by_B_safe_count"] = sum(
        1 for row in rows if str(row["decision_reason_C3R"]) == "c3r_blocks_unsafe_B"
    )
    summary["c3r_blocked_by_uncertainty_count"] = sum(
        1 for row in rows if str(row["decision_reason_C3R"]) == "c3r_blocks_high_syndrome_uncertainty"
    )
    return summary


def _group_rows(rows: Sequence[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(dict(row))
    table = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_rows(grouped[group_key]))
        table.append(entry)
    return table


def _c3r_uncertainty_bin_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("[0.00,0.25)", 0.0, 0.25),
        ("[0.25,0.50)", 0.25, 0.50),
        ("[0.50,0.75)", 0.50, 0.75),
        ("[0.75,1.00)", 0.75, 1.00),
        ("[1.00,+inf)", 1.00, float("inf")),
    ]
    table: list[dict[str, Any]] = []
    for label, low, high in bins:
        subset = [
            dict(row)
            for row in rows
            if low <= float(row["c3r_raw_syndrome_uncertainty"]) < high
        ]
        if not subset:
            continue
        entry = {
            "c3r_raw_syndrome_uncertainty_bin": label,
            "uncertainty_min": float(low),
            "uncertainty_max": None if high == float("inf") else float(high),
        }
        entry.update(_aggregate_rows(subset))
        table.append(entry)
    return table


def _reason_rows(rows: Sequence[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    key = f"decision_reason_{mode}"
    for row in rows:
        reason = str(row[key])
        counts[reason] = counts.get(reason, 0) + 1
    total = len(list(rows))
    return [
        {"mode": mode, "reason": reason, "count": count, "rate": 0.0 if total == 0 else float(count / total)}
        for reason, count in sorted(counts.items())
    ]


def _build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Hybrid C1/C2/C3/C3R Baseline Summary",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "backend",
            "fid_gain_A_mean",
            "fid_gain_B_mean",
            "fid_gain_C1_mean",
            "fid_gain_C2_mean",
            "fid_gain_C3_mean",
            "fid_gain_C3R_mean",
            "logical_success_rate_C2",
            "logical_success_rate_C3R",
            "nonworsen_rate_C2",
            "nonworsen_rate_C3R",
            "false_safe_rate_A",
            "false_safe_rate_B",
            "false_safe_rate_C1",
            "false_safe_rate_C2",
            "false_safe_rate_C3",
            "false_safe_rate_C3R",
            "false_safe_fidelity_rate_A",
            "false_safe_fidelity_rate_B",
            "false_safe_fidelity_rate_C1",
            "false_safe_fidelity_rate_C2",
            "false_safe_fidelity_rate_C3",
            "false_safe_fidelity_rate_C3R",
            "fid_gain_q05_C2",
            "fid_gain_q05_C3R",
            "fid_gain_cvar05_C2",
            "fid_gain_cvar05_C3R",
            "chosen_B_rate_C2",
            "chosen_B_rate_C3R",
            "decision_disagreement_rate_C3R_vs_C2",
            "c3r_blocks_c2_switch_rate",
            "c3r_prevented_harmful_switch_rate_given_block",
            "c3r_missed_beneficial_switch_rate_given_block",
        ]),
        "",
        "## By Code and Noise Family",
        "",
        _markdown_table(result["tables"]["by_code_and_noise_family"], [
            "code_family",
            "noise_family",
            "cases",
            "fid_gain_A_mean",
            "fid_gain_B_mean",
            "fid_gain_C1_mean",
            "fid_gain_C2_mean",
            "fid_gain_C3_mean",
            "fid_gain_C3R_mean",
            "false_safe_rate_A",
            "false_safe_rate_C1",
            "false_safe_rate_C2",
            "false_safe_rate_C3",
            "false_safe_rate_C3R",
        ]),
        "",
        "## By Syndrome Observation Ratio",
        "",
        _markdown_table(result["tables"]["by_syndrome_obs_ratio"], [
            "syndrome_observation_ratio",
            "cases",
            "fid_gain_C1_mean",
            "fid_gain_C2_mean",
            "fid_gain_C3_mean",
            "fid_gain_C3R_mean",
            "false_safe_rate_C1",
            "false_safe_rate_C2",
            "false_safe_rate_C3",
            "false_safe_rate_C3R",
            "false_safe_fidelity_rate_C1",
            "false_safe_fidelity_rate_C2",
            "false_safe_fidelity_rate_C3",
            "false_safe_fidelity_rate_C3R",
        ]),
        "",
        "## By Syndrome Noise Probability",
        "",
        _markdown_table(result["tables"]["by_syndrome_noise_prob"], [
            "syndrome_noise_prob",
            "cases",
            "fid_gain_C1_mean",
            "fid_gain_C2_mean",
            "fid_gain_C3_mean",
            "fid_gain_C3R_mean",
            "false_safe_rate_C1",
            "false_safe_rate_C2",
            "false_safe_rate_C3",
            "false_safe_rate_C3R",
            "false_safe_fidelity_rate_C1",
            "false_safe_fidelity_rate_C2",
            "false_safe_fidelity_rate_C3",
            "false_safe_fidelity_rate_C3R",
        ]),
        "",
        "## C3R Gate Summary",
        "",
        _markdown_table([result["overall"]], [
            "c2_B_count",
            "c3r_block_count",
            "c3r_gate_c2_switch_rate",
            "c3r_gate_score_margin_rate",
            "c3r_gate_leave_A_rate",
            "c3r_gate_B_safe_rate",
            "c3r_gate_uncertainty_rate",
            "c3r_allow_B_rate",
            "c3r_gate_c2_switch_rate_given_c2_B",
            "c3r_gate_score_margin_rate_given_c2_B",
            "c3r_gate_leave_A_rate_given_c2_B",
            "c3r_gate_B_safe_rate_given_c2_B",
            "c3r_gate_uncertainty_rate_given_c2_B",
            "c3r_allow_B_rate_given_c2_B",
            "c3r_syndrome_uncertainty_mean",
            "c3r_raw_syndrome_uncertainty_mean",
            "c3r_blocked_by_score_margin_count",
            "c3r_blocked_by_leave_A_count",
            "c3r_blocked_by_B_safe_count",
            "c3r_blocked_by_uncertainty_count",
        ]),
        "",
        "## C3R Switch Intervention Quality",
        "",
        _markdown_table([result["overall"]], [
            "c2_B_count",
            "c3r_block_count",
            "c3r_block_rate_given_c2_B",
            "c3r_harmful_block_precision",
            "c3r_harmful_switch_recall",
            "c3r_beneficial_switch_block_rate",
            "c3r_beneficial_switch_retention",
            "c3r_prevented_loss_sum",
            "c3r_missed_gain_sum",
            "c3r_net_intervention_gain",
            "c3r_intervention_gain_sum",
        ]),
        "",
        "## Tail Risk and Oracle Diagnostics",
        "",
        _markdown_table([result["overall"]], [
            "fid_gain_q05_C2",
            "fid_gain_q05_C3R",
            "fid_gain_cvar05_C2",
            "fid_gain_cvar05_C3R",
            "observed_failure_boundary_rate_C2",
            "observed_failure_boundary_rate_C3R",
            "true_failure_boundary_rate_C2",
            "true_failure_boundary_rate_C3R",
            "chosen_candidate_violation_mean_C2",
            "chosen_candidate_violation_mean_C3R",
            "admissible_rate_C2",
            "admissible_rate_C3R",
            "oracle_regret_mean_C2",
            "oracle_regret_mean_C3R",
            "oracle_regret_q95_C2",
            "oracle_regret_q95_C3R",
        ]),
        "",
        "## C3R By Raw Uncertainty Bin",
        "",
        _markdown_table(result["tables"].get("c3r_by_uncertainty_bin", []), [
            "c3r_raw_syndrome_uncertainty_bin",
            "cases",
            "c2_B_count",
            "c3r_block_count",
            "c3r_block_rate_given_c2_B",
            "c3r_harmful_block_precision",
            "c3r_harmful_switch_recall",
            "c3r_beneficial_switch_block_rate",
            "c3r_net_intervention_gain",
        ]),
        "",
        "## Decision Reasons",
        "",
        _markdown_table(result["tables"]["reason_summary"], [
            "mode",
            "reason",
            "count",
            "rate",
        ]),
    ]
    return "\n".join(lines) + "\n"


def run(
    *,
    codes: Sequence[str],
    state_configs: dict[str, str],
    kinds_by_code: dict[str, Sequence[str]],
    noise_families: Sequence[str],
    strengths: Sequence[float],
    depths: Sequence[int],
    seeds: Sequence[int],
    fidelity_margin: float,
    logical_success_threshold: float,
    c2_cfg: PolicyScoreConfig,
    c3_cfg: PolicyScoreConfig,
    c3r_cfg: C3RPolicyConfig | None = None,
    syndrome_obs_cfg: SyndromeObservationConfig | None = None,
    c1_objective_tol: float,
    c1_tie_break_requires_syndrome_consistent: bool,
    experiment_config: str,
    output_stem: str,
) -> dict[str, Any]:
    syndrome_obs_cfg = syndrome_obs_cfg or SyndromeObservationConfig()
    c3r_cfg = c3r_cfg or C3RPolicyConfig()
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{output_stem}-{timestamp}"
    rows: list[dict[str, Any]] = []
    base_configs = {}
    for code in codes:
        cfg = load_config(experiment_config=experiment_config, state_config=state_configs[code])
        cfg.setdefault("hybrid_c123", {})
        cfg["hybrid_c123"]["kinds_by_code"] = {
            str(key): list(value) for key, value in kinds_by_code.items()
        }
        base_configs[str(code)] = cfg
    sample_index = 0
    for seed in seeds:
        for code in codes:
            base_config = deepcopy(base_configs[str(code)])
            for family in noise_families:
                for strength in strengths:
                    for depth in depths:
                        row = _sample_row(
                            run_id=run_id,
                            timestamp=timestamp,
                            config_name=str(experiment_config),
                            code_family=str(code),
                            state_config_path=str(state_configs[str(code)]),
                            base_config=base_config,
                            seed=int(seed),
                            sample_index=int(sample_index),
                            noise_family=str(family),
                            noise_strength=float(strength),
                            noise_depth=int(depth),
                            logical_success_threshold=float(logical_success_threshold),
                            fidelity_margin=float(fidelity_margin),
                            c2_cfg=c2_cfg,
                            c3_cfg=c3_cfg,
                            c3r_cfg=c3r_cfg,
                            syndrome_obs_cfg=syndrome_obs_cfg,
                            c1_objective_tol=float(c1_objective_tol),
                            c1_tie_break_requires_syndrome_consistent=bool(
                                c1_tie_break_requires_syndrome_consistent
                            ),
                        )
                        rows.append(row)
                        sample_index += 1
    result = {
        "grid": {
            "codes": [str(item) for item in codes],
            "noise_families": [str(item) for item in noise_families],
            "strengths": [float(item) for item in strengths],
            "depths": [int(item) for item in depths],
            "seeds": [int(item) for item in seeds],
            "fidelity_margin": float(fidelity_margin),
            "logical_success_threshold": float(logical_success_threshold),
        },
        "policies": {
            "c2": c2_cfg.__dict__,
            "c3": c3_cfg.__dict__,
            "c3r": c3r_cfg.__dict__,
            "c1": {
                "objective_tol": float(c1_objective_tol),
                "tie_break_requires_syndrome_consistent": bool(c1_tie_break_requires_syndrome_consistent),
            },
        },
        "syndrome_observation": syndrome_obs_cfg.__dict__,
        "overall": _aggregate_rows(rows),
        "rows": rows,
        "tables": {
            "by_code": _group_rows(rows, keys=("code_family",)),
            "by_noise_family": _group_rows(rows, keys=("noise_family",)),
            "by_code_and_noise_family": _group_rows(rows, keys=("code_family", "noise_family")),
            "by_syndrome_obs_ratio": _group_rows(rows, keys=("syndrome_observation_ratio",)),
            "by_code_and_obs_ratio": _group_rows(
                rows, keys=("code_family", "syndrome_observation_ratio")
            ),
            "by_code_noise_and_obs_ratio": _group_rows(
                rows, keys=("code_family", "noise_family", "syndrome_observation_ratio")
            ),
            "by_syndrome_noise_prob": _group_rows(rows, keys=("syndrome_noise_prob",)),
            "by_code_noise_and_syndrome_noise": _group_rows(
                rows, keys=("code_family", "noise_family", "syndrome_noise_prob")
            ),
            "by_code_noise_obs_and_syndrome_noise": _group_rows(
                rows,
                keys=(
                    "code_family",
                    "noise_family",
                    "syndrome_observation_ratio",
                    "syndrome_noise_prob",
                ),
            ),
            "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
            "reason_summary": [
                *_reason_rows(rows, mode="C1"),
                *_reason_rows(rows, mode="C2"),
                *_reason_rows(rows, mode="C3"),
                *_reason_rows(rows, mode="C3R"),
            ],
        },
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare A/B/C1/C2/C3/C3R hybrid recovery policies.")
    parser.add_argument("--config", default="experiment/hybrid_c123_baseline.yaml")
    parser.add_argument("--codes", default=None)
    parser.add_argument("--noise-families", default=None)
    parser.add_argument("--strengths", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def _parse_csv(raw: str | None, caster):
    if raw is None:
        return None
    return [caster(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    cfg = dict(config.get("hybrid_c123", {}))
    codes = _parse_csv(args.codes, str) or list(cfg.get("codes", DEFAULT_CODES))
    state_configs = {
        str(key): str(value)
        for key, value in dict(cfg.get("state_configs", {})).items()
    }
    kinds_by_code = {
        str(key): [str(item) for item in value]
        for key, value in dict(cfg.get("kinds_by_code", {})).items()
        if isinstance(value, (list, tuple))
    }
    noise_families = _parse_csv(args.noise_families, str) or [str(item) for item in cfg.get("noise_families", [])]
    strengths = _parse_csv(args.strengths, float) or [float(item) for item in cfg.get("strengths", [])]
    depths = _parse_csv(args.depths, int) or [int(item) for item in cfg.get("depths", [])]
    seeds = _parse_csv(args.seeds, int) or [int(item) for item in cfg.get("seeds", [])]
    c2 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c2", {})).items()})
    c3 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c3", {})).items()})
    c3r = C3RPolicyConfig(**{key: float(value) for key, value in dict(cfg.get("c3r", {})).items()})
    syndrome_obs = SyndromeObservationConfig(
        **{
            key: float(value)
            for key, value in dict(cfg.get("syndrome_observation", {})).items()
        }
    )
    c1_cfg = dict(cfg.get("c1", {}))
    stem = resolve_output_stem(config, "hybrid_c123_baseline", args.output_stem)
    result = run(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        noise_families=noise_families,
        strengths=strengths,
        depths=depths,
        seeds=seeds,
        fidelity_margin=float(cfg.get("fidelity_margin", 0.01)),
        logical_success_threshold=float(cfg.get("logical_success_threshold", 0.99)),
        c2_cfg=c2,
        c3_cfg=c3,
        c3r_cfg=c3r,
        syndrome_obs_cfg=syndrome_obs,
        c1_objective_tol=float(c1_cfg.get("objective_tol", 1.0e-9)),
        c1_tie_break_requires_syndrome_consistent=bool(
            c1_cfg.get("tie_break_requires_syndrome_consistent", True)
        ),
        experiment_config=args.config,
        output_stem=stem,
    )
    json_path = write_json_result(result, f"{stem}_summary")
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / "hybrid_c123_baseline_raw.csv", result["rows"])
    markdown_path = tables_dir / "hybrid_c123_baseline_summary.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

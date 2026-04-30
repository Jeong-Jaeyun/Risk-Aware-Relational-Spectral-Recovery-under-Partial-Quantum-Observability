from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .run_recovery_objective import run as run_recovery_objective


DEFAULT_NOISE_KINDS = ("bitflip", "phaseflip", "dephasing")
DEFAULT_STRENGTHS = (0.02, 0.05, 0.08, 0.12, 0.16)
DEFAULT_KAPPAS = (1.0, 1.5, 2.0)
DEFAULT_BANK_WIDTHS_DEG = (5.0, 10.0, 20.0)
NOISE_QUBITS = {
    "bitflip": 0,
    "phaseflip": 0,
    "dephasing": 1,
    "depolarizing": 0,
}


TERM_KEYS = ("obs_fit", "ref_anchor", "lap", "smooth", "clock", "phi_ref")


def _parse_csv_list(raw: str | None, caster) -> list[Any] | None:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",")]
    values = [part for part in parts if part]
    if not values:
        return []
    return [caster(item) for item in values]


def _phase_coeff_payload(angle: float) -> list[dict[str, float]]:
    return [
        {"real": 1.0, "imag": 0.0},
        {"real": float(np.cos(angle)), "imag": float(np.sin(angle))},
    ]


def _reference_bank_for_width(width_deg: float) -> list[dict[str, Any]]:
    width_rad = float(np.deg2rad(width_deg))
    center = float(np.pi / 2.0)
    return [
        {
            "label": "phase_pi_over_2",
            "state": {
                "mode": "null",
                "align_spectra": True,
                "null_mode": "custom",
                "null_coeffs": _phase_coeff_payload(center),
            },
        },
        {
            "label": f"phase_minus_{width_deg:g}deg",
            "state": {
                "mode": "null",
                "align_spectra": True,
                "null_mode": "custom",
                "null_coeffs": _phase_coeff_payload(center - width_rad),
            },
        },
        {
            "label": f"phase_plus_{width_deg:g}deg",
            "state": {
                "mode": "null",
                "align_spectra": True,
                "null_mode": "custom",
                "null_coeffs": _phase_coeff_payload(center + width_rad),
            },
        },
    ]


def _scenario_config(
    base_config: dict[str, Any],
    *,
    noise_kind: str,
    noise_strength: float,
    admissibility_kappa: float,
    bank_width_deg: float,
) -> dict[str, Any]:
    if noise_kind not in NOISE_QUBITS:
        raise ValueError(f"Unsupported sweep noise kind: {noise_kind!r}")
    config = deepcopy(base_config)
    config.setdefault("noise", {})
    config["noise"]["schedule"] = [{
        "kind": noise_kind,
        "qubit": int(NOISE_QUBITS[noise_kind]),
        "p": float(noise_strength),
    }]
    config.setdefault("admissibility", {})
    config["admissibility"]["kappa"] = float(admissibility_kappa)
    config.setdefault("reference_bank", {})
    config["reference_bank"]["trajectories"] = _reference_bank_for_width(bank_width_deg)
    return config


def _row_from_result(
    *,
    noise_kind: str,
    noise_strength: float,
    admissibility_kappa: float,
    bank_width_deg: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    summary = result["summary"]
    stage2 = result["stage2_refinement"]
    decomposition = result["term_decomposition"]
    best_admissible_weights = stage2.get("best_admissible_weights")
    best_admissible_objective = summary.get("stage2_best_admissible_objective")
    has_interior = (
        best_admissible_weights is not None
        and sum(1 for weight in best_admissible_weights if float(weight) > 1e-9) > 1
    )
    clean_gain = (
        float(summary["clean_to_observed_distance"])
        - float(summary["clean_to_recovered_distance"])
    )
    candidate_objective_gain = (
        None
        if best_admissible_objective is None
        else float(summary["stage2_candidate_objective_gain_vs_stage1"])
    )
    stage2_candidate_clean_delta = summary["clean_to_stage2_candidate_distance"]
    if stage2_candidate_clean_delta is None:
        candidate_taxonomy = "no_feasible_stage2"
        candidate_location = "no_feasible_stage2"
    else:
        candidate_location = "interior" if has_interior else "boundary"
        objective_improves = (candidate_objective_gain is not None) and (candidate_objective_gain > 1e-9)
        clean_improves = (
            float(summary["clean_to_stage2_candidate_distance"])
            - float(summary["clean_to_stage1_distance"])
        ) <= 1e-9
        if objective_improves and clean_improves:
            candidate_taxonomy = "objective_improves_clean_improves"
        elif objective_improves and not clean_improves:
            candidate_taxonomy = "objective_improves_clean_worsens"
        elif (not objective_improves) and clean_improves:
            candidate_taxonomy = "objective_worsens_clean_improves"
        else:
            candidate_taxonomy = "objective_worsens_clean_worsens"

    weighted_deltas = decomposition.get("delta_stage2_candidate_weighted_vs_stage1") or {}
    dominant_weighted_term = None
    if weighted_deltas:
        dominant_weighted_term = min(weighted_deltas, key=lambda key: weighted_deltas[key])
    return {
        "noise_kind": noise_kind,
        "noise_strength": float(noise_strength),
        "admissibility_kappa": float(admissibility_kappa),
        "bank_width_deg": float(bank_width_deg),
        "stage1_best_label": summary["stage1_best_label"],
        "reference_anchor_label": summary["reference_anchor_label"],
        "stage2_applied": bool(summary["stage2_applied"]),
        "final_stage": summary["final_stage"],
        "recovered_objective": float(summary["recovered_objective"]),
        "stage1_objective": float(summary["stage1_objective"]),
        "objective_gain_vs_reference_anchor": float(
            summary["objective_gain_vs_reference_anchor"]
        ),
        "observed_admissible": bool(summary["observed_admissible"]),
        "stage1_recovered_admissible": bool(summary["stage1_recovered_admissible"]),
        "recovered_admissible": bool(summary["recovered_admissible"]),
        "clean_to_stage1_distance": float(summary["clean_to_stage1_distance"]),
        "clean_to_reference_anchor_distance": float(summary["clean_to_reference_anchor_distance"]),
        "clean_to_observed_distance": float(summary["clean_to_observed_distance"]),
        "clean_to_recovered_distance": float(summary["clean_to_recovered_distance"]),
        "stage2_clean_distance_delta_vs_stage1": float(
            summary["stage2_clean_distance_delta_vs_stage1"]
        ),
        "observed_to_stage1_distance": float(summary["observed_to_stage1_distance"]),
        "observed_to_recovered_distance": float(summary["observed_to_recovered_distance"]),
        "clean_distance_gain": clean_gain,
        "stage2_feasible_count": int(stage2.get("feasible_count", 0)),
        "stage2_interior_feasible_count": int(stage2.get("interior_feasible_count", 0)),
        "stage2_best_admissible_weights": best_admissible_weights,
        "stage2_best_admissible_objective": (
            None if best_admissible_objective is None else float(best_admissible_objective)
        ),
        "stage2_candidate_objective_gain_vs_stage1": (
            None if candidate_objective_gain is None else float(candidate_objective_gain)
        ),
        "stage2_applied_objective_gain_vs_stage1": float(summary["stage2_objective_gain_vs_stage1"]),
        "stage2_has_interior_best_admissible_point": bool(has_interior),
        "stage2_candidate_location": candidate_location,
        "stage2_candidate_taxonomy": candidate_taxonomy,
        "stage2_candidate_dominant_weighted_term": dominant_weighted_term,
        "stage2_candidate_clean_distance_delta_vs_stage1": (
            None
            if summary["clean_to_stage2_candidate_distance"] is None
            else float(summary["clean_to_stage2_candidate_distance"])
            - float(summary["clean_to_stage1_distance"])
        ),
        "stage2_candidate_observed_distance_delta_vs_stage1": (
            None
            if summary["observed_to_stage2_candidate_distance"] is None
            else float(summary["observed_to_stage2_candidate_distance"])
            - float(summary["observed_to_stage1_distance"])
        ),
        "stage2_candidate_obs_fit_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("obs_fit", 0.0))
        ),
        "stage2_candidate_lap_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("lap", 0.0))
        ),
        "stage2_candidate_smooth_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("smooth", 0.0))
        ),
        "stage2_candidate_ref_anchor_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("ref_anchor", 0.0))
        ),
        "stage2_candidate_phi_ref_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("phi_ref", 0.0))
        ),
        "stage2_candidate_clock_delta": (
            None
            if decomposition["delta_stage2_candidate_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_vs_stage1"].get("clock", 0.0))
        ),
        "stage2_candidate_weighted_obs_fit_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("obs_fit", 0.0))
        ),
        "stage2_candidate_weighted_lap_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("lap", 0.0))
        ),
        "stage2_candidate_weighted_smooth_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("smooth", 0.0))
        ),
        "stage2_candidate_weighted_ref_anchor_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("ref_anchor", 0.0))
        ),
        "stage2_candidate_weighted_phi_ref_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("phi_ref", 0.0))
        ),
        "stage2_candidate_weighted_clock_delta": (
            None
            if decomposition["delta_stage2_candidate_weighted_vs_stage1"] is None
            else float(decomposition["delta_stage2_candidate_weighted_vs_stage1"].get("clock", 0.0))
        ),
    }


def _mean_bool(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [1.0 if bool(row[key]) else 0.0 for row in rows]
    return 0.0 if not values else float(mean(values))


def _mean_numeric(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(mean(values))


def _rate_by_value(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not rows:
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    total = float(len(rows))
    return {
        value: float(count / total)
        for value, count in sorted(counts.items())
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def improvement_stats(group_rows: list[dict[str, Any]]) -> tuple[float, float | None]:
        gains = [
            float(row["stage2_candidate_objective_gain_vs_stage1"])
            for row in group_rows
            if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
            and float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1e-9
        ]
        if not group_rows:
            return 0.0, None
        return (
            float(len(gains) / len(group_rows)),
            None if not gains else float(mean(gains)),
        )

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        improvement_rate, mean_improvement = improvement_stats(group_rows)
        return {
            "cases": len(group_rows),
            "observed_nonadmissible_rate": (
                0.0 if not group_rows else 1.0 - _mean_bool(group_rows, "observed_admissible")
            ),
            "recovered_admissible_rate": _mean_bool(group_rows, "recovered_admissible"),
            "stage2_applied_rate": _mean_bool(group_rows, "stage2_applied"),
            "interior_admissible_stage2_rate": _mean_bool(
                group_rows, "stage2_has_interior_best_admissible_point"
            ),
            "mean_clean_distance_gain": _mean_numeric(group_rows, "clean_distance_gain"),
            "mean_stage2_candidate_objective_gain_vs_stage1": _mean_numeric(
                group_rows, "stage2_candidate_objective_gain_vs_stage1"
            ),
            "mean_stage2_applied_objective_gain_vs_stage1": _mean_numeric(
                group_rows, "stage2_applied_objective_gain_vs_stage1"
            ),
            "stage2_improvement_rate_vs_stage1": improvement_rate,
            "mean_stage2_candidate_objective_gain_vs_stage1_positive": mean_improvement,
            "location_rate": _rate_by_value(group_rows, "stage2_candidate_location"),
            "taxonomy_rate": _rate_by_value(group_rows, "stage2_candidate_taxonomy"),
            "dominant_weighted_term_rate": _rate_by_value(
                [row for row in group_rows if row.get("stage2_candidate_dominant_weighted_term") is not None],
                "stage2_candidate_dominant_weighted_term",
            ),
        }

    by_noise_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_noise_kind.setdefault(str(row["noise_kind"]), []).append(row)
    return {
        "overall": summarize(rows),
        "by_noise_kind": {
            kind: summarize(group_rows)
            for kind, group_rows in sorted(by_noise_kind.items())
        },
    }


def _write_csv_rows(rows: list[dict[str, Any]], stem: str) -> Path:
    target_dir = RESULT_ROOT / "tables"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def run(
    base_config: dict[str, Any],
    *,
    noise_kinds: Iterable[str] = DEFAULT_NOISE_KINDS,
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
    kappas: Iterable[float] = DEFAULT_KAPPAS,
    bank_widths_deg: Iterable[float] = DEFAULT_BANK_WIDTHS_DEG,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for noise_kind in noise_kinds:
        for noise_strength in strengths:
            for admissibility_kappa in kappas:
                for bank_width_deg in bank_widths_deg:
                    scenario = _scenario_config(
                        base_config,
                        noise_kind=str(noise_kind),
                        noise_strength=float(noise_strength),
                        admissibility_kappa=float(admissibility_kappa),
                        bank_width_deg=float(bank_width_deg),
                    )
                    result = run_recovery_objective(scenario)
                    rows.append(_row_from_result(
                        noise_kind=str(noise_kind),
                        noise_strength=float(noise_strength),
                        admissibility_kappa=float(admissibility_kappa),
                        bank_width_deg=float(bank_width_deg),
                        result=result,
                    ))
    return {
        "grid": {
            "noise_kinds": [str(item) for item in noise_kinds],
            "strengths": [float(item) for item in strengths],
            "kappas": [float(item) for item in kappas],
            "bank_widths_deg": [float(item) for item in bank_widths_deg],
        },
        "summary": _summarize_rows(rows),
        "rows": rows,
    }


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep recovery behavior over noise, admissibility kappa, and bank width."
    )
    parser.add_argument("--config", default="experiment/recovery_eval.yaml")
    parser.add_argument("--state-config", default="states/null_dynamic.yaml")
    parser.add_argument("--noise-kinds", default="bitflip,phaseflip,dephasing")
    parser.add_argument("--strengths", default="0.02,0.05,0.08,0.12,0.16")
    parser.add_argument("--kappas", default="1.0,1.5,2.0")
    parser.add_argument("--bank-widths-deg", default="5,10,20")
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_sweep_parser()
    args = parser.parse_args()
    base_config = load_config(
        experiment_config=args.config,
        state_config=args.state_config,
    )
    result = run(
        base_config,
        noise_kinds=_parse_csv_list(args.noise_kinds, str) or list(DEFAULT_NOISE_KINDS),
        strengths=_parse_csv_list(args.strengths, float) or list(DEFAULT_STRENGTHS),
        kappas=_parse_csv_list(args.kappas, float) or list(DEFAULT_KAPPAS),
        bank_widths_deg=_parse_csv_list(args.bank_widths_deg, float) or list(DEFAULT_BANK_WIDTHS_DEG),
    )
    stem = resolve_output_stem(base_config, "recovery_sweep", args.output_stem)
    json_path = write_json_result(result, stem)
    csv_path = _write_csv_rows(result["rows"], stem)
    print(json.dumps(to_serializable(result["summary"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()

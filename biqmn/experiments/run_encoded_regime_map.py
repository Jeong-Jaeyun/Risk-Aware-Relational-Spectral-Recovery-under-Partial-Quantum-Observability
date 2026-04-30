"""Regime-map sweep for the encoded A/B/C QEC baseline.

The sweep fixes a grid over

    code x noise_family x noise_strength x noise_depth x seed

and runs the encoded syndrome / relational / hybrid baselines for every grid
cell. Outputs are structured so they can drive:

* by-code / by-family / by-strength / by-depth tables
* disagreement analysis
* failure-boundary summaries
* recovery-mode A/B/C comparison tables
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from .common import (
    RESULT_ROOT,
    load_config,
    resolve_output_stem,
    to_serializable,
    write_json_result,
)
from .run_encoded_qec_baseline import (
    _distribution_stats,
    _markdown_table,
    _sample_row,
    _write_csv,
)


DEFAULT_CODES = ("bitflip", "phaseflip")
DEFAULT_FAMILIES = (
    "bitflip",
    "phaseflip",
    "dephasing",
    "depolarizing",
    "amplitude_damping",
    "coherent_x",
    "coherent_z",
    "mixed",
)
DEFAULT_STRENGTHS = (0.03, 0.10, 0.15)
DEFAULT_DEPTHS = (1, 2, 3)
DEFAULT_SEEDS = (11, 17, 23)


def _parse_csv_list(raw: str | None, caster) -> list[Any] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    return [caster(item) for item in values]


def _parse_int_ranges(raw: str | None) -> list[int] | None:
    return _parse_csv_list(raw, int)


def _resolve_state_configs(raw: Any, fallback: dict[str, str]) -> dict[str, str]:
    resolved = dict(fallback)
    if isinstance(raw, dict):
        resolved.update({str(key): str(value) for key, value in raw.items()})
    return resolved


def _resolve_kinds_by_code(raw: Any, fallback: Sequence[str]) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, (list, tuple)):
                resolved[str(key)] = [str(item) for item in value]
    for code in DEFAULT_CODES:
        resolved.setdefault(str(code), [str(item) for item in fallback])
    return resolved


def _resolve_families_by_code(raw: Any) -> dict[str, list[str]] | None:
    if not isinstance(raw, dict):
        return None
    resolved: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            resolved[str(key)] = [str(item) for item in value]
    return resolved or None


def _schedule_for_family(
    rng: np.random.Generator,
    *,
    code: str,
    family: str,
    strength: float,
    depth: int,
    n_system: int,
    kinds_by_code: dict[str, Sequence[str]],
) -> list[dict[str, Any]]:
    if int(depth) <= 0:
        raise ValueError("Regime-map depth must be positive.")
    if family == "mixed":
        kind_space = list(kinds_by_code.get(code, []))
        if not kind_space:
            raise ValueError(f"No kind space configured for code={code!r}")
    else:
        kind_space = [str(family)]
    schedule = []
    for _ in range(int(depth)):
        kind = str(rng.choice(kind_space))
        schedule.append({
            "kind": kind,
            "qubit": int(rng.integers(0, int(n_system))),
            "p": float(strength),
        })
    return schedule


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


def _rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return float(sum(1.0 if predicate(row) else 0.0 for row in rows) / len(rows))


def _aggregate_rows(rows: list[dict[str, Any]], *, fidelity_margin: float) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    summary = {
        "cases": len(rows),
        "code_count": len(sorted({str(row["code_type"]) for row in rows})),
        "mean_noise_strength": _mean([float(row["noise_strength"]) for row in rows]),
        "mean_noise_depth": _mean([float(row["noise_depth"]) for row in rows]),
        "fidelity_before_mean": _mean([float(row["fidelity_before"]) for row in rows]),
        "fidelity_after_A_mean": _mean([float(row["fidelity_after_A"]) for row in rows]),
        "fidelity_after_B_mean": _mean([float(row["fidelity_after_B"]) for row in rows]),
        "fidelity_after_C_mean": _mean([float(row["fidelity_after_C"]) for row in rows]),
        "gain_A_mean": _mean([float(row["gain_A"]) for row in rows]),
        "gain_B_mean": _mean([float(row["gain_B"]) for row in rows]),
        "gain_C_mean": _mean([float(row["gain_C"]) for row in rows]),
        "nonworsen_rate_A": _rate(rows, lambda row: bool(row["A_recovery_nonworsen"])),
        "nonworsen_rate_B": _rate(rows, lambda row: bool(row["B_recovery_nonworsen"])),
        "nonworsen_rate_C": _rate(rows, lambda row: bool(row["C_recovery_nonworsen"])),
        "admissible_rate_A": _rate(rows, lambda row: bool(row["admissible_A"])),
        "admissible_rate_B": _rate(rows, lambda row: bool(row["admissible_B"])),
        "admissible_rate_C": _rate(rows, lambda row: bool(row["recovered_C_admissible"])),
        "hybrid_use_relational_rate": _rate(rows, lambda row: bool(row["hybrid_use_relational"])),
        "veto_rate": _rate(rows, lambda row: str(row["reason_C"]) == "veto_nonadmissible_A"),
        "tie_break_rate": _rate(rows, lambda row: str(row["reason_C"]) == "tie_break_objective"),
        "false_safe_rate": _rate(rows, lambda row: bool(row["syndrome_consistent_but_trajectory_inconsistent"])),
        "decision_disagreement_rate": _rate(rows, lambda row: bool(row["hybrid_use_relational"])),
        "relational_better_rate": _rate(
            rows,
            lambda row: float(row["fidelity_after_B"]) > float(row["fidelity_after_A"]) + float(fidelity_margin),
        ),
        "conservative_gap_rate": _rate(
            rows,
            lambda row: (
                str(row["chosen_C"]) == "A"
                and float(row["fidelity_after_B"]) > float(row["fidelity_after_C"]) + float(fidelity_margin)
            ),
        ),
        "A_inadmissible_B_admissible_rate": _rate(
            rows,
            lambda row: (not bool(row["admissible_A"])) and bool(row["admissible_B"]),
        ),
        "failure_boundary_rate": _rate(
            rows,
            lambda row: (
                bool(row["syndrome_consistent"])
                and (
                    (not bool(row["admissible_A"]))
                    or float(row["fidelity_after_B"]) > float(row["fidelity_after_A"]) + float(fidelity_margin)
                )
            ),
        ),
        "simulation_backend": str(rows[0]["backend"]),
        "seed_count": len(sorted({int(row["seed"]) for row in rows})),
    }
    summary.update(_distribution_stats([float(row["gain_A"]) for row in rows], "gain_A"))
    summary.update(_distribution_stats([float(row["gain_B"]) for row in rows], "gain_B"))
    summary.update(_distribution_stats([float(row["gain_C"]) for row in rows], "gain_C"))
    summary.update(_distribution_stats([float(row["fidelity_after_B"]) - float(row["fidelity_after_A"]) for row in rows], "fid_B_minus_A"))
    summary.update(_distribution_stats([float(row["clean_observed_distance"]) for row in rows], "clean_observed_distance"))
    return summary


def _group_rows(rows: list[dict[str, Any]], *, keys: Sequence[str], fidelity_margin: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    table = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_rows(grouped[group_key], fidelity_margin=float(fidelity_margin)))
        table.append(entry)
    return table


def _build_mode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_rows: list[dict[str, Any]] = []
    for row in rows:
        for mode in ("A", "B", "C"):
            mode_rows.append({
                "experiment_id": row["experiment_id"],
                "seed": row["seed"],
                "code_type": row["code_type"],
                "noise_family": row["noise_family"],
                "noise_strength": row["noise_strength"],
                "noise_depth": row["noise_depth"],
                "recovery_mode": mode,
                "fidelity_after": row[f"fidelity_after_{mode}"],
                "gain": row[f"gain_{mode}"],
                "admissible": row["recovered_C_admissible"] if mode == "C" else row[f"admissible_{mode}"],
                "nonworsen": row[f"{mode}_recovery_nonworsen"],
            })
    return mode_rows


def _aggregate_mode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    summary = {
        "cases": len(rows),
        "fidelity_after_mean": _mean([float(row["fidelity_after"]) for row in rows]),
        "gain_mean": _mean([float(row["gain"]) for row in rows]),
        "admissible_rate": _rate(rows, lambda row: bool(row["admissible"])),
        "nonworsen_rate": _rate(rows, lambda row: bool(row["nonworsen"])),
    }
    summary.update(_distribution_stats([float(row["gain"]) for row in rows], "gain"))
    return summary


def _group_mode_rows(rows: list[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    table = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_mode_rows(grouped[group_key]))
        table.append(entry)
    return table


def _top_boundary_rows(rows: list[dict[str, Any]], *, fidelity_margin: float, limit: int = 12) -> list[dict[str, Any]]:
    ranked = [
        row for row in rows
        if bool(row["syndrome_consistent"])
        and (
            (not bool(row["admissible_A"]))
            or float(row["fidelity_after_B"]) > float(row["fidelity_after_A"]) + float(fidelity_margin)
        )
    ]
    ranked.sort(
        key=lambda row: (
            float(row["fidelity_after_B"]) - float(row["fidelity_after_A"]),
            float(row["clean_observed_distance"]),
        ),
        reverse=True,
    )
    return ranked[: int(limit)]


def _build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Encoded QEC Regime Map",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "simulation_backend",
            "gain_A_mean",
            "gain_B_mean",
            "gain_C_mean",
            "false_safe_rate",
            "decision_disagreement_rate",
            "veto_rate",
            "tie_break_rate",
            "failure_boundary_rate",
        ]),
        "",
        "## By Code",
        "",
        _markdown_table(result["tables"]["by_code"], [
            "code_type",
            "cases",
            "gain_A_mean",
            "gain_B_mean",
            "gain_C_mean",
            "false_safe_rate",
            "decision_disagreement_rate",
            "failure_boundary_rate",
        ]),
        "",
        "## By Noise Family",
        "",
        _markdown_table(result["tables"]["by_noise_family"], [
            "noise_family",
            "cases",
            "gain_A_mean",
            "gain_B_mean",
            "gain_C_mean",
            "false_safe_rate",
            "decision_disagreement_rate",
            "failure_boundary_rate",
        ]),
        "",
        "## By Noise Strength",
        "",
        _markdown_table(result["tables"]["by_noise_strength"], [
            "noise_strength",
            "cases",
            "gain_A_mean",
            "gain_B_mean",
            "gain_C_mean",
            "false_safe_rate",
            "decision_disagreement_rate",
            "failure_boundary_rate",
        ]),
        "",
        "## By Noise Depth",
        "",
        _markdown_table(result["tables"]["by_noise_depth"], [
            "noise_depth",
            "cases",
            "gain_A_mean",
            "gain_B_mean",
            "gain_C_mean",
            "false_safe_rate",
            "decision_disagreement_rate",
            "failure_boundary_rate",
        ]),
        "",
        "## Disagreement Analysis",
        "",
        _markdown_table(result["tables"]["disagreement_analysis"], [
            "code_type",
            "noise_family",
            "cases",
            "decision_disagreement_rate",
            "veto_rate",
            "tie_break_rate",
            "relational_better_rate",
            "conservative_gap_rate",
        ]),
        "",
        "## Failure Boundary Summary",
        "",
        _markdown_table(result["tables"]["failure_boundary_summary"], [
            "code_type",
            "noise_family",
            "noise_strength",
            "noise_depth",
            "cases",
            "false_safe_rate",
            "failure_boundary_rate",
            "decision_disagreement_rate",
        ]),
        "",
        "## Recovery Mode Summary",
        "",
        _markdown_table(result["tables"]["by_recovery_mode"], [
            "recovery_mode",
            "cases",
            "fidelity_after_mean",
            "gain_mean",
            "admissible_rate",
            "nonworsen_rate",
        ]),
    ]
    if result["tables"]["top_failure_boundary_cases"]:
        lines.extend([
            "",
            "## Top Failure Boundary Cases",
            "",
            _markdown_table(result["tables"]["top_failure_boundary_cases"], [
                "experiment_id",
                "code_type",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "syndrome",
                "reason_C",
                "admissible_A",
                "admissible_B",
                "fidelity_after_A",
                "fidelity_after_B",
                "clean_observed_distance",
            ]),
        ])
    return "\n".join(lines) + "\n"


def run(
    *,
    codes: Sequence[str],
    state_configs: dict[str, str],
    kinds_by_code: dict[str, Sequence[str]],
    noise_families: Sequence[str],
    families_by_code: dict[str, Sequence[str]] | None = None,
    strengths: Sequence[float],
    depths: Sequence[int],
    seeds: Sequence[int],
    fidelity_margin: float,
    trajectory_inconsistency_threshold: float,
    syndrome_consistent_threshold: float,
    hybrid_objective_tol: float,
    tie_break_requires_syndrome_consistent: bool,
    experiment_config: str = "experiment/encoded_qec_baseline.yaml",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    base_configs = {
        code: load_config(experiment_config=experiment_config, state_config=state_configs[code])
        for code in codes
    }
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for code in codes:
            base_config = deepcopy(base_configs[code])
            n_system = int(base_config.get("system", {}).get("n_qubits", 3))
            family_list = list(families_by_code.get(str(code), noise_families)) if families_by_code else list(noise_families)
            for family in family_list:
                for strength in strengths:
                    for depth in depths:
                        schedule = _schedule_for_family(
                            rng,
                            code=str(code),
                            family=str(family),
                            strength=float(strength),
                            depth=int(depth),
                            n_system=n_system,
                            kinds_by_code=kinds_by_code,
                        )
                        row = _sample_row(
                            code=str(code),
                            base_seed=int(seed),
                            sample_index=int(len(rows)),
                            schedule=schedule,
                            state_config_path=state_configs[code],
                            base_config=base_config,
                            trajectory_inconsistency_threshold=float(trajectory_inconsistency_threshold),
                            syndrome_consistent_threshold=float(syndrome_consistent_threshold),
                            hybrid_objective_tol=float(hybrid_objective_tol),
                            tie_break_requires_syndrome_consistent=bool(
                                tie_break_requires_syndrome_consistent
                            ),
                        )
                        row["noise_strength"] = float(strength)
                        rows.append(row)

    mode_rows = _build_mode_rows(rows)
    tables = {
        "by_code": _group_rows(rows, keys=("code_type",), fidelity_margin=float(fidelity_margin)),
        "by_noise_family": _group_rows(rows, keys=("noise_family",), fidelity_margin=float(fidelity_margin)),
        "by_noise_strength": _group_rows(rows, keys=("noise_strength",), fidelity_margin=float(fidelity_margin)),
        "by_noise_depth": _group_rows(rows, keys=("noise_depth",), fidelity_margin=float(fidelity_margin)),
        "by_code_and_noise_family": _group_rows(
            rows,
            keys=("code_type", "noise_family"),
            fidelity_margin=float(fidelity_margin),
        ),
        "by_code_and_noise_strength": _group_rows(
            rows,
            keys=("code_type", "noise_strength"),
            fidelity_margin=float(fidelity_margin),
        ),
        "by_code_and_noise_depth": _group_rows(
            rows,
            keys=("code_type", "noise_depth"),
            fidelity_margin=float(fidelity_margin),
        ),
        "disagreement_analysis": _group_rows(
            rows,
            keys=("code_type", "noise_family"),
            fidelity_margin=float(fidelity_margin),
        ),
        "failure_boundary_summary": _group_rows(
            rows,
            keys=("code_type", "noise_family", "noise_strength", "noise_depth"),
            fidelity_margin=float(fidelity_margin),
        ),
        "by_seed": _group_rows(rows, keys=("seed",), fidelity_margin=float(fidelity_margin)),
        "by_recovery_mode": _group_mode_rows(mode_rows, keys=("recovery_mode",)),
        "by_code_and_recovery_mode": _group_mode_rows(mode_rows, keys=("code_type", "recovery_mode")),
        "top_failure_boundary_cases": _top_boundary_rows(rows, fidelity_margin=float(fidelity_margin), limit=12),
    }
    result = {
        "grid": {
            "codes": [str(item) for item in codes],
            "noise_families": [str(item) for item in noise_families],
            "families_by_code": (
                {
                    str(key): [str(item) for item in value]
                    for key, value in families_by_code.items()
                }
                if families_by_code
                else None
            ),
            "strengths": [float(item) for item in strengths],
            "depths": [int(item) for item in depths],
            "seeds": [int(item) for item in seeds],
            "fidelity_margin": float(fidelity_margin),
        },
        "overall": _aggregate_rows(rows, fidelity_margin=float(fidelity_margin)),
        "tables": tables,
        "rows": rows,
        "mode_rows": mode_rows,
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the encoded-QEC regime map sweep.")
    parser.add_argument("--config", default="experiment/encoded_regime_map.yaml")
    parser.add_argument("--codes", default=None)
    parser.add_argument("--noise-families", default=None)
    parser.add_argument("--strengths", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--fidelity-margin", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(experiment_config=args.config)
    encoded_cfg = dict(base_config.get("encoded_qec", {}))
    regime_cfg = dict(base_config.get("encoded_regime_map", {}))
    default_state_configs = {
        "bitflip": "states/repetition_bitflip.yaml",
        "phaseflip": "states/repetition_phaseflip.yaml",
    }
    state_configs = _resolve_state_configs(encoded_cfg.get("state_configs"), default_state_configs)
    fallback_kinds = list(
        encoded_cfg.get(
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
    )
    kinds_by_code = _resolve_kinds_by_code(encoded_cfg.get("kinds_by_code"), fallback_kinds)
    families_by_code = _resolve_families_by_code(regime_cfg.get("families_by_code"))
    codes = _parse_csv_list(args.codes, str) or list(regime_cfg.get("codes", encoded_cfg.get("codes", DEFAULT_CODES)))
    noise_families = _parse_csv_list(args.noise_families, str) or list(regime_cfg.get("noise_families", DEFAULT_FAMILIES))
    strengths = _parse_csv_list(args.strengths, float) or [float(item) for item in regime_cfg.get("strengths", DEFAULT_STRENGTHS)]
    depths = _parse_int_ranges(args.depths) or [int(item) for item in regime_cfg.get("depths", DEFAULT_DEPTHS)]
    seeds = _parse_int_ranges(args.seeds) or [int(item) for item in regime_cfg.get("seeds", DEFAULT_SEEDS)]
    fidelity_margin = float(args.fidelity_margin or regime_cfg.get("fidelity_margin", 0.01))
    hybrid_cfg = dict(encoded_cfg.get("hybrid", {}))
    result = run(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        noise_families=noise_families,
        families_by_code=families_by_code,
        strengths=strengths,
        depths=depths,
        seeds=seeds,
        fidelity_margin=fidelity_margin,
        trajectory_inconsistency_threshold=float(encoded_cfg.get("trajectory_inconsistency_threshold", 0.05)),
        syndrome_consistent_threshold=float(encoded_cfg.get("syndrome_consistent_threshold", 0.9)),
        hybrid_objective_tol=float(hybrid_cfg.get("objective_tol", 1.0e-9)),
        tie_break_requires_syndrome_consistent=bool(
            hybrid_cfg.get("tie_break_requires_syndrome_consistent", True)
        ),
        experiment_config=args.config,
    )
    stem = resolve_output_stem(base_config, "encoded_regime_map", args.output_stem)
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_rows.csv", result["rows"])
    _write_csv(tables_dir / f"{stem}_mode_rows.csv", result["mode_rows"])
    for name, rows in result["tables"].items():
        if isinstance(rows, list):
            _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

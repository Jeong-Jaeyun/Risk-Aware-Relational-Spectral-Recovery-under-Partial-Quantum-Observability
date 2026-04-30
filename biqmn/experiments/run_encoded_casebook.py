"""Representative-case extractor for the encoded QEC A/B/C baseline.

This runner reuses :mod:`run_encoded_qec_baseline` and selects a compact set of
human-readable cases that support the paper-facing message:

* syndrome consistency can still hide trajectory inconsistency
* relational recovery can outperform the syndrome baseline
* hybrid C intervenes either by veto or by objective tie-break
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, Sequence

from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv, run as run_encoded_qec_baseline


Predicate = Callable[[Dict[str, Any], float, float], bool]
SortKey = Callable[[Dict[str, Any]], Any]


def _project_case(row: dict[str, Any], *, group: str, title: str, rank: int) -> dict[str, Any]:
    return {
        "group": group,
        "group_title": title,
        "rank_in_group": int(rank),
        "experiment_id": row["experiment_id"],
        "code_type": row["code_type"],
        "noise_family": row["noise_family"],
        "noise_strength": row["noise_strength"],
        "noise_depth": row["noise_depth"],
        "seed": row["seed"],
        "backend": row["backend"],
        "clean_state_label": row["clean_state_label"],
        "syndrome": row["syndrome"],
        "candidate_A": row["candidate_A"],
        "candidate_B": row["candidate_B"],
        "chosen_C": row["chosen_C"],
        "reason_C": row["reason_C"],
        "clean_observed_distance": row["clean_observed_distance"],
        "traj_dist_A": row["traj_dist_A"],
        "traj_dist_B": row["traj_dist_B"],
        "admissible_A": row["admissible_A"],
        "admissible_B": row["admissible_B"],
        "objective_A": row["objective_A"],
        "objective_B": row["objective_B"],
        "fidelity_before": row["fidelity_before"],
        "fidelity_after_A": row["fidelity_after_A"],
        "fidelity_after_B": row["fidelity_after_B"],
        "fidelity_after_C": row["fidelity_after_C"],
        "gain_A": row["gain_A"],
        "gain_B": row["gain_B"],
        "gain_C": row["gain_C"],
        "fid_B_minus_A": float(row["fidelity_after_B"] - row["fidelity_after_A"]),
        "hybrid_objective_gain_B_vs_A": row["hybrid_objective_gain_B_vs_A"],
        "schedule_signature": row["schedule_signature"],
        "schedule": row["schedule"],
        "syndrome_summary": row["syndrome_summary"],
    }


def _case_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "agreement_keep_syndrome",
            "title": "Agreement / Keep Syndrome",
            "description": "A and B are both admissible, hybrid keeps A, and the A/B fidelity gap stays small.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: (
                bool(row["admissible_A"])
                and bool(row["admissible_B"])
                and str(row["chosen_C"]) == "A"
                and abs(float(row["fid_A_minus_B"])) <= float(agreement_tolerance)
            ),
            "sort_key": lambda row: (
                float(row["syndrome_mean_no_error"]),
                -abs(float(row["fid_A_minus_B"])),
                -float(row["clean_observed_distance"]),
            ),
            "reverse": True,
        },
        {
            "name": "syndrome_consistent_trajectory_inconsistent",
            "title": "Syndrome-Consistent Yet Trajectory-Inconsistent",
            "description": "The syndrome looks benign but the observed relational trajectory still leaves the clean path.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: bool(
                row["syndrome_consistent_but_trajectory_inconsistent"]
            ),
            "sort_key": lambda row: (
                float(row["clean_observed_distance"]),
                float(row["hybrid_objective_gain_B_vs_A"]),
            ),
            "reverse": True,
        },
        {
            "name": "relational_fidelity_gain",
            "title": "Relational Fidelity Gain",
            "description": "Relational recovery B beats syndrome recovery A by a configurable fidelity margin.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: (
                float(row["fidelity_after_B"]) > float(row["fidelity_after_A"]) + float(fidelity_margin)
            ),
            "sort_key": lambda row: (
                float(row["fidelity_after_B"] - row["fidelity_after_A"]),
                float(row["clean_observed_distance"]),
            ),
            "reverse": True,
        },
        {
            "name": "hybrid_veto_triggered",
            "title": "Hybrid Veto",
            "description": "Hybrid C discards the syndrome candidate because A is trajectory-inadmissible.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: (
                str(row["reason_C"]) == "veto_nonadmissible_A"
            ),
            "sort_key": lambda row: (
                float(row["hybrid_objective_gain_B_vs_A"]),
                float(row["fidelity_after_B"] - row["fidelity_after_A"]),
            ),
            "reverse": True,
        },
        {
            "name": "hybrid_tie_break_triggered",
            "title": "Hybrid Tie-Break",
            "description": "Hybrid C keeps the syndrome-first philosophy but prefers B when the relational objective is strictly better.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: (
                str(row["reason_C"]) == "tie_break_objective"
            ),
            "sort_key": lambda row: (
                float(row["hybrid_objective_gain_B_vs_A"]),
                float(row["fidelity_after_B"] - row["fidelity_after_A"]),
            ),
            "reverse": True,
        },
        {
            "name": "conservative_hybrid_gap",
            "title": "Conservative Hybrid Gap",
            "description": "Hybrid C keeps A even though B has higher fidelity, exposing the cost of a safety-first rule.",
            "predicate": lambda row, fidelity_margin, agreement_tolerance: (
                str(row["chosen_C"]) == "A"
                and float(row["fidelity_after_B"]) > float(row["fidelity_after_C"]) + float(fidelity_margin)
            ),
            "sort_key": lambda row: (
                float(row["fidelity_after_B"] - row["fidelity_after_C"]),
                float(row["hybrid_objective_gain_B_vs_A"]),
            ),
            "reverse": True,
        },
    ]


def _select_cases(
    rows: list[dict[str, Any]],
    *,
    per_group: int,
    fidelity_margin: float,
    agreement_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    for spec in _case_specs():
        matches = [
            row
            for row in rows
            if spec["predicate"](row, float(fidelity_margin), float(agreement_tolerance))
        ]
        ordered = sorted(matches, key=spec["sort_key"], reverse=bool(spec["reverse"]))
        chosen = [
            _project_case(row, group=spec["name"], title=spec["title"], rank=index + 1)
            for index, row in enumerate(ordered[: int(per_group)])
        ]
        selected_rows.extend(chosen)
        summary = {
            "group": spec["name"],
            "title": spec["title"],
            "matched_cases": len(matches),
            "selected_cases": len(chosen),
            "description": spec["description"],
        }
        group_summary.append(summary)
        groups[spec["name"]] = {
            "title": spec["title"],
            "description": spec["description"],
            "matched_cases": len(matches),
            "selected_cases": chosen,
        }
    return selected_rows, group_summary, groups


def _build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Encoded QEC Representative Cases",
        "",
        "## Source Overall",
        "",
        _markdown_table([result["source_overall"]], [
            "cases",
            "simulation_backend",
            "fid_observed_mean",
            "fid_recovered_A_mean",
            "fid_recovered_B_mean",
            "fid_recovered_C_mean",
            "hybrid_uses_relational_rate",
            "hybrid_veto_rate",
            "hybrid_tie_break_rate",
            "syndrome_consistent_but_trajectory_inconsistent_rate",
        ]),
        "",
        "## Group Summary",
        "",
        _markdown_table(result["tables"]["group_summary"], [
            "group",
            "title",
            "matched_cases",
            "selected_cases",
        ]),
    ]
    for group_name, group in result["groups"].items():
        lines.extend([
            "",
            f"## {group['title']}",
            "",
            group["description"],
            "",
        ])
        if group["selected_cases"]:
            lines.append(
                _markdown_table(group["selected_cases"], [
                    "rank_in_group",
                    "experiment_id",
                    "code_type",
                    "noise_family",
                    "noise_strength",
                    "noise_depth",
                    "syndrome",
                    "chosen_C",
                    "reason_C",
                    "admissible_A",
                    "admissible_B",
                    "fidelity_before",
                    "fidelity_after_A",
                    "fidelity_after_B",
                    "fidelity_after_C",
                    "gain_B",
                    "clean_observed_distance",
                ])
            )
        else:
            lines.append("No cases matched this group.")
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
    per_group: int = 3,
    fidelity_margin: float = 0.01,
    agreement_tolerance: float = 0.01,
    trajectory_inconsistency_threshold: float = 0.05,
    syndrome_consistent_threshold: float = 0.9,
    hybrid_objective_tol: float = 1.0e-9,
    tie_break_requires_syndrome_consistent: bool = True,
    experiment_config: str = "experiment/encoded_qec_baseline.yaml",
) -> dict[str, Any]:
    baseline = run_encoded_qec_baseline(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        n_samples=n_samples,
        seed=seed,
        min_steps=min_steps,
        max_steps=max_steps,
        kinds=kinds,
        p_min=p_min,
        p_max=p_max,
        trajectory_inconsistency_threshold=trajectory_inconsistency_threshold,
        syndrome_consistent_threshold=syndrome_consistent_threshold,
        hybrid_objective_tol=hybrid_objective_tol,
        tie_break_requires_syndrome_consistent=tie_break_requires_syndrome_consistent,
        experiment_config=experiment_config,
    )
    selected_rows, group_summary, groups = _select_cases(
        baseline["rows"],
        per_group=int(per_group),
        fidelity_margin=float(fidelity_margin),
        agreement_tolerance=float(agreement_tolerance),
    )
    result = {
        "selection": {
            "per_group": int(per_group),
            "fidelity_margin": float(fidelity_margin),
            "agreement_tolerance": float(agreement_tolerance),
        },
        "source_grid": baseline["grid"],
        "source_overall": baseline["overall"],
        "tables": {
            "group_summary": group_summary,
            "selected_cases": selected_rows,
        },
        "groups": groups,
        "representative_cases": selected_rows,
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract representative encoded-QEC baseline cases.")
    parser.add_argument("--config", default="experiment/encoded_qec_baseline.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--codes", default=None, help="Comma-separated codes")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--fidelity-margin", type=float, default=0.01)
    parser.add_argument("--agreement-tolerance", type=float, default=0.01)
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
    state_configs = {
        "bitflip": "states/repetition_bitflip.yaml",
        "phaseflip": "states/repetition_phaseflip.yaml",
    }
    state_configs.update({str(k): str(v) for k, v in dict(cfg.get("state_configs", {})).items()})
    kinds_by_code_raw = cfg.get("kinds_by_code")
    kinds_by_code = None
    if isinstance(kinds_by_code_raw, dict):
        kinds_by_code = {
            str(key): [str(item) for item in value]
            for key, value in kinds_by_code_raw.items()
            if isinstance(value, (list, tuple))
        }
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
        per_group=int(args.per_group),
        fidelity_margin=float(args.fidelity_margin),
        agreement_tolerance=float(args.agreement_tolerance),
        trajectory_inconsistency_threshold=float(cfg.get("trajectory_inconsistency_threshold", 0.05)),
        syndrome_consistent_threshold=float(cfg.get("syndrome_consistent_threshold", 0.9)),
        hybrid_objective_tol=float(hybrid_cfg.get("objective_tol", 1.0e-9)),
        tie_break_requires_syndrome_consistent=bool(
            hybrid_cfg.get("tie_break_requires_syndrome_consistent", True)
        ),
        experiment_config=args.config,
    )
    stem = args.output_stem or "encoded_qec_casebook"
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_selected_cases.csv", result["tables"]["selected_cases"])
    _write_csv(tables_dir / f"{stem}_group_summary.csv", result["tables"]["group_summary"])
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["tables"]["group_summary"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

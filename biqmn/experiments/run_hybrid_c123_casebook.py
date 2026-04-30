"""Representative-case extraction for hybrid C1/C2/C3/C3R policy comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv
from .run_hybrid_c123_baseline import (
    C3RPolicyConfig,
    DEFAULT_CODES,
    PolicyScoreConfig,
    _parse_csv,
    run as run_hybrid_c123_baseline,
)


Predicate = Callable[[Dict[str, Any], Dict[str, float]], bool]
SortKey = Callable[[Dict[str, Any]], Any]


def _load_rows(source_stem: str) -> list[dict[str, Any]]:
    path = RESULT_ROOT / "raw" / f"{source_stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source payload: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Payload {path} does not contain a 'rows' list.")
    return [dict(row) for row in rows]


def _project_case(row: dict[str, Any], *, case_type: str, title: str, rank: int) -> dict[str, Any]:
    return {
        "case_type": case_type,
        "title": title,
        "rank_in_group": int(rank),
        "experiment_id": row["experiment_id"],
        "code_family": row["code_family"],
        "noise_family": row["noise_family"],
        "noise_strength": row["noise_strength"],
        "noise_depth": row["noise_depth"],
        "seed": row["seed"],
        "syndrome_label": row["syndrome_label"],
        "candidate_C1": row["candidate_C1"],
        "candidate_C2": row["candidate_C2"],
        "candidate_C3": row["candidate_C3"],
        "candidate_C3R": row.get("candidate_C3R"),
        "decision_reason_C1": row["decision_reason_C1"],
        "decision_reason_C2": row["decision_reason_C2"],
        "decision_reason_C3": row["decision_reason_C3"],
        "decision_reason_C3R": row.get("decision_reason_C3R"),
        "admissible_C1": row["admissible_C1"],
        "admissible_C2": row["admissible_C2"],
        "admissible_C3": row["admissible_C3"],
        "admissible_C3R": row.get("admissible_C3R"),
        "false_safe_flag_C1": row["false_safe_flag_C1"],
        "false_safe_flag_C2": row["false_safe_flag_C2"],
        "false_safe_flag_C3": row["false_safe_flag_C3"],
        "false_safe_flag_C3R": row.get("false_safe_flag_C3R"),
        "fid_gain_C1": row["fid_gain_C1"],
        "fid_gain_C2": row["fid_gain_C2"],
        "fid_gain_C3": row["fid_gain_C3"],
        "fid_gain_C3R": row.get("fid_gain_C3R"),
        "fidelity_after_C1": row["fidelity_after_C1"],
        "fidelity_after_C2": row["fidelity_after_C2"],
        "fidelity_after_C3": row["fidelity_after_C3"],
        "fidelity_after_C3R": row.get("fidelity_after_C3R"),
        "logical_success_C1": row["logical_success_C1"],
        "logical_success_C2": row["logical_success_C2"],
        "logical_success_C3": row["logical_success_C3"],
        "logical_success_C3R": row.get("logical_success_C3R"),
        "score_C2_A": row.get("score_C2_A"),
        "score_C2_B": row.get("score_C2_B"),
        "score_C3_A": row.get("score_C3_A"),
        "score_C3_B": row.get("score_C3_B"),
        "score_C3R_A": row.get("score_C3R_A"),
        "score_C3R_B": row.get("score_C3R_B"),
        "c3r_score_margin": row.get("c3r_score_margin"),
        "c3r_violation_A": row.get("c3r_violation_A"),
        "c3r_violation_B": row.get("c3r_violation_B"),
        "c3r_syndrome_uncertainty": row.get("c3r_syndrome_uncertainty"),
    }


def _case_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "type1_c1_sufficient",
            "title": "Type 1: C1 Sufficient",
            "description": "C1, C2, and C3 behave similarly in both gain and structural safety.",
            "predicate": lambda row, p: (
                abs(float(row["fid_gain_C1"]) - float(row["fid_gain_C2"])) <= float(p["similarity_tol"])
                and abs(float(row["fid_gain_C1"]) - float(row["fid_gain_C3"])) <= float(p["similarity_tol"])
                and bool(row["false_safe_flag_C1"]) == bool(row["false_safe_flag_C2"]) == bool(row["false_safe_flag_C3"])
            ),
            "sort_key": lambda row: (
                -float(row["fid_gain_C1"]),
                abs(float(row["fid_gain_C1"]) - float(row["fid_gain_C2"])),
            ),
        },
        {
            "name": "type2_c2_wins",
            "title": "Type 2: C2 Wins",
            "description": "C2 improves fidelity gain over C1 without introducing additional false-safe risk.",
            "predicate": lambda row, p: (
                float(row["fid_gain_C2"]) > float(row["fid_gain_C1"]) + float(p["gain_margin"])
                and int(bool(row["false_safe_flag_C2"])) <= int(bool(row["false_safe_flag_C1"]))
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_C2"]) - float(row["fid_gain_C1"]),
                -int(bool(row["false_safe_flag_C2"])),
            ),
        },
        {
            "name": "type3_c3_necessary",
            "title": "Type 3: C3 Necessary",
            "description": "C3 is the only hybrid that suppresses structural risk in this regime.",
            "predicate": lambda row, p: (
                bool(row["false_safe_flag_A"])
                and bool(row["false_safe_flag_C1"])
                and bool(row["false_safe_flag_C2"])
                and not bool(row["false_safe_flag_C3"])
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_C3"]),
                -float(row["fid_gain_C2"]),
            ),
        },
        {
            "name": "type4_c2_overaggressive",
            "title": "Type 4: C2 Overaggressive",
            "description": "C2 buys gain but worsens false-safe behavior relative to C1.",
            "predicate": lambda row, p: (
                float(row["fid_gain_C2"]) > float(row["fid_gain_C1"]) + float(p["gain_margin"])
                and bool(row["false_safe_flag_C2"])
                and not bool(row["false_safe_flag_C1"])
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_C2"]) - float(row["fid_gain_C1"]),
                float(row["fid_gain_C2"]),
            ),
        },
        {
            "name": "typeR1_c2_sufficient_c3r_allows",
            "title": "Type R1: C2 Sufficient",
            "description": "C2 selects B, C3R allows B, and the relational candidate is beneficial or non-harmful.",
            "predicate": lambda row, p: (
                str(row.get("candidate_C2")) == "B"
                and str(row.get("candidate_C3R")) == "B"
                and float(row["fid_gain_B"]) >= float(row["fid_gain_A"]) - float(p["gain_margin"])
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_B"]) - float(row["fid_gain_A"]),
                float(row.get("c3r_score_margin") or 0.0),
            ),
        },
        {
            "name": "typeR2_c3r_prevents_harmful_switch",
            "title": "Type R2: C3R Prevents Harmful Switch",
            "description": "C2 selects B, C3R blocks the switch, and B would have lost fidelity relative to A.",
            "predicate": lambda row, p: (
                str(row.get("candidate_C2")) == "B"
                and str(row.get("candidate_C3R")) == "A"
                and float(row["fid_gain_B"]) < float(row["fid_gain_A"]) - float(p["gain_margin"])
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_A"]) - float(row["fid_gain_B"]),
                float(row.get("c3r_syndrome_uncertainty") or 0.0),
            ),
        },
        {
            "name": "typeR3_c3r_overconservative",
            "title": "Type R3: C3R Overconservative",
            "description": "C2 selects B, C3R blocks the switch, but B would have improved fidelity relative to A.",
            "predicate": lambda row, p: (
                str(row.get("candidate_C2")) == "B"
                and str(row.get("candidate_C3R")) == "A"
                and float(row["fid_gain_B"]) > float(row["fid_gain_A"]) + float(p["gain_margin"])
            ),
            "sort_key": lambda row: (
                float(row["fid_gain_B"]) - float(row["fid_gain_A"]),
                float(row.get("c3r_score_margin") or 0.0),
            ),
        },
        {
            "name": "typeR4_c3r_inactive",
            "title": "Type R4: C3R Inactive",
            "description": "C2 already selects A, so the C3R gate has no switch to inspect.",
            "predicate": lambda row, p: (
                str(row.get("candidate_C2")) == "A"
                and str(row.get("candidate_C3R")) == "A"
            ),
            "sort_key": lambda row: (
                -float(row.get("c3r_syndrome_uncertainty") or 0.0),
                float(row.get("c3r_score_margin") or 0.0),
            ),
        },
    ]


def _select_cases(
    rows: Sequence[dict[str, Any]],
    *,
    per_group: int,
    params: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    for spec in _case_specs():
        matches = [dict(row) for row in rows if spec["predicate"](row, params)]
        ordered = sorted(matches, key=spec["sort_key"], reverse=True)
        chosen = [
            _project_case(row, case_type=spec["name"], title=spec["title"], rank=index + 1)
            for index, row in enumerate(ordered[: int(per_group)])
        ]
        selected_rows.extend(chosen)
        summary = {
            "case_type": spec["name"],
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
        "# Hybrid C1/C2/C3/C3R Casebook",
        "",
        "## Group Summary",
        "",
        _markdown_table(result["tables"]["group_summary"], [
            "case_type",
            "title",
            "matched_cases",
            "selected_cases",
        ]),
    ]
    for case_type, group in result["groups"].items():
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
                    "code_family",
                    "noise_family",
                    "noise_strength",
                    "noise_depth",
                    "decision_reason_C1",
                    "decision_reason_C2",
                    "decision_reason_C3",
                    "decision_reason_C3R",
                    "fid_gain_C1",
                    "fid_gain_C2",
                    "fid_gain_C3",
                    "fid_gain_C3R",
                    "false_safe_flag_C1",
                    "false_safe_flag_C2",
                    "false_safe_flag_C3",
                    "false_safe_flag_C3R",
                    "c3r_score_margin",
                    "c3r_syndrome_uncertainty",
                ])
            )
        else:
            lines.append("No cases matched this group.")
    return "\n".join(lines) + "\n"


def run(
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    source_stem: str | None = None,
    codes: Sequence[str] | None = None,
    state_configs: dict[str, str] | None = None,
    kinds_by_code: dict[str, Sequence[str]] | None = None,
    noise_families: Sequence[str] | None = None,
    strengths: Sequence[float] | None = None,
    depths: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    fidelity_margin: float = 0.01,
    logical_success_threshold: float = 0.99,
    c2_cfg: PolicyScoreConfig | None = None,
    c3_cfg: PolicyScoreConfig | None = None,
    c3r_cfg: C3RPolicyConfig | None = None,
    c1_objective_tol: float = 1.0e-9,
    c1_tie_break_requires_syndrome_consistent: bool = True,
    experiment_config: str = "experiment/hybrid_c123_regime_map.yaml",
    per_group: int = 4,
    gain_margin: float = 0.01,
    similarity_tol: float = 0.005,
) -> dict[str, Any]:
    if rows is None:
        if source_stem:
            source_rows = _load_rows(source_stem)
        else:
            if (
                codes is None
                or state_configs is None
                or kinds_by_code is None
                or noise_families is None
                or strengths is None
                or depths is None
                or seeds is None
                or c2_cfg is None
                or c3_cfg is None
            ):
                raise ValueError("Either provide rows/source_stem or a full baseline grid.")
            baseline = run_hybrid_c123_baseline(
                codes=codes,
                state_configs=state_configs,
                kinds_by_code=kinds_by_code,
                noise_families=noise_families,
                strengths=strengths,
                depths=depths,
                seeds=seeds,
                fidelity_margin=fidelity_margin,
                logical_success_threshold=logical_success_threshold,
                c2_cfg=c2_cfg,
                c3_cfg=c3_cfg,
                c3r_cfg=c3r_cfg,
                c1_objective_tol=c1_objective_tol,
                c1_tie_break_requires_syndrome_consistent=c1_tie_break_requires_syndrome_consistent,
                experiment_config=experiment_config,
                output_stem="hybrid_c123_casebook_source",
            )
            source_rows = [dict(row) for row in baseline["rows"]]
    else:
        source_rows = [dict(row) for row in rows]
    params = {
        "gain_margin": float(gain_margin),
        "similarity_tol": float(similarity_tol),
    }
    selected_rows, group_summary, groups = _select_cases(
        source_rows,
        per_group=int(per_group),
        params=params,
    )
    result = {
        "source_stem": source_stem,
        "selection": {
            "per_group": int(per_group),
            "gain_margin": float(gain_margin),
            "similarity_tol": float(similarity_tol),
        },
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
    parser = argparse.ArgumentParser(description="Extract hybrid C1/C2/C3/C3R representative cases.")
    parser.add_argument("--config", default="experiment/hybrid_c123_regime_map.yaml")
    parser.add_argument("--source-stem", default="hybrid_c123_regime_map")
    parser.add_argument("--per-group", type=int, default=None)
    parser.add_argument("--gain-margin", type=float, default=None)
    parser.add_argument("--similarity-tol", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    case_cfg = dict(config.get("hybrid_c123_casebook", {}))
    result = run(
        source_stem=str(args.source_stem or case_cfg.get("source_stem", "hybrid_c123_regime_map")),
        per_group=int(args.per_group or case_cfg.get("per_group", 4)),
        gain_margin=float(args.gain_margin or case_cfg.get("gain_margin", 0.01)),
        similarity_tol=float(args.similarity_tol or case_cfg.get("similarity_tol", 0.005)),
    )
    stem = args.output_stem or "hybrid_c123_casebook"
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

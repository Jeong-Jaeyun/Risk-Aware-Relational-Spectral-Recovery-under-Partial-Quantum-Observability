"""Casebook extractor for the coherent detector/veto branch."""
from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from .coherent_veto_common import enrich_rows, load_result_rows, select_negative_gain_cases
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv


def _project(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "code_family": row["code_type"],
        "noise_family": row["noise_family"],
        "noise_strength": row["noise_strength"],
        "noise_depth": row["noise_depth"],
        "seed": row["seed"],
        "syndrome": row["syndrome"],
        "traj_inconsistency_score": row["traj_inconsistency_score"],
        "flag_structural_risk": row["flag_structural_risk"],
        "fidelity_after_A": row["fidelity_after_A"],
        "fidelity_after_B": row["fidelity_after_B"],
        "fid_gain_A": row["gain_A"],
        "fid_gain_B": row["gain_B"],
        "decision_disagreement_AB": row["decision_disagreement_AB"],
        "false_safe_flag_A": row["false_safe_flag_A"],
        "reason_C": row["reason_C"],
    }


def _agreement_cases(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    agreement_eps: float,
) -> list[dict[str, Any]]:
    subset = [
        dict(row)
        for row in rows
        if str(row["code_type"]) == "bitflip"
        and str(row["noise_family"]) == "coherent_x"
        and abs(float(row["gain_A"]) - float(row["gain_B"])) <= float(agreement_eps)
    ]
    subset.sort(
        key=lambda row: (
            -float(row["traj_inconsistency_score"]),
            float(row["gain_B"]),
            int(row["seed"]),
        )
    )
    return subset[: int(limit)]


def _markdown(title: str, rows: Sequence[dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    if rows:
        lines.append(
            _markdown_table(list(rows), [
                "experiment_id",
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "seed",
                "syndrome",
                "traj_inconsistency_score",
                "flag_structural_risk",
                "fidelity_after_A",
                "fidelity_after_B",
                "fid_gain_A",
                "fid_gain_B",
                "decision_disagreement_AB",
            ])
        )
    else:
        lines.append("No cases matched.")
    return "\n".join(lines) + "\n"


def run(
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    source_stem: str = "encoded_coherent_validation",
    score_field: str = "clean_observed_distance",
    fidelity_margin: float = 0.01,
    negative_gain_eps: float = 0.0,
    flag_threshold_quantile: float = 0.70,
    negative_case_limit: int = 10,
    agreement_case_limit: int = 10,
    agreement_eps: float = 1.0e-9,
) -> dict[str, Any]:
    source_rows = list(rows) if rows is not None else load_result_rows(source_stem)
    enriched = enrich_rows(
        source_rows,
        fidelity_margin=float(fidelity_margin),
        score_field=str(score_field),
        negative_gain_eps=float(negative_gain_eps),
    )
    from .coherent_veto_common import quantile_thresholds

    threshold = quantile_thresholds(enriched, [float(flag_threshold_quantile)])[0][1]
    for row in enriched:
        row["flag_structural_risk"] = bool(float(row["traj_inconsistency_score"]) >= float(threshold))

    negative_cases = [_project(row) for row in select_negative_gain_cases(
        enriched,
        family="coherent_z",
        code="phaseflip",
        limit=int(negative_case_limit),
    )]
    agreement_cases = [_project(row) for row in _agreement_cases(
        enriched,
        limit=int(agreement_case_limit),
        agreement_eps=float(agreement_eps),
    )]
    result = {
        "source_stem": source_stem,
        "selection": {
            "score_field": str(score_field),
            "fidelity_margin": float(fidelity_margin),
            "negative_gain_eps": float(negative_gain_eps),
            "flag_threshold_quantile": float(flag_threshold_quantile),
            "flag_threshold": float(threshold),
            "negative_case_limit": int(negative_case_limit),
            "agreement_case_limit": int(agreement_case_limit),
            "agreement_eps": float(agreement_eps),
        },
        "negative_gain_cases": negative_cases,
        "agreement_cases": agreement_cases,
        "negative_markdown": _markdown("Coherent Casebook Negative Gain", negative_cases),
        "agreement_markdown": _markdown("Coherent Casebook Agreement", agreement_cases),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract coherent branch casebooks.")
    parser.add_argument("--config", default="experiment/coherent_branch.yaml")
    parser.add_argument("--source-stem", default=None)
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--fidelity-margin", type=float, default=None)
    parser.add_argument("--negative-gain-eps", type=float, default=None)
    parser.add_argument("--flag-threshold-quantile", type=float, default=None)
    parser.add_argument("--negative-case-limit", type=int, default=None)
    parser.add_argument("--agreement-case-limit", type=int, default=None)
    parser.add_argument("--agreement-eps", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    case_cfg = dict(config.get("coherent_casebook", {}))
    result = run(
        source_stem=str(args.source_stem or case_cfg.get("source_stem", "encoded_coherent_validation")),
        score_field=str(args.score_field or case_cfg.get("score_field", "clean_observed_distance")),
        fidelity_margin=float(args.fidelity_margin or case_cfg.get("fidelity_margin", 0.01)),
        negative_gain_eps=float(args.negative_gain_eps or case_cfg.get("negative_gain_eps", 0.0)),
        flag_threshold_quantile=float(
            args.flag_threshold_quantile or config.get("coherent_v1", {}).get("flag_threshold_quantile", 0.70)
        ),
        negative_case_limit=int(args.negative_case_limit or case_cfg.get("negative_case_limit", 10)),
        agreement_case_limit=int(args.agreement_case_limit or case_cfg.get("agreement_case_limit", 10)),
        agreement_eps=float(args.agreement_eps or case_cfg.get("agreement_eps", 1.0e-9)),
    )
    stem = args.output_stem or "coherent_casebook"
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_negative_gain.csv", result["negative_gain_cases"])
    _write_csv(tables_dir / f"{stem}_agreement.csv", result["agreement_cases"])
    negative_md_path = tables_dir / "coherent_casebook_negative_gain.md"
    negative_md_path.write_text(result["negative_markdown"], encoding="utf-8")
    agreement_md_path = tables_dir / "coherent_casebook_agreement.md"
    agreement_md_path.write_text(result["agreement_markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["selection"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_negative_markdown={negative_md_path}")
    print(f"saved_agreement_markdown={agreement_md_path}")


if __name__ == "__main__":
    main()

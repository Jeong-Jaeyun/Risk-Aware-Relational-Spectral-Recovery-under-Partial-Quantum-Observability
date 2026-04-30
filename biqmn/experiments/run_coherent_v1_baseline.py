"""Coherent V1 baseline: syndrome A plus structural-risk flag only.

This runner fixes the coherent branch as a detector / veto branch rather than a
recovery selector:

* final recovery remains A
* B is computed for comparison only
* the relational trajectory score is converted into `flag_structural_risk`
"""
from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any, Sequence

from .coherent_veto_common import enrich_rows, load_result_rows, quantile_thresholds, summarize_rows
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_coherent_veto_analysis import PAIR_ORDER, _save_score_vs_gain_figure, _subset
from .run_encoded_qec_baseline import _markdown_table, _write_csv


def _threshold_from_quantile(rows: Sequence[dict[str, Any]], quantile: float) -> float:
    thresholds = quantile_thresholds(rows, [float(quantile)])
    if not thresholds:
        raise ValueError("Cannot derive a V1 threshold from an empty row set.")
    return float(thresholds[0][1])


def _project_row(row: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "code_family": row["code_type"],
        "noise_family": row["noise_family"],
        "noise_strength": row["noise_strength"],
        "noise_depth": row["noise_depth"],
        "seed": row["seed"],
        "syndrome": row["syndrome"],
        "fid_gain_A": row["gain_A"],
        "fid_gain_B": row["gain_B"],
        "traj_inconsistency_score": row["traj_inconsistency_score"],
        "flag_structural_risk": bool(float(row["traj_inconsistency_score"]) >= float(threshold)),
        "false_safe_flag_A": row["false_safe_flag_A"],
        "decision_disagreement_AB": row["decision_disagreement_AB"],
        "admissible_A": row["admissible_A"],
        "admissible_B": row["admissible_B"],
        "fidelity_after_A": row["fidelity_after_A"],
        "fidelity_after_B": row["fidelity_after_B"],
        "reason_C": row["reason_C"],
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "flag_rate": float(sum(1.0 for row in rows if bool(row["flag_structural_risk"])) / len(rows)),
        "false_safe_rate_A": float(sum(1.0 for row in rows if bool(row["false_safe_flag_A"])) / len(rows)),
        "decision_disagreement_rate_AB": float(
            sum(1.0 for row in rows if bool(row["decision_disagreement_AB"])) / len(rows)
        ),
        "traj_inconsistency_score_mean": float(mean(float(row["traj_inconsistency_score"]) for row in rows)),
        "fid_gain_A_mean": float(mean(float(row["fid_gain_A"]) for row in rows)),
        "fid_gain_B_mean": float(mean(float(row["fid_gain_B"]) for row in rows)),
    }


def _group_summary(rows: Sequence[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(dict(row))
    out: list[dict[str, Any]] = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_rows(grouped[group_key]))
        out.append(entry)
    return out


def _build_markdown(result: dict[str, Any]) -> str:
    return "\n".join([
        "# Coherent V1 Summary",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "flag_threshold_quantile",
            "flag_threshold",
            "flag_rate",
            "false_safe_rate_A",
            "decision_disagreement_rate_AB",
            "traj_inconsistency_score_mean",
            "fid_gain_A_mean",
            "fid_gain_B_mean",
        ]),
        "",
        "## By Pair",
        "",
        _markdown_table(result["tables"]["by_pair"], [
            "code_family",
            "noise_family",
            "cases",
            "flag_rate",
            "false_safe_rate_A",
            "decision_disagreement_rate_AB",
            "traj_inconsistency_score_mean",
            "fid_gain_A_mean",
            "fid_gain_B_mean",
        ]),
        "",
        "## By Noise Strength",
        "",
        _markdown_table(result["tables"]["by_noise_strength"], [
            "noise_strength",
            "cases",
            "flag_rate",
            "false_safe_rate_A",
            "traj_inconsistency_score_mean",
            "fid_gain_A_mean",
            "fid_gain_B_mean",
        ]),
        "",
        "## Figure",
        "",
        f"- `figure_a_score_vs_gain_B`: `{result['figures']['figure_a_score_vs_gain_B']}`",
        "",
    ]) + "\n"


def run(
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    source_stem: str = "encoded_coherent_validation",
    score_field: str = "clean_observed_distance",
    fidelity_margin: float = 0.01,
    negative_gain_eps: float = 0.0,
    flag_threshold_quantile: float = 0.70,
    plot_stem: str = "coherent_v1",
) -> dict[str, Any]:
    source_rows = list(rows) if rows is not None else load_result_rows(source_stem)
    enriched = enrich_rows(
        source_rows,
        fidelity_margin=float(fidelity_margin),
        score_field=str(score_field),
        negative_gain_eps=float(negative_gain_eps),
    )
    threshold = _threshold_from_quantile(enriched, float(flag_threshold_quantile))
    projected_rows = [_project_row(row, threshold=threshold) for row in enriched]
    overall = _aggregate_rows(projected_rows)
    overall["flag_threshold_quantile"] = float(flag_threshold_quantile)
    overall["flag_threshold"] = float(threshold)
    by_pair = []
    for code_type, noise_family in PAIR_ORDER:
        subset = [
            row for row in projected_rows
            if str(row["code_family"]) == str(code_type) and str(row["noise_family"]) == str(noise_family)
        ]
        if subset:
            summary = _aggregate_rows(subset)
            summary["code_family"] = str(code_type)
            summary["noise_family"] = str(noise_family)
            by_pair.append(summary)
    by_noise_strength = _group_summary(projected_rows, keys=("noise_strength",))
    figure_path = _save_score_vs_gain_figure(enriched, plot_stem)
    result = {
        "source_stem": source_stem,
        "selection": {
            "score_field": str(score_field),
            "fidelity_margin": float(fidelity_margin),
            "negative_gain_eps": float(negative_gain_eps),
            "flag_threshold_quantile": float(flag_threshold_quantile),
            "flag_threshold": float(threshold),
        },
        "overall": overall,
        "rows": projected_rows,
        "tables": {
            "by_pair": by_pair,
            "by_noise_strength": by_noise_strength,
        },
        "figures": {
            "figure_a_score_vs_gain_B": str(figure_path),
        },
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export coherent V1 baseline rows and summary.")
    parser.add_argument("--config", default="experiment/coherent_branch.yaml")
    parser.add_argument("--source-stem", default=None)
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--fidelity-margin", type=float, default=None)
    parser.add_argument("--negative-gain-eps", type=float, default=None)
    parser.add_argument("--flag-threshold-quantile", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    branch_cfg = dict(config.get("coherent_v1", {}))
    stem = args.output_stem or "coherent_v1"
    result = run(
        source_stem=str(args.source_stem or branch_cfg.get("source_stem", "encoded_coherent_validation")),
        score_field=str(args.score_field or branch_cfg.get("score_field", "clean_observed_distance")),
        fidelity_margin=float(args.fidelity_margin or branch_cfg.get("fidelity_margin", 0.01)),
        negative_gain_eps=float(args.negative_gain_eps or branch_cfg.get("negative_gain_eps", 0.0)),
        flag_threshold_quantile=float(
            args.flag_threshold_quantile or branch_cfg.get("flag_threshold_quantile", 0.70)
        ),
        plot_stem=stem,
    )
    json_path = write_json_result(result, f"{stem}_summary")
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / "coherent_v1_analysis_raw.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / "coherent_v1_summary.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .run_recovery_sweep import (
    DEFAULT_BANK_WIDTHS_DEG,
    DEFAULT_NOISE_KINDS,
    DEFAULT_STRENGTHS,
    run as run_recovery_sweep,
)


DEFAULT_REF_ANCHOR_WEIGHTS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _parse_csv_list(raw: str | None, caster) -> list[Any] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []
    return [caster(item) for item in items]


def _with_ref_anchor_weight(base_config: dict[str, Any], ref_anchor_weight: float) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    recovery_cfg = cfg.setdefault("recovery", {})
    weights = dict(recovery_cfg.get("weights", recovery_cfg.get("betas", {})))
    weights["ref_anchor"] = float(ref_anchor_weight)
    weights["phi_ref"] = 0.0
    recovery_cfg["weights"] = weights
    stage2_cfg = recovery_cfg.setdefault("stage2", {})
    stage2_cfg["apply_rule"] = "diagnostic_only"
    return cfg


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    pairs = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if len(pairs) < 2:
        return None
    xs = np.asarray([pair[0] for pair in pairs], dtype=float)
    ys = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _mode_stat(values: Sequence[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    label, count = counts.most_common(1)[0]
    return label, float(count / len(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "mean_stage1_clean_distance": None,
            "mean_clean_to_reference_anchor_distance": None,
            "stage2_helpful_tradeoff_rate": 0.0,
            "stage2_candidate_improvement_rate": 0.0,
            "mean_stage2_candidate_objective_gain_vs_stage1": None,
            "corr_gain_vs_clean_delta": None,
            "final_recovery_mode_label": None,
            "final_recovery_stability": 0.0,
            "stage1_anchor_match_rate": 0.0,
            "recovered_admissible_rate": 0.0,
        }

    helpful_tradeoff = [
        row
        for row in rows
        if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
        and float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1e-9
        and float(row["stage2_candidate_clean_distance_delta_vs_stage1"]) <= 1e-9
    ]
    candidate_improves = [
        row
        for row in rows
        if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
        and float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1e-9
    ]
    mode_label, mode_rate = _mode_stat([str(row["stage1_best_label"]) for row in rows])
    anchor_match_rate = float(
        sum(
            1.0 if str(row["stage1_best_label"]) == str(row["reference_anchor_label"]) else 0.0
            for row in rows
        ) / len(rows)
    )
    return {
        "cases": len(rows),
        "mean_stage1_clean_distance": _mean(
            [float(row["clean_to_stage1_distance"]) for row in rows]
        ),
        "mean_clean_to_reference_anchor_distance": _mean(
            [float(row["clean_to_reference_anchor_distance"]) for row in rows]
        ),
        "stage2_helpful_tradeoff_rate": float(len(helpful_tradeoff) / len(rows)),
        "stage2_candidate_improvement_rate": float(len(candidate_improves) / len(rows)),
        "mean_stage2_candidate_objective_gain_vs_stage1": _mean(
            [
                float(row["stage2_candidate_objective_gain_vs_stage1"])
                for row in rows
                if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
            ]
        ),
        "corr_gain_vs_clean_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_clean_distance_delta_vs_stage1"
        ),
        "final_recovery_mode_label": mode_label,
        "final_recovery_stability": mode_rate,
        "stage1_anchor_match_rate": anchor_match_rate,
        "recovered_admissible_rate": float(
            sum(1.0 if bool(row["recovered_admissible"]) else 0.0 for row in rows) / len(rows)
        ),
    }


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Ref-Anchor Validation",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "mean_stage1_clean_distance",
            "stage2_helpful_tradeoff_rate",
            "corr_gain_vs_clean_delta",
            "final_recovery_mode_label",
            "final_recovery_stability",
            "stage1_anchor_match_rate",
        ]),
        "",
        "## By Ref-Anchor Weight And Bank Width",
        "",
        _markdown_table(result["tables"]["by_ref_anchor_weight_and_bank_width"], [
            "ref_anchor_weight",
            "bank_width_deg",
            "cases",
            "mean_stage1_clean_distance",
            "stage2_helpful_tradeoff_rate",
            "corr_gain_vs_clean_delta",
            "final_recovery_stability",
            "stage1_anchor_match_rate",
        ]),
        "",
        "## By Ref-Anchor Weight",
        "",
        _markdown_table(result["tables"]["by_ref_anchor_weight"], [
            "ref_anchor_weight",
            "cases",
            "mean_stage1_clean_distance",
            "stage2_helpful_tradeoff_rate",
            "corr_gain_vs_clean_delta",
            "final_recovery_stability",
            "stage1_anchor_match_rate",
        ]),
        "",
        "## By Bank Width",
        "",
        _markdown_table(result["tables"]["by_bank_width_deg"], [
            "bank_width_deg",
            "cases",
            "mean_stage1_clean_distance",
            "stage2_helpful_tradeoff_rate",
            "corr_gain_vs_clean_delta",
            "final_recovery_stability",
            "stage1_anchor_match_rate",
        ]),
    ]
    return "\n".join(lines) + "\n"


def run(
    base_config: dict[str, Any],
    *,
    ref_anchor_weights: Iterable[float] = DEFAULT_REF_ANCHOR_WEIGHTS,
    bank_widths_deg: Iterable[float] = DEFAULT_BANK_WIDTHS_DEG,
    noise_kinds: Iterable[str] = DEFAULT_NOISE_KINDS,
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
    kappas: Iterable[float] | None = None,
) -> dict[str, Any]:
    if kappas is None:
        kappas = [float(base_config.get("admissibility", {}).get("kappa", 1.5))]

    scenario_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for ref_anchor_weight in ref_anchor_weights:
        weight_config = _with_ref_anchor_weight(base_config, float(ref_anchor_weight))
        for bank_width_deg in bank_widths_deg:
            sweep = run_recovery_sweep(
                weight_config,
                noise_kinds=noise_kinds,
                strengths=strengths,
                kappas=kappas,
                bank_widths_deg=[float(bank_width_deg)],
            )
            cell_scenarios = []
            for row in sweep["rows"]:
                item = {
                    **row,
                    "ref_anchor_weight": float(ref_anchor_weight),
                }
                scenario_rows.append(item)
                cell_scenarios.append(item)
            cell = {
                "ref_anchor_weight": float(ref_anchor_weight),
                "bank_width_deg": float(bank_width_deg),
                **_aggregate_rows(cell_scenarios),
            }
            cell_rows.append(cell)

    tables = {
        "by_ref_anchor_weight_and_bank_width": cell_rows,
        "by_ref_anchor_weight": _group_rows(scenario_rows, keys=("ref_anchor_weight",)),
        "by_bank_width_deg": _group_rows(scenario_rows, keys=("bank_width_deg",)),
    }
    result = {
        "grid": {
            "ref_anchor_weights": [float(item) for item in ref_anchor_weights],
            "bank_widths_deg": [float(item) for item in bank_widths_deg],
            "noise_kinds": [str(item) for item in noise_kinds],
            "strengths": [float(item) for item in strengths],
            "kappas": [float(item) for item in kappas],
        },
        "overall": _aggregate_rows(scenario_rows),
        "tables": tables,
        "rows": scenario_rows,
        "cells": cell_rows,
    }
    result["markdown"] = _build_markdown_report(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the new_ref_anchor objective over ref-anchor weight x bank-width sweeps."
    )
    parser.add_argument("--config", default="experiment/recovery_eval.yaml")
    parser.add_argument("--state-config", default="states/null_dynamic.yaml")
    parser.add_argument("--ref-anchor-weights", default="4,8,16,32,64,128")
    parser.add_argument("--bank-widths-deg", default="5,10,20")
    parser.add_argument("--noise-kinds", default="bitflip,phaseflip,dephasing")
    parser.add_argument("--strengths", default="0.02,0.05,0.08,0.12,0.16")
    parser.add_argument("--kappas", default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(
        experiment_config=args.config,
        state_config=args.state_config,
    )
    result = run(
        base_config,
        ref_anchor_weights=_parse_csv_list(args.ref_anchor_weights, float) or list(DEFAULT_REF_ANCHOR_WEIGHTS),
        bank_widths_deg=_parse_csv_list(args.bank_widths_deg, float) or list(DEFAULT_BANK_WIDTHS_DEG),
        noise_kinds=_parse_csv_list(args.noise_kinds, str) or list(DEFAULT_NOISE_KINDS),
        strengths=_parse_csv_list(args.strengths, float) or list(DEFAULT_STRENGTHS),
        kappas=_parse_csv_list(args.kappas, float),
    )
    stem = resolve_output_stem(base_config, "ref_anchor_validation", args.output_stem)
    json_path = write_json_result(result, stem)

    tables_dir = RESULT_ROOT / "tables"
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")

    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

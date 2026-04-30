from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .run_recovery_sweep import (
    DEFAULT_BANK_WIDTHS_DEG,
    DEFAULT_KAPPAS,
    DEFAULT_NOISE_KINDS,
    DEFAULT_STRENGTHS,
    run as run_recovery_sweep,
)


def _parse_csv_list(raw: str | None, caster) -> list[Any] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []
    return [caster(item) for item in items]


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


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "stage2_applied_rate": 0.0,
            "stage2_candidate_improvement_rate": 0.0,
            "recovered_admissible_rate": 0.0,
            "mean_stage1_objective": None,
            "mean_final_objective": None,
            "mean_stage2_candidate_objective_gain_vs_stage1": None,
            "mean_stage2_applied_objective_gain_vs_stage1": None,
            "mean_clean_to_stage1_distance": None,
            "mean_clean_to_final_distance": None,
            "mean_stage2_clean_distance_delta_vs_stage1": None,
            "stage2_clean_nonworsen_rate": 0.0,
            "stage2_helpful_tradeoff_rate": 0.0,
            "corr_gain_vs_clean_delta": None,
            "corr_gain_vs_obs_fit_delta": None,
            "corr_gain_vs_ref_anchor_delta": None,
            "corr_gain_vs_phi_ref_delta": None,
            "corr_gain_vs_clock_delta": None,
            "corr_gain_vs_smooth_delta": None,
        }

    stage2_applied = [row for row in rows if bool(row["stage2_applied"])]
    candidate_improves = [
        row
        for row in rows
        if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
        and float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1e-9
    ]
    clean_nonworsen = [
        row for row in rows if float(row["stage2_clean_distance_delta_vs_stage1"]) <= 1e-9
    ]
    helpful_tradeoff = [
        row
        for row in rows
        if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
        and float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1e-9
        and float(row["stage2_clean_distance_delta_vs_stage1"]) <= 1e-9
    ]
    candidate_gains = [
        float(row["stage2_candidate_objective_gain_vs_stage1"])
        for row in rows
        if row.get("stage2_candidate_objective_gain_vs_stage1") is not None
    ]
    return {
        "cases": len(rows),
        "stage2_applied_rate": float(len(stage2_applied) / len(rows)),
        "stage2_candidate_improvement_rate": float(len(candidate_improves) / len(rows)),
        "recovered_admissible_rate": float(
            sum(1.0 if bool(row["recovered_admissible"]) else 0.0 for row in rows) / len(rows)
        ),
        "mean_stage1_objective": _mean([float(row["stage1_objective"]) for row in rows]),
        "mean_final_objective": _mean([float(row["recovered_objective"]) for row in rows]),
        "mean_stage2_candidate_objective_gain_vs_stage1": _mean(candidate_gains),
        "mean_stage2_applied_objective_gain_vs_stage1": _mean(
            [float(row["stage2_applied_objective_gain_vs_stage1"]) for row in rows]
        ),
        "mean_clean_to_stage1_distance": _mean(
            [float(row["clean_to_stage1_distance"]) for row in rows]
        ),
        "mean_clean_to_final_distance": _mean(
            [float(row["clean_to_recovered_distance"]) for row in rows]
        ),
        "mean_stage2_clean_distance_delta_vs_stage1": _mean(
            [float(row["stage2_clean_distance_delta_vs_stage1"]) for row in rows]
        ),
        "stage2_clean_nonworsen_rate": float(len(clean_nonworsen) / len(rows)),
        "stage2_helpful_tradeoff_rate": float(len(helpful_tradeoff) / len(rows)),
        "corr_gain_vs_clean_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_clean_distance_delta_vs_stage1"
        ),
        "corr_gain_vs_obs_fit_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_obs_fit_delta"
        ),
        "corr_gain_vs_ref_anchor_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_ref_anchor_delta"
        ),
        "corr_gain_vs_phi_ref_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_phi_ref_delta"
        ),
        "corr_gain_vs_clock_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_clock_delta"
        ),
        "corr_gain_vs_smooth_delta": _correlation(
            rows, "stage2_candidate_objective_gain_vs_stage1", "stage2_candidate_smooth_delta"
        ),
    }


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    table = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_aggregate_rows(grouped[group_key]))
        table.append(entry)
    return table


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Recovery Ablation",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "stage2_applied_rate",
            "stage2_candidate_improvement_rate",
            "recovered_admissible_rate",
            "mean_stage2_candidate_objective_gain_vs_stage1",
            "mean_stage2_clean_distance_delta_vs_stage1",
            "stage2_helpful_tradeoff_rate",
            "corr_gain_vs_clean_delta",
            "corr_gain_vs_obs_fit_delta",
            "corr_gain_vs_ref_anchor_delta",
            "corr_gain_vs_phi_ref_delta",
            "corr_gain_vs_clock_delta",
            "corr_gain_vs_smooth_delta",
        ]),
        "",
        "## By Noise Kind",
        "",
        _markdown_table(result["tables"]["by_noise_kind"], [
            "noise_kind",
            "cases",
            "stage2_applied_rate",
            "stage2_candidate_improvement_rate",
            "mean_stage2_candidate_objective_gain_vs_stage1",
            "mean_stage2_clean_distance_delta_vs_stage1",
            "stage2_helpful_tradeoff_rate",
        ]),
        "",
        "## By Admissibility Kappa",
        "",
        _markdown_table(result["tables"]["by_admissibility_kappa"], [
            "admissibility_kappa",
            "cases",
            "stage2_applied_rate",
            "stage2_candidate_improvement_rate",
            "mean_stage2_candidate_objective_gain_vs_stage1",
            "stage2_helpful_tradeoff_rate",
        ]),
        "",
        "## By Bank Width",
        "",
        _markdown_table(result["tables"]["by_bank_width_deg"], [
            "bank_width_deg",
            "cases",
            "stage2_applied_rate",
            "stage2_candidate_improvement_rate",
            "mean_stage2_candidate_objective_gain_vs_stage1",
            "stage2_helpful_tradeoff_rate",
        ]),
        "",
        "## Taxonomy By Noise Kind",
        "",
        _markdown_table(result["tables"]["taxonomy_by_noise_kind"], [
            "noise_kind",
            "stage2_candidate_taxonomy",
            "cases",
            "rate",
        ]),
        "",
        "## Dominant Weighted Term By Noise Kind",
        "",
        _markdown_table(result["tables"]["dominant_term_by_noise_kind"], [
            "noise_kind",
            "stage2_candidate_dominant_weighted_term",
            "cases",
            "rate",
        ]),
        "",
        "## Stage-2 Location By Noise Kind",
        "",
        _markdown_table(result["tables"]["location_by_noise_kind"], [
            "noise_kind",
            "stage2_candidate_location",
            "cases",
            "rate",
        ]),
        "",
        "## By Stage-2 Location",
        "",
        _markdown_table(result["tables"]["by_stage2_candidate_location"], [
            "stage2_candidate_location",
            "cases",
            "stage2_applied_rate",
            "stage2_candidate_improvement_rate",
            "mean_stage2_candidate_objective_gain_vs_stage1",
            "mean_stage2_clean_distance_delta_vs_stage1",
            "stage2_helpful_tradeoff_rate",
        ]),
    ]
    objective_variant_rows = result["tables"].get("by_objective_variant")
    if objective_variant_rows:
        lines.extend([
            "",
            "## By Objective Variant",
            "",
            _markdown_table(objective_variant_rows, [
                "objective_variant",
                "cases",
                "stage2_candidate_improvement_rate",
                "mean_stage2_candidate_objective_gain_vs_stage1",
                "mean_stage2_clean_distance_delta_vs_stage1",
                "stage2_helpful_tradeoff_rate",
            ]),
        ])
    return "\n".join(lines) + "\n"


def _distribution_table(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    totals: dict[Any, int] = defaultdict(int)
    for row in rows:
        group_value = row.get(group_key)
        value = row.get(value_key)
        if value is None:
            continue
        grouped[(group_value, value)].append(row)
        totals[group_value] += 1

    table = []
    for (group_value, value), subset in sorted(grouped.items()):
        total = totals[group_value]
        table.append({
            group_key: group_value,
            value_key: value,
            "cases": len(subset),
            "rate": float(len(subset) / total) if total else 0.0,
        })
    return table


def _apply_objective_variant(base_config: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    recovery_cfg = cfg.setdefault("recovery", {})
    weights = dict(recovery_cfg.get("weights", recovery_cfg.get("betas", {})))
    obs_fit = float(weights.get("obs_fit", weights.get("data", 1.0)))
    anchor_weight = float(weights.get("ref_anchor", weights.get("anchor", 0.0)))
    phi_ref_weight = float(weights.get("phi_ref", weights.get("ref", 0.0)))
    legacy_ref_weight = phi_ref_weight if phi_ref_weight > 0.0 else anchor_weight

    if variant == "old_base":
        new_weights = {
            "obs_fit": obs_fit,
            "ref_anchor": 0.0,
            "lap": float(weights.get("lap", 0.0)),
            "smooth": float(weights.get("smooth", 0.0)),
            "clock": float(weights.get("clock", 0.0)),
            "phi_ref": 0.0,
        }
    elif variant == "old_plus_phi_ref":
        new_weights = {
            "obs_fit": obs_fit,
            "ref_anchor": 0.0,
            "lap": float(weights.get("lap", 0.0)),
            "smooth": float(weights.get("smooth", 0.0)),
            "clock": float(weights.get("clock", 0.0)),
            "phi_ref": legacy_ref_weight,
        }
    elif variant == "new_ref_anchor":
        new_weights = {
            "obs_fit": obs_fit,
            "ref_anchor": anchor_weight,
            "lap": float(weights.get("lap", 0.0)),
            "smooth": float(weights.get("smooth", 0.0)),
            "clock": float(weights.get("clock", 0.0)),
            "phi_ref": 0.0,
        }
    else:
        raise ValueError(
            "objective variant must be one of {'old_base', 'old_plus_phi_ref', 'new_ref_anchor'}"
        )
    recovery_cfg["weights"] = new_weights
    return cfg


def run(
    base_config: dict[str, Any],
    *,
    noise_kinds: Iterable[str] = DEFAULT_NOISE_KINDS,
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
    kappas: Iterable[float] = DEFAULT_KAPPAS,
    bank_widths_deg: Iterable[float] = DEFAULT_BANK_WIDTHS_DEG,
    objective_variants: Iterable[str] | None = None,
) -> dict[str, Any]:
    sweep = run_recovery_sweep(
        base_config,
        noise_kinds=noise_kinds,
        strengths=strengths,
        kappas=kappas,
        bank_widths_deg=bank_widths_deg,
    )
    rows = sweep["rows"]
    tables = {
        "by_noise_kind": _group_rows(rows, keys=("noise_kind",)),
        "by_admissibility_kappa": _group_rows(rows, keys=("admissibility_kappa",)),
        "by_bank_width_deg": _group_rows(rows, keys=("bank_width_deg",)),
        "by_noise_kind_and_strength": _group_rows(rows, keys=("noise_kind", "noise_strength")),
        "taxonomy_by_noise_kind": _distribution_table(
            rows,
            group_key="noise_kind",
            value_key="stage2_candidate_taxonomy",
        ),
        "dominant_term_by_noise_kind": _distribution_table(
            rows,
            group_key="noise_kind",
            value_key="stage2_candidate_dominant_weighted_term",
        ),
        "location_by_noise_kind": _distribution_table(
            rows,
            group_key="noise_kind",
            value_key="stage2_candidate_location",
        ),
        "by_stage2_candidate_location": _group_rows(rows, keys=("stage2_candidate_location",)),
    }
    result = {
        "grid": sweep["grid"],
        "overall": _aggregate_rows(rows),
        "tables": tables,
        "rows": rows,
    }
    if objective_variants is not None:
        objective_variant_rows = []
        for variant in objective_variants:
            variant_result = run_recovery_sweep(
                _apply_objective_variant(base_config, str(variant)),
                noise_kinds=noise_kinds,
                strengths=strengths,
                kappas=kappas,
                bank_widths_deg=bank_widths_deg,
            )
            objective_variant_rows.append({
                "objective_variant": str(variant),
                **_aggregate_rows(variant_result["rows"]),
            })
        result["tables"]["by_objective_variant"] = objective_variant_rows
    result["markdown"] = _build_markdown_report(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export paper-facing Stage-1 vs Stage-2 recovery ablation tables."
    )
    parser.add_argument("--config", default="experiment/recovery_eval.yaml")
    parser.add_argument("--state-config", default="states/null_dynamic.yaml")
    parser.add_argument("--noise-kinds", default="bitflip,phaseflip,dephasing")
    parser.add_argument("--strengths", default="0.02,0.05,0.08,0.12,0.16")
    parser.add_argument("--kappas", default="1.0,1.5,2.0")
    parser.add_argument("--bank-widths-deg", default="5,10,20")
    parser.add_argument("--objective-variants", default=None)
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
        noise_kinds=_parse_csv_list(args.noise_kinds, str) or list(DEFAULT_NOISE_KINDS),
        strengths=_parse_csv_list(args.strengths, float) or list(DEFAULT_STRENGTHS),
        kappas=_parse_csv_list(args.kappas, float) or list(DEFAULT_KAPPAS),
        bank_widths_deg=_parse_csv_list(args.bank_widths_deg, float) or list(DEFAULT_BANK_WIDTHS_DEG),
        objective_variants=_parse_csv_list(args.objective_variants, str),
    )
    stem = resolve_output_stem(base_config, "recovery_ablation", args.output_stem)
    json_path = write_json_result(result, stem)

    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_scenarios.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")

    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

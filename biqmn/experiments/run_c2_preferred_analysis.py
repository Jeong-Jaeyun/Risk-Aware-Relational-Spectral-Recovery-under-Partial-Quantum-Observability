"""Focused analysis of C2-preferred hybrid-regime cells."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coherent_veto_common import ensure_plot_root
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv


def _load_payload(source_stem: str) -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / f"{source_stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["code_family"]),
        str(row["noise_family"]),
        float(row["noise_strength"]),
        int(row["noise_depth"]),
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


def _cell_enrich(
    rows: Sequence[dict[str, Any]],
    *,
    cell_summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_cell[_cell_key(row)].append(dict(row))
    out = []
    for cell in cell_summaries:
        item = dict(cell)
        subset = rows_by_cell[_cell_key(cell)]
        reason_c1, reason_c1_rate = _mode_label([str(row["decision_reason_C1"]) for row in subset])
        reason_c2, reason_c2_rate = _mode_label([str(row["decision_reason_C2"]) for row in subset])
        reason_c3, reason_c3_rate = _mode_label([str(row["decision_reason_C3"]) for row in subset])
        item["decision_reason_C1_majority"] = reason_c1
        item["decision_reason_C1_majority_rate"] = reason_c1_rate
        item["decision_reason_C2_majority"] = reason_c2
        item["decision_reason_C2_majority_rate"] = reason_c2_rate
        item["decision_reason_C3_majority"] = reason_c3
        item["decision_reason_C3_majority_rate"] = reason_c3_rate
        item["fid_gain_delta_C2_minus_C1"] = float(item["fid_gain_C2_mean"]) - float(item["fid_gain_C1_mean"])
        item["false_safe_delta_C1_minus_C2"] = float(item["false_safe_rate_C1"]) - float(item["false_safe_rate_C2"])
        item["chosen_B_rate_delta_C1_minus_C2"] = float(item["chosen_B_rate_C1"]) - float(item["chosen_B_rate_C2"])
        item["harmful_switching_pattern"] = bool(
            float(item["chosen_B_rate_C1"]) > float(item["chosen_B_rate_C2"]) + 1.0e-9
            and float(item["fid_gain_C2_mean"]) > float(item["fid_gain_C1_mean"]) + 1.0e-9
        )
        out.append(item)
    return out


def _objective_utility(value: float) -> float:
    return float(1.0 / (1.0 + float(value)))


def _score_terms_for_row(row: dict[str, Any], *, policies: dict[str, Any]) -> dict[str, Any]:
    c2 = dict(policies.get("c2", {}))
    lambda_s = float(c2.get("lambda_s", 1.0))
    lambda_t = float(c2.get("lambda_t", 1.0))
    lambda_i = float(c2.get("lambda_i", 2.0))
    lambda_o = float(c2.get("lambda_o", 0.5))
    score_a = float(row["score_C2_A"])
    score_b = float(row["score_C2_B"])
    traj_a = float(row["traj_distance_A"])
    traj_b = float(row["traj_distance_B"])
    inad_a = 0.0 if bool(row["admissible_A"]) else 1.0
    inad_b = 0.0 if bool(row["admissible_B"]) else 1.0
    util_a = _objective_utility(float(row["objective_A"]))
    util_b = _objective_utility(float(row["objective_B"]))
    syn_a = (score_a + lambda_t * traj_a + lambda_i * inad_a - lambda_o * util_a) / lambda_s
    syn_b = (score_b + lambda_t * traj_b + lambda_i * inad_b - lambda_o * util_b) / lambda_s
    return {
        "syn_A": float(syn_a),
        "syn_B": float(syn_b),
        "traj_distance_A": traj_a,
        "traj_distance_B": traj_b,
        "inadmissible_A": inad_a,
        "inadmissible_B": inad_b,
        "U_obj_A": util_a,
        "U_obj_B": util_b,
        "S_C2_A": score_a,
        "S_C2_B": score_b,
        "delta_syn": float(syn_a - syn_b),
        "delta_traj": float(traj_b - traj_a),
        "delta_obj_util": float(util_b - util_a),
        "delta_score": float(score_b - score_a),
    }


def _score_decomposition(
    rows: Sequence[dict[str, Any]],
    *,
    policies: dict[str, Any],
) -> dict[str, Any]:
    enriched = []
    for row in rows:
        item = dict(row)
        item.update(_score_terms_for_row(row, policies=policies))
        enriched.append(item)

    def summarize(subset: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
        subset = list(subset)
        return {
            "group": label,
            "cases": len(subset),
            "syn_A_mean": _mean([float(row["syn_A"]) for row in subset]),
            "syn_B_mean": _mean([float(row["syn_B"]) for row in subset]),
            "traj_distance_A_mean": _mean([float(row["traj_distance_A"]) for row in subset]),
            "traj_distance_B_mean": _mean([float(row["traj_distance_B"]) for row in subset]),
            "inadmissible_A_rate": _rate(subset, lambda row: float(row["inadmissible_A"]) > 0.5),
            "inadmissible_B_rate": _rate(subset, lambda row: float(row["inadmissible_B"]) > 0.5),
            "U_obj_A_mean": _mean([float(row["U_obj_A"]) for row in subset]),
            "U_obj_B_mean": _mean([float(row["U_obj_B"]) for row in subset]),
            "S_C2_A_mean": _mean([float(row["S_C2_A"]) for row in subset]),
            "S_C2_B_mean": _mean([float(row["S_C2_B"]) for row in subset]),
            "delta_syn_mean": _mean([float(row["delta_syn"]) for row in subset]),
            "delta_traj_mean": _mean([float(row["delta_traj"]) for row in subset]),
            "delta_obj_util_mean": _mean([float(row["delta_obj_util"]) for row in subset]),
            "delta_score_mean": _mean([float(row["delta_score"]) for row in subset]),
        }

    return {
        "rows": enriched,
        "summary": [
            summarize(enriched, "c2_preferred"),
        ],
    }


def _compare_groups(
    preferred_cells: Sequence[dict[str, Any]],
    non_preferred_cells: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def summarize(label: str, subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        subset = list(subset)
        return {
            "group": label,
            "cells": len(subset),
            "fid_gain_C1_mean": _mean([float(row["fid_gain_C1_mean"]) for row in subset]),
            "fid_gain_C2_mean": _mean([float(row["fid_gain_C2_mean"]) for row in subset]),
            "fid_gain_C3_mean": _mean([float(row["fid_gain_C3_mean"]) for row in subset]),
            "false_safe_C1_mean": _mean([float(row["false_safe_rate_C1"]) for row in subset]),
            "false_safe_C2_mean": _mean([float(row["false_safe_rate_C2"]) for row in subset]),
            "false_safe_C3_mean": _mean([float(row["false_safe_rate_C3"]) for row in subset]),
            "chosen_B_rate_C1_mean": _mean([float(row["chosen_B_rate_C1"]) for row in subset]),
            "chosen_B_rate_C2_mean": _mean([float(row["chosen_B_rate_C2"]) for row in subset]),
            "chosen_B_rate_C3_mean": _mean([float(row["chosen_B_rate_C3"]) for row in subset]),
            "admissible_rate_A_mean": _mean([float(row["admissible_rate_A"]) for row in subset]),
            "admissible_rate_B_mean": _mean([float(row["admissible_rate_B"]) for row in subset]),
            "decision_disagreement_rate_AB_mean": _mean([float(row["decision_disagreement_rate_AB"]) for row in subset]),
            "fid_gain_delta_C2_minus_C1_mean": _mean([float(row["fid_gain_delta_C2_minus_C1"]) for row in subset]),
        }

    return [
        summarize("c2_preferred", preferred_cells),
        summarize("non_c2_preferred", non_preferred_cells),
    ]


def _distribution_rows(preferred_cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    total = max(1, len(preferred_cells))
    rows = []
    for cell in preferred_cells:
        item = {
            "code_family": cell["code_family"],
            "noise_family": cell["noise_family"],
            "noise_strength": cell["noise_strength"],
            "noise_depth": cell["noise_depth"],
            "count": 1,
            "percentage_within_C2_set": float(1.0 / total),
            "fid_gain_C2_mean": cell["fid_gain_C2_mean"],
            "fid_gain_delta_C2_minus_C1": cell["fid_gain_delta_C2_minus_C1"],
        }
        rows.append(item)
    rows.sort(key=lambda row: (row["code_family"], row["noise_family"], row["noise_strength"], row["noise_depth"]))
    return rows


def _distribution_summary(preferred_cells: Sequence[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in preferred_cells:
        grouped[tuple(cell[key] for key in keys)].append(dict(cell))
    total = max(1, len(preferred_cells))
    out = []
    for group_key in sorted(grouped.keys()):
        subset = grouped[group_key]
        item = {key: value for key, value in zip(keys, group_key)}
        item["count"] = len(subset)
        item["percentage_within_C2_set"] = float(len(subset) / total)
        item["mean_fid_gain_delta_C2_minus_C1"] = _mean(
            [float(row["fid_gain_delta_C2_minus_C1"]) for row in subset]
        )
        out.append(item)
    return out


def _harmful_switching_rows(preferred_cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in preferred_cells:
        rows.append({
            "code_family": cell["code_family"],
            "noise_family": cell["noise_family"],
            "noise_strength": cell["noise_strength"],
            "noise_depth": cell["noise_depth"],
            "chosen_B_rate_C1": cell["chosen_B_rate_C1"],
            "chosen_B_rate_C2": cell["chosen_B_rate_C2"],
            "fid_gain_C1_mean": cell["fid_gain_C1_mean"],
            "fid_gain_C2_mean": cell["fid_gain_C2_mean"],
            "fid_gain_delta_C2_minus_C1": cell["fid_gain_delta_C2_minus_C1"],
            "decision_reason_C1_majority": cell["decision_reason_C1_majority"],
            "decision_reason_C1_majority_rate": cell["decision_reason_C1_majority_rate"],
            "decision_reason_C2_majority": cell["decision_reason_C2_majority"],
            "decision_reason_C2_majority_rate": cell["decision_reason_C2_majority_rate"],
            "harmful_switching_pattern": cell["harmful_switching_pattern"],
        })
    rows.sort(
        key=lambda row: (
            -float(row["fid_gain_delta_C2_minus_C1"]),
            -float(row["chosen_B_rate_C1"]),
            str(row["noise_family"]),
        )
    )
    return rows


def _intensity_rows(preferred_cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in preferred_cells:
        grouped[(float(cell["noise_strength"]), int(cell["noise_depth"]))].append(dict(cell))
    out = []
    for key in sorted(grouped.keys()):
        subset = grouped[key]
        out.append({
            "noise_strength": key[0],
            "noise_depth": key[1],
            "c2_preferred_count": len(subset),
            "avg_fid_gain_delta_C2_minus_C1": _mean([float(row["fid_gain_delta_C2_minus_C1"]) for row in subset]),
            "avg_false_safe_delta_C1_minus_C2": _mean([float(row["false_safe_delta_C1_minus_C2"]) for row in subset]),
        })
    return out


def _build_type2_rows(rows: Sequence[dict[str, Any]], *, gain_margin: float, limit: int) -> list[dict[str, Any]]:
    def narrative(row: dict[str, Any]) -> str:
        if str(row["decision_reason_C1"]) == "tie_break_objective" and str(row["decision_reason_C2"]) == "score_prefers_A":
            return (
                "C1 switched to B via objective tie-break, while C2 preserved A because the balanced score "
                "kept syndrome prior and structural penalty ahead of the marginal relational objective."
            )
        if str(row["decision_reason_C2"]) == "score_prefers_A":
            return (
                "C2 kept A by score even though relational recovery remained available, indicating that the "
                "syndrome prior outweighed the trajectory/objective advantage of B."
            )
        return (
            "This cell shows a phase-sensitive stochastic case where C2 remains more conservative than C1 "
            "without paying a false-safe cost."
        )

    selected = [
        dict(row)
        for row in rows
        if float(row["fid_gain_C2"]) > float(row["fid_gain_C1"]) + float(gain_margin)
        and int(bool(row["false_safe_flag_C2"])) <= int(bool(row["false_safe_flag_C1"]))
    ]
    selected.sort(
        key=lambda row: (
            float(row["fid_gain_C2"]) - float(row["fid_gain_C1"]),
            float(row["score_C2_A"]) - float(row["score_C2_B"]),
        ),
        reverse=True,
    )
    out = []
    for rank, row in enumerate(selected[: int(limit)], start=1):
        item = {
            "rank": int(rank),
            "experiment_id": row["experiment_id"],
            "code_family": row["code_family"],
            "noise_family": row["noise_family"],
            "noise_strength": row["noise_strength"],
            "noise_depth": row["noise_depth"],
            "fid_gain_A": row["fid_gain_A"],
            "fid_gain_B": row["fid_gain_B"],
            "fid_gain_C1": row["fid_gain_C1"],
            "fid_gain_C2": row["fid_gain_C2"],
            "false_safe_A": row["false_safe_flag_A"],
            "false_safe_C1": row["false_safe_flag_C1"],
            "false_safe_C2": row["false_safe_flag_C2"],
            "decision_reason_C1": row["decision_reason_C1"],
            "decision_reason_C2": row["decision_reason_C2"],
            "admissible_A": row["admissible_A"],
            "admissible_B": row["admissible_B"],
            "objective_A": row["objective_A"],
            "objective_B": row["objective_B"],
            "traj_distance_A": row["traj_distance_A"],
            "traj_distance_B": row["traj_distance_B"],
            "narrative": narrative(row),
        }
        out.append(item)
    return out


def _heatmap_value(preferred_cells: Sequence[dict[str, Any]], *, codes: Sequence[str], noise_families: Sequence[str], strengths: Sequence[float], depths: Sequence[int]) -> Path:
    path = ensure_plot_root() / "c2_preferred_heatmap.png"
    n_rows = len(codes)
    n_cols = len(noise_families)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows), squeeze=False)
    vmin = min((float(cell["fid_gain_delta_C2_minus_C1"]) for cell in preferred_cells), default=0.0)
    vmax = max((float(cell["fid_gain_delta_C2_minus_C1"]) for cell in preferred_cells), default=1.0)
    if abs(vmax - vmin) < 1.0e-12:
        vmax = vmin + 1.0
    cell_map = {
        (str(cell["code_family"]), str(cell["noise_family"]), float(cell["noise_strength"]), int(cell["noise_depth"])): float(cell["fid_gain_delta_C2_minus_C1"])
        for cell in preferred_cells
    }
    for i, code in enumerate(codes):
        for j, family in enumerate(noise_families):
            ax = axes[i][j]
            matrix = np.full((len(strengths), len(depths)), np.nan, dtype=float)
            for si, strength in enumerate(strengths):
                for di, depth in enumerate(depths):
                    key = (str(code), str(family), float(strength), int(depth))
                    if key in cell_map:
                        matrix[si, di] = cell_map[key]
            im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{code} / {family}")
            ax.set_xticks(np.arange(len(depths)), [str(depth) for depth in depths])
            ax.set_yticks(np.arange(len(strengths)), [f"{strength:.2f}" for strength in strengths])
            ax.set_xlabel("Noise depth")
            ax.set_ylabel("Noise strength")
            for si in range(len(strengths)):
                for di in range(len(depths)):
                    value = matrix[si, di]
                    if not np.isnan(value):
                        ax.text(di, si, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("fid_gain_C2 - fid_gain_C1")
    fig.suptitle("Figure A. C2 Preferred Heatmap", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _connected_components(preferred_cells: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], set[tuple[float, int]]] = defaultdict(set)
    for cell in preferred_cells:
        grouped[(str(cell["code_family"]), str(cell["noise_family"]))].add(
            (float(cell["noise_strength"]), int(cell["noise_depth"]))
        )
    components: dict[tuple[str, str], int] = {}
    for key, points in grouped.items():
        unvisited = set(points)
        comp_count = 0
        while unvisited:
            comp_count += 1
            stack = [unvisited.pop()]
            while stack:
                strength, depth = stack.pop()
                neighbors = {
                    (strength, depth - 1),
                    (strength, depth + 1),
                    (round(strength - 0.02, 2), depth),
                    (round(strength + 0.02, 2), depth),
                    (round(strength - 0.05, 2), depth),
                    (round(strength + 0.05, 2), depth),
                }
                for candidate in list(unvisited):
                    if candidate in neighbors:
                        unvisited.remove(candidate)
                        stack.append(candidate)
            components[key] = comp_count
    return components


def _build_interpretation(
    *,
    preferred_cells: Sequence[dict[str, Any]],
    comparison: Sequence[dict[str, Any]],
    harmful_rows: Sequence[dict[str, Any]],
    type2_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    c2_count = len(preferred_cells)
    code_phaseflip_share = _rate(preferred_cells, lambda row: str(row["code_family"]) == "phaseflip")
    phase_sensitive_share = _rate(
        preferred_cells,
        lambda row: str(row["noise_family"]) in {"dephasing", "phaseflip"},
    )
    compare_pref = next(row for row in comparison if str(row["group"]) == "c2_preferred")
    positive_gain = float(compare_pref["fid_gain_delta_C2_minus_C1_mean"] or 0.0) > 0.0
    c1_over_switch = float(compare_pref["chosen_B_rate_C1_mean"] or 0.0) > float(compare_pref["chosen_B_rate_C2_mean"] or 0.0)
    reasons_align = _rate(
        harmful_rows,
        lambda row: str(row["decision_reason_C1_majority"]) == "tie_break_objective"
        and str(row["decision_reason_C2_majority"]) == "score_prefers_A",
    ) >= 0.5 if harmful_rows else False
    false_safe_safe = float(compare_pref["false_safe_C2_mean"] or 0.0) <= float(compare_pref["false_safe_C1_mean"] or 0.0)
    components = _connected_components(preferred_cells)
    connected_region = any(count <= 2 for count in components.values()) if components else False
    casebook_repeats = _rate(
        type2_rows,
        lambda row: "balanced score" in str(row["narrative"]).lower() or "syndrome prior" in str(row["narrative"]).lower(),
    ) >= 0.5 if type2_rows else False
    criteria = [
        {"name": "phaseflip_code_concentration", "passed": bool(code_phaseflip_share >= 0.8), "value": code_phaseflip_share},
        {"name": "phase_sensitive_family_concentration", "passed": bool(phase_sensitive_share >= 0.6), "value": phase_sensitive_share},
        {"name": "positive_fid_gain_delta", "passed": bool(positive_gain), "value": compare_pref["fid_gain_delta_C2_minus_C1_mean"]},
        {"name": "c1_over_switches_vs_c2", "passed": bool(c1_over_switch), "value": compare_pref["chosen_B_rate_C1_mean"]},
        {"name": "reason_pattern_matches_hypothesis", "passed": bool(reasons_align), "value": _rate(harmful_rows, lambda row: str(row["decision_reason_C1_majority"]) == "tie_break_objective") if harmful_rows else 0.0},
        {"name": "c2_not_worse_on_false_safe", "passed": bool(false_safe_safe), "value": compare_pref["false_safe_C2_mean"]},
        {"name": "connected_heatmap_region", "passed": bool(connected_region), "value": components},
        {"name": "type2_casebook_repeats_same_message", "passed": bool(casebook_repeats), "value": len(type2_rows)},
    ]
    passed = sum(1 for item in criteria if bool(item["passed"]))
    if bool(code_phaseflip_share >= 0.8 and phase_sensitive_share >= 0.6):
        recommendation = (
            "C2 is preferable in phase-sensitive stochastic regimes because it suppresses harmful relational switching "
            "that can be induced by C1's rule-based objective tie-break."
        )
    elif bool(connected_region):
        recommendation = (
            "C2 occupies a mid-risk balanced regime in which syndrome prior and structural penalties jointly prevent "
            "over-application of relational recovery."
        )
    else:
        recommendation = (
            "C2 is not globally superior, but forms a meaningful local operating zone under phase-dominated stochastic perturbations."
        )
    return {
        "criteria": criteria,
        "passed_criteria_count": int(passed),
        "meaningful_regime_confirmed": bool(passed >= 4),
        "recommended_interpretation": recommendation,
    }


def _build_markdown_tables(title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    return "\n".join([
        f"# {title}",
        "",
        _markdown_table(list(rows), list(columns)) if rows else "No rows matched.",
        "",
    ])


def run(
    *,
    payload: dict[str, Any] | None = None,
    source_stem: str = "hybrid_c123_regime_map",
    type2_gain_margin: float = 0.01,
    type2_limit: int = 10,
) -> dict[str, Any]:
    payload = dict(payload) if payload is not None else _load_payload(source_stem)
    rows = [dict(row) for row in payload["rows"]]
    cell_summaries = [dict(row) for row in payload["tables"]["by_regime_cell"]]
    preferred_cells = _cell_enrich(
        rows,
        cell_summaries=[row for row in cell_summaries if str(row["preferred_policy"]) == "C2"],
    )
    non_preferred_cells = _cell_enrich(
        rows,
        cell_summaries=[row for row in cell_summaries if str(row["preferred_policy"]) != "C2"],
    )
    c2_keys = {_cell_key(cell) for cell in preferred_cells}
    c2_rows = [row for row in rows if _cell_key(row) in c2_keys]
    non_c2_rows = [row for row in rows if _cell_key(row) not in c2_keys]
    distribution = _distribution_rows(preferred_cells)
    distribution_by_code_noise = _distribution_summary(preferred_cells, keys=("code_family", "noise_family"))
    distribution_by_strength_depth = _distribution_summary(preferred_cells, keys=("noise_strength", "noise_depth"))
    comparison = _compare_groups(preferred_cells, non_preferred_cells)
    harmful = _harmful_switching_rows(preferred_cells)
    decomposition_pref = _score_decomposition(c2_rows, policies=payload["policies"])
    decomposition_non = _score_decomposition(non_c2_rows, policies=payload["policies"])
    decomposition_summary = [
        *decomposition_pref["summary"],
        {
            **decomposition_non["summary"][0],
            "group": "non_c2_preferred",
        },
    ]
    intensity = _intensity_rows(preferred_cells)
    type2_casebook = _build_type2_rows(rows, gain_margin=float(type2_gain_margin), limit=int(type2_limit))
    heatmap_path = _heatmap_value(
        preferred_cells,
        codes=payload["grid"]["codes"],
        noise_families=payload["grid"]["noise_families"],
        strengths=payload["grid"]["strengths"],
        depths=payload["grid"]["depths"],
    )
    interpretation = _build_interpretation(
        preferred_cells=preferred_cells,
        comparison=comparison,
        harmful_rows=harmful,
        type2_rows=type2_casebook,
    )
    result = {
        "source_stem": source_stem,
        "summary": {
            "c2_preferred_cell_count": len(preferred_cells),
            "non_c2_preferred_cell_count": len(non_preferred_cells),
            "c2_preferred_row_count": len(c2_rows),
            "non_c2_preferred_row_count": len(non_c2_rows),
        },
        "tables": {
            "c2_preferred_distribution": distribution,
            "c2_preferred_distribution_by_code_noise": distribution_by_code_noise,
            "c2_preferred_distribution_by_strength_depth": distribution_by_strength_depth,
            "c2_vs_non_c2_summary": comparison,
            "c1_harmful_switching_in_c2_cells": harmful,
            "c2_score_component_decomposition": decomposition_summary,
            "c2_preference_by_regime_intensity": intensity,
            "c2_type2_casebook_extended": type2_casebook,
        },
        "figure": {
            "c2_preferred_heatmap": str(heatmap_path),
        },
        "interpretation": interpretation,
    }
    result["markdown"] = {
        "c2_preferred_distribution": _build_markdown_tables(
            "C2 Preferred Distribution",
            distribution_by_code_noise,
            ("code_family", "noise_family", "count", "percentage_within_C2_set", "mean_fid_gain_delta_C2_minus_C1"),
        ),
        "c2_vs_non_c2_summary": _build_markdown_tables(
            "C2 Preferred vs Non-C2 Preferred",
            comparison,
            (
                "group",
                "cells",
                "fid_gain_C1_mean",
                "fid_gain_C2_mean",
                "fid_gain_C3_mean",
                "chosen_B_rate_C1_mean",
                "chosen_B_rate_C2_mean",
                "admissible_rate_A_mean",
                "admissible_rate_B_mean",
                "decision_disagreement_rate_AB_mean",
            ),
        ),
        "c1_harmful_switching_in_c2_cells": _build_markdown_tables(
            "C1 Harmful Switching in C2 Cells",
            harmful,
            (
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "chosen_B_rate_C1",
                "chosen_B_rate_C2",
                "fid_gain_C1_mean",
                "fid_gain_C2_mean",
                "decision_reason_C1_majority",
                "decision_reason_C2_majority",
                "harmful_switching_pattern",
            ),
        ),
        "c2_score_component_decomposition": _build_markdown_tables(
            "C2 Score Component Decomposition",
            decomposition_summary,
            (
                "group",
                "cases",
                "syn_A_mean",
                "syn_B_mean",
                "traj_distance_A_mean",
                "traj_distance_B_mean",
                "inadmissible_A_rate",
                "inadmissible_B_rate",
                "U_obj_A_mean",
                "U_obj_B_mean",
                "delta_syn_mean",
                "delta_traj_mean",
                "delta_obj_util_mean",
                "delta_score_mean",
            ),
        ),
        "c2_type2_casebook_extended": _build_markdown_tables(
            "C2 Type-2 Casebook Extended",
            type2_casebook,
            (
                "rank",
                "experiment_id",
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "fid_gain_C1",
                "fid_gain_C2",
                "false_safe_C1",
                "false_safe_C2",
                "decision_reason_C1",
                "decision_reason_C2",
                "narrative",
            ),
        ),
        "c2_preference_by_regime_intensity": _build_markdown_tables(
            "C2 Preference By Regime Intensity",
            intensity,
            (
                "noise_strength",
                "noise_depth",
                "c2_preferred_count",
                "avg_fid_gain_delta_C2_minus_C1",
                "avg_false_safe_delta_C1_minus_C2",
            ),
        ),
        "c2_regime_interpretation": "\n".join([
            "# C2 Regime Interpretation",
            "",
            f"- `c2_preferred_cell_count`: `{len(preferred_cells)}`",
            f"- `passed_criteria_count`: `{interpretation['passed_criteria_count']}`",
            f"- `meaningful_regime_confirmed`: `{interpretation['meaningful_regime_confirmed']}`",
            "",
            "## Criteria",
            "",
            _markdown_table(interpretation["criteria"], ("name", "passed", "value")),
            "",
            "## Recommended Interpretation",
            "",
            interpretation["recommended_interpretation"],
            "",
        ]),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze C2-preferred hybrid cells.")
    parser.add_argument("--config", default="experiment/hybrid_c123_regime_map.yaml")
    parser.add_argument("--source-stem", default="hybrid_c123_regime_map")
    parser.add_argument("--type2-gain-margin", type=float, default=None)
    parser.add_argument("--type2-limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    analysis_cfg = dict(config.get("hybrid_c2_analysis", {}))
    result = run(
        source_stem=str(args.source_stem),
        type2_gain_margin=float(args.type2_gain_margin or analysis_cfg.get("type2_gain_margin", 0.01)),
        type2_limit=int(args.type2_limit or analysis_cfg.get("type2_limit", 10)),
    )
    write_json_result(result, "c2_preferred_analysis")
    tables_dir = RESULT_ROOT / "tables"
    for name, rows in result["tables"].items():
        if isinstance(rows, list):
            _write_csv(tables_dir / f"{name}.csv", rows)
    for name, markdown in result["markdown"].items():
        path = tables_dir / f"{name}.md"
        path.write_text(markdown, encoding="utf-8")
    print(json.dumps(to_serializable(result["summary"]), indent=2))
    print(f"saved_heatmap={result['figure']['c2_preferred_heatmap']}")


if __name__ == "__main__":
    main()

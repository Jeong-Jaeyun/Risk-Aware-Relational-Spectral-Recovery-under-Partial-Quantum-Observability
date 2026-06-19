"""Regime-map sweep and figure export for hybrid C1/C2/C3/C3R policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coherent_veto_common import ensure_plot_root
from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv
from .run_hybrid_c123_baseline import (
    C3RPolicyConfig,
    DEFAULT_CODES,
    PolicyScoreConfig,
    SyndromeObservationConfig,
    _aggregate_rows,
    _c3r_uncertainty_bin_rows,
    _group_rows,
    _parse_csv,
    _reason_rows,
    run as run_hybrid_c123_baseline,
)


POLICY_ORDER = ("C1", "C2", "C3", "C3R")
MODE_ORDER = ("A", "B", "C1", "C2", "C3", "C3R")
POLICY_COLOR = {
    "A": "#4C78A8",
    "B": "#F58518",
    "C1": "#54A24B",
    "C2": "#E45756",
    "C3": "#72B7B2",
    "C3R": "#B279A2",
}
PREFERRED_POLICY_COLOR = {
    "C1": "#54A24B",
    "C2": "#E45756",
    "C3": "#72B7B2",
    "C3R": "#B279A2",
}


def _policy_rank(policy: str) -> int:
    try:
        return POLICY_ORDER.index(str(policy))
    except ValueError:
        return len(POLICY_ORDER)


def _preferred_policy(
    row: dict[str, Any],
    *,
    safety_tolerance: float,
    gain_tolerance: float,
) -> dict[str, Any]:
    metrics = []
    for policy in POLICY_ORDER:
        metrics.append({
            "policy": policy,
            "false_safe_fidelity_rate": float(row[f"false_safe_fidelity_rate_{policy}"]),
            "false_safe_rate": float(row[f"false_safe_rate_{policy}"]),
            "fid_gain_mean": float(row[f"fid_gain_{policy}_mean"]),
            "logical_success_rate": float(row[f"logical_success_rate_{policy}"]),
            "nonworsen_rate": float(row[f"nonworsen_rate_{policy}"]),
            "chosen_B_rate": float(row[f"chosen_B_rate_{policy}"]),
        })
    min_false_safe_fidelity = min(item["false_safe_fidelity_rate"] for item in metrics)
    safety_candidates = [
        item
        for item in metrics
        if item["false_safe_fidelity_rate"] <= min_false_safe_fidelity + float(safety_tolerance)
    ]
    best_gain = max(item["fid_gain_mean"] for item in safety_candidates)
    gain_candidates = [
        item
        for item in safety_candidates
        if item["fid_gain_mean"] >= best_gain - float(gain_tolerance)
    ]
    best_logical = max(item["logical_success_rate"] for item in gain_candidates)
    logical_candidates = [
        item for item in gain_candidates if item["logical_success_rate"] >= best_logical - 1.0e-12
    ]
    best_nonworsen = max(item["nonworsen_rate"] for item in logical_candidates)
    nonworsen_candidates = [
        item for item in logical_candidates if item["nonworsen_rate"] >= best_nonworsen - 1.0e-12
    ]
    chosen = sorted(
        nonworsen_candidates,
        key=lambda item: (float(item["chosen_B_rate"]), _policy_rank(str(item["policy"]))),
    )[0]
    return {
        "preferred_policy": str(chosen["policy"]),
        "preferred_false_safe_fidelity_rate": float(chosen["false_safe_fidelity_rate"]),
        "preferred_false_safe_rate": float(chosen["false_safe_rate"]),
        "preferred_fid_gain_mean": float(chosen["fid_gain_mean"]),
        "preferred_logical_success_rate": float(chosen["logical_success_rate"]),
        "preferred_nonworsen_rate": float(chosen["nonworsen_rate"]),
        "preferred_chosen_B_rate": float(chosen["chosen_B_rate"]),
        "min_false_safe_fidelity_rate": float(min_false_safe_fidelity),
        "min_false_safe_rate": float(min(item["false_safe_rate"] for item in metrics)),
        "max_fid_gain_mean_within_safety_band": float(best_gain),
    }


def _annotate_regime_cells(
    rows: Sequence[dict[str, Any]],
    *,
    safety_tolerance: float,
    gain_tolerance: float,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item.update(
            _preferred_policy(
                item,
                safety_tolerance=float(safety_tolerance),
                gain_tolerance=float(gain_tolerance),
            )
        )
        out.append(item)
    return out


def _reason_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        *_reason_rows(rows, mode="C1"),
        *_reason_rows(rows, mode="C2"),
        *_reason_rows(rows, mode="C3"),
        *_reason_rows(rows, mode="C3R"),
    ]


def _group_preferred_policy(rows: Sequence[dict[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(dict(row))
    table = []
    for group_key in sorted(grouped.keys()):
        subset = grouped[group_key]
        counts: dict[str, int] = {}
        for row in subset:
            counts[str(row["preferred_policy"])] = counts.get(str(row["preferred_policy"]), 0) + 1
        preferred_policy = sorted(
            counts.items(),
            key=lambda item: (-item[1], _policy_rank(item[0])),
        )[0][0]
        entry = {key: value for key, value in zip(keys, group_key)}
        entry["cases"] = len(subset)
        entry["preferred_policy"] = str(preferred_policy)
        entry["preferred_policy_rate"] = float(counts[preferred_policy] / len(subset))
        entry["mean_preferred_fid_gain"] = float(
            np.mean([float(row["preferred_fid_gain_mean"]) for row in subset])
        )
        entry["mean_preferred_false_safe_fidelity_rate"] = float(
            np.mean([float(row["preferred_false_safe_fidelity_rate"]) for row in subset])
        )
        entry["mean_preferred_false_safe_rate"] = float(
            np.mean([float(row["preferred_false_safe_rate"]) for row in subset])
        )
        table.append(entry)
    return table


def _bar_positions(n_groups: int, n_series: int, width: float = 0.16) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n_groups, dtype=float)
    offsets = (np.arange(n_series, dtype=float) - (n_series - 1) / 2.0) * width
    return x, offsets


def _save_fid_gain_comparison(table: Sequence[dict[str, Any]], *, filename: str) -> Path:
    path = ensure_plot_root() / filename
    labels = [f"{row['code_family']}\n{row['noise_family']}" for row in table]
    x, offsets = _bar_positions(len(table), len(MODE_ORDER))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.2))
    for index, mode in enumerate(MODE_ORDER):
        values = [float(row[f"fid_gain_{mode}_mean"]) for row in table]
        ax.bar(x + offsets[index], values, width=0.15, label=mode, color=POLICY_COLOR[mode])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("Mean fidelity gain")
    ax.set_title("Figure 1. Hybrid C1/C2/C3/C3R Fidelity Gain Comparison")
    ax.legend(ncols=len(MODE_ORDER), fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_false_safe_comparison(table: Sequence[dict[str, Any]], *, filename: str) -> Path:
    path = ensure_plot_root() / filename
    labels = [f"{row['code_family']}\n{row['noise_family']}" for row in table]
    x, offsets = _bar_positions(len(table), len(MODE_ORDER))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.2))
    for index, mode in enumerate(MODE_ORDER):
        values = [float(row[f"false_safe_rate_{mode}"]) for row in table]
        ax.bar(x + offsets[index], values, width=0.15, label=mode, color=POLICY_COLOR[mode])
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("False-safe rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure 2. Hybrid C1/C2/C3/C3R False-Safe Comparison")
    ax.legend(ncols=len(MODE_ORDER), fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_reason_composition(table: Sequence[dict[str, Any]], *, filename: str) -> Path:
    path = ensure_plot_root() / filename
    modes = list(POLICY_ORDER)
    reasons_by_mode: dict[str, list[str]] = {mode: [] for mode in modes}
    for row in table:
        mode = str(row["mode"])
        if mode in reasons_by_mode:
            reasons_by_mode[mode].append(str(row["reason"]))
    all_reasons = sorted({reason for reasons in reasons_by_mode.values() for reason in reasons})
    if not all_reasons:
        all_reasons = ["none"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(modes), dtype=float)
    bottom = np.zeros(len(modes), dtype=float)
    cmap = plt.get_cmap("tab20")
    for index, reason in enumerate(all_reasons):
        heights = []
        for mode in modes:
            match = next((row for row in table if str(row["mode"]) == mode and str(row["reason"]) == reason), None)
            heights.append(float(match["rate"]) if match is not None else 0.0)
        ax.bar(x, heights, bottom=bottom, width=0.55, label=reason, color=cmap(index))
        bottom += np.asarray(heights, dtype=float)
    ax.set_xticks(x, modes)
    ax.set_ylabel("Decision-reason rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure 3. Hybrid Policy Reason Composition")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_tradeoff_frontier(overall: dict[str, Any], *, filename: str) -> Path:
    path = ensure_plot_root() / filename
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for mode in MODE_ORDER:
        x = float(overall[f"false_safe_fidelity_rate_{mode}"])
        y = float(overall[f"fid_gain_{mode}_mean"])
        ax.scatter([x], [y], s=90, color=POLICY_COLOR[mode], label=mode)
        ax.annotate(mode, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("False-safe fidelity rate")
    ax.set_ylabel("Mean fidelity gain")
    ax.set_title("Figure 4. Hybrid Trade-off Frontier")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_regime_boundary_map(
    cells: Sequence[dict[str, Any]],
    *,
    codes: Sequence[str],
    noise_families: Sequence[str],
    strengths: Sequence[float],
    depths: Sequence[int],
    filename: str,
) -> Path:
    path = ensure_plot_root() / filename
    col_labels = [f"p={strength:.2f}\nd={depth}" for strength in strengths for depth in depths]
    n_rows = len(codes) * len(noise_families)
    n_cols = len(col_labels)
    matrix = np.full((n_rows, n_cols), np.nan, dtype=float)
    row_labels: list[str] = []
    code_index = {str(code): index for index, code in enumerate(codes)}
    family_index = {str(family): index for index, family in enumerate(noise_families)}
    policy_to_value = {policy: float(index) for index, policy in enumerate(POLICY_ORDER)}
    for code in codes:
        for family in noise_families:
            row_labels.append(f"{code}\n{family}")
    for cell in cells:
        row_idx = code_index[str(cell["code_family"])] * len(noise_families) + family_index[str(cell["noise_family"])]
        col_idx = list(strengths).index(float(cell["noise_strength"])) * len(depths) + list(depths).index(int(cell["noise_depth"]))
        matrix[row_idx, col_idx] = policy_to_value.get(str(cell["preferred_policy"]), np.nan)
    cmap = matplotlib.colors.ListedColormap([
        PREFERRED_POLICY_COLOR[policy] for policy in POLICY_ORDER
    ])
    norm = matplotlib.colors.BoundaryNorm(
        [index - 0.5 for index in range(len(POLICY_ORDER) + 1)],
        cmap.N,
    )
    fig, ax = plt.subplots(figsize=(max(11, n_cols * 0.9), max(6, n_rows * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(n_cols), col_labels, rotation=0)
    ax.set_yticks(np.arange(n_rows), row_labels)
    ax.set_title("Figure 5. Hybrid Regime Boundary Map")
    ax.set_xlabel("Noise strength / depth")
    ax.set_ylabel("Code / noise family")
    cbar = fig.colorbar(im, ax=ax, ticks=[float(index) for index in range(len(POLICY_ORDER))])
    cbar.ax.set_yticklabels(list(POLICY_ORDER))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Hybrid C1/C2/C3/C3R Regime Map",
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
            "decision_disagreement_rate_C1A",
            "decision_disagreement_rate_C2A",
            "decision_disagreement_rate_C3A",
            "decision_disagreement_rate_C3RA",
            "decision_disagreement_rate_C3R_vs_C2",
        ]),
        "",
        "## By Code And Noise Family",
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
        "## Preferred Policy By Regime Cell",
        "",
        _markdown_table(result["tables"]["by_regime_cell"], [
            "code_family",
            "noise_family",
            "noise_strength",
            "noise_depth",
            "cases",
            "preferred_policy",
            "preferred_fid_gain_mean",
            "preferred_false_safe_fidelity_rate",
            "preferred_false_safe_rate",
            "preferred_chosen_B_rate",
            "min_false_safe_rate",
            "min_false_safe_fidelity_rate",
        ]),
        "",
        "## Preferred Policy Summary",
        "",
        _markdown_table(result["tables"]["preferred_policy_summary"], [
            "preferred_policy",
            "cases",
            "preferred_policy_rate",
            "mean_preferred_fid_gain",
            "mean_preferred_false_safe_fidelity_rate",
            "mean_preferred_false_safe_rate",
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
        "",
        "## Figures",
        "",
    ]
    for name, path in result["figures"].items():
        lines.append(f"- `{name}`: `{path}`")
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
    syndrome_obs_cfg: SyndromeObservationConfig | None,
    c1_objective_tol: float,
    c1_tie_break_requires_syndrome_consistent: bool,
    regime_safety_tolerance: float,
    regime_gain_tolerance: float,
    experiment_config: str,
    output_stem: str,
    plot_prefix: str,
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
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
        syndrome_obs_cfg=syndrome_obs_cfg,
        c1_objective_tol=c1_objective_tol,
        c1_tie_break_requires_syndrome_consistent=c1_tie_break_requires_syndrome_consistent,
        experiment_config=experiment_config,
        output_stem=output_stem,
        max_workers=max_workers,
        resume=resume,
    )
    rows = [dict(row) for row in baseline["rows"]]
    by_regime_cell = _annotate_regime_cells(
        _group_rows(
            rows,
            keys=("code_family", "noise_family", "noise_strength", "noise_depth"),
        ),
        safety_tolerance=float(regime_safety_tolerance),
        gain_tolerance=float(regime_gain_tolerance),
    )
    tables = {
        "by_code": _group_rows(rows, keys=("code_family",)),
        "by_noise_family": _group_rows(rows, keys=("noise_family",)),
        "by_noise_strength": _group_rows(rows, keys=("noise_strength",)),
        "by_noise_depth": _group_rows(rows, keys=("noise_depth",)),
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
        "by_regime_cell": by_regime_cell,
        "preferred_policy_summary": _group_preferred_policy(by_regime_cell, keys=("preferred_policy",)),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
        "reason_summary": _reason_summary(rows),
    }
    figures = {
        "hybrid_c123_fid_gain_comparison": str(
            _save_fid_gain_comparison(
                tables["by_code_and_noise_family"],
                filename=f"{plot_prefix}_fid_gain_comparison.png",
            )
        ),
        "hybrid_c123_false_safe_comparison": str(
            _save_false_safe_comparison(
                tables["by_code_and_noise_family"],
                filename=f"{plot_prefix}_false_safe_comparison.png",
            )
        ),
        "hybrid_c123_reason_composition": str(
            _save_reason_composition(
                tables["reason_summary"],
                filename=f"{plot_prefix}_reason_composition.png",
            )
        ),
        "hybrid_c123_tradeoff_frontier": str(
            _save_tradeoff_frontier(
                baseline["overall"],
                filename=f"{plot_prefix}_tradeoff_frontier.png",
            )
        ),
        "hybrid_c123_regime_boundary_map": str(
            _save_regime_boundary_map(
                by_regime_cell,
                codes=list(codes),
                noise_families=list(noise_families),
                strengths=list(strengths),
                depths=list(depths),
                filename=f"{plot_prefix}_regime_boundary_map.png",
            )
        ),
    }
    result = {
        "grid": baseline["grid"],
        "policies": baseline["policies"],
        "syndrome_observation": baseline["syndrome_observation"],
        "selection_rules": {
            "regime_safety_tolerance": float(regime_safety_tolerance),
            "regime_gain_tolerance": float(regime_gain_tolerance),
            "preferred_policy_rule": (
                "min false-safe-fidelity first, then max fidelity gain, logical success, "
                "nonworsen, and lower chosen-B rate"
            ),
        },
        "overall": baseline["overall"],
        "rows": rows,
        "tables": tables,
        "figures": figures,
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid C1/C2/C3/C3R regime-map sweep.")
    parser.add_argument("--config", default="experiment/hybrid_c123_regime_map.yaml")
    parser.add_argument("--codes", default=None)
    parser.add_argument("--noise-families", default=None)
    parser.add_argument("--strengths", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--plot-prefix", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    cfg = dict(config.get("hybrid_c123", {}))
    regime_cfg = dict(config.get("hybrid_c123_regime_map", {}))
    codes = _parse_csv(args.codes, str) or list(cfg.get("codes", DEFAULT_CODES))
    state_configs = {str(key): str(value) for key, value in dict(cfg.get("state_configs", {})).items()}
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
    stem = resolve_output_stem(config, "hybrid_c123_regime_map", args.output_stem)
    plot_prefix = args.plot_prefix or "hybrid_c123"
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
        regime_safety_tolerance=float(regime_cfg.get("safety_tolerance", 0.02)),
        regime_gain_tolerance=float(regime_cfg.get("gain_tolerance", 0.005)),
        experiment_config=args.config,
        output_stem=stem,
        plot_prefix=plot_prefix,
        max_workers=int(args.workers if args.workers is not None else os.environ.get("BIQMN_WORKERS", "1")),
        resume=not bool(args.no_resume),
    )
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_raw.csv", result["rows"])
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

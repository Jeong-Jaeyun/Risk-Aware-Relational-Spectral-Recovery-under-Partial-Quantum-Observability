"""Partial-syndrome pilot sweep for hybrid C1/C2/C3/C3R policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
    run as run_hybrid_c123_baseline,
)
from .run_hybrid_c123_regime_map import _annotate_regime_cells


def _preferred_policy_counts_by_ratio(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(float(cell["syndrome_observation_ratio"]), []).append(dict(cell))
    rows: list[dict[str, Any]] = []
    for ratio in sorted(grouped.keys(), reverse=True):
        subset = grouped[ratio]
        counts = {"C1": 0, "C2": 0, "C3": 0, "C3R": 0}
        for cell in subset:
            counts[str(cell["preferred_policy"])] += 1
        total = len(subset)
        rows.append(
            {
                "syndrome_observation_ratio": float(ratio),
                "cases": int(total),
                "C1_count": int(counts["C1"]),
                "C2_count": int(counts["C2"]),
                "C3_count": int(counts["C3"]),
                "C3R_count": int(counts["C3R"]),
                "C1_rate": 0.0 if total == 0 else float(counts["C1"] / total),
                "C2_rate": 0.0 if total == 0 else float(counts["C2"] / total),
                "C3_rate": 0.0 if total == 0 else float(counts["C3"] / total),
                "C3R_rate": 0.0 if total == 0 else float(counts["C3R"] / total),
            }
        )
    return rows


def _save_preferred_policy_counts(rows: Sequence[dict[str, Any]], *, filename: str) -> Path:
    path = ensure_plot_root() / filename
    labels = [f"{row['syndrome_observation_ratio']:.2f}" for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bottom = np.zeros(len(rows), dtype=float)
    colors = {"C1": "#54A24B", "C2": "#E45756", "C3": "#72B7B2", "C3R": "#B279A2"}
    for policy in ("C1", "C2", "C3", "C3R"):
        values = np.asarray([float(row[f"{policy}_count"]) for row in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, width=0.6, color=colors[policy], label=policy)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_xlabel("Syndrome observation ratio")
    ax.set_ylabel("Preferred-policy cell count")
    ax.set_title("Partial Syndrome: Preferred Policy Counts vs Observation Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_metric_by_ratio(
    rows: Sequence[dict[str, Any]],
    *,
    field_template: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    path = ensure_plot_root() / filename
    ratios = [float(row["syndrome_observation_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {"C1": "#54A24B", "C2": "#E45756", "C3": "#72B7B2", "C3R": "#B279A2"}
    for policy in ("C1", "C2", "C3", "C3R"):
        values = [float(row[field_template.format(policy=policy)]) for row in rows]
        ax.plot(ratios, values, marker="o", linewidth=2.0, color=colors[policy], label=policy)
    ax.invert_xaxis()
    ax.set_xlabel("Syndrome observation ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Partial Syndrome Baseline",
            "",
            "## Overall",
            "",
            _markdown_table(
                [result["overall"]],
                [
                    "cases",
                    "backend",
                    "fid_gain_C1_mean",
                    "fid_gain_C2_mean",
                    "fid_gain_C3_mean",
                    "fid_gain_C3R_mean",
                    "logical_success_rate_C2",
                    "logical_success_rate_C3R",
                    "nonworsen_rate_C2",
                    "nonworsen_rate_C3R",
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
                ],
            ),
            "",
            "## By Observation Ratio",
            "",
            _markdown_table(
                result["tables"]["by_syndrome_obs_ratio"],
                [
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
                ],
            ),
            "",
            "## Preferred Policy Counts By Observation Ratio",
            "",
            _markdown_table(
                result["tables"]["preferred_policy_counts_by_ratio"],
                [
                    "syndrome_observation_ratio",
                    "cases",
                    "C1_count",
                    "C2_count",
                    "C3_count",
                    "C3R_count",
                    "C1_rate",
                    "C2_rate",
                    "C3_rate",
                    "C3R_rate",
                ],
            ),
            "",
            "## C3R Gate Summary",
            "",
            _markdown_table(
                [result["overall"]],
                [
                    "chosen_B_rate_C2",
                    "chosen_B_rate_C3R",
                    "decision_disagreement_rate_C3R_vs_C2",
                    "c2_B_count",
                    "c3r_block_count",
                    "c3r_blocks_c2_switch_rate",
                    "c3r_gate_uncertainty_rate_given_c2_B",
                    "c3r_allow_B_rate_given_c2_B",
                    "c3r_prevented_harmful_switch_rate_given_block",
                    "c3r_missed_beneficial_switch_rate_given_block",
                    "c3r_raw_syndrome_uncertainty_mean",
                ],
            ),
            "",
            "## C3R Switch Intervention Quality",
            "",
            _markdown_table(
                [result["overall"]],
                [
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
                ],
            ),
            "",
            "## Tail Risk and Oracle Diagnostics",
            "",
            _markdown_table(
                [result["overall"]],
                [
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
                ],
            ),
            "",
            "## C3R By Raw Uncertainty Bin",
            "",
            _markdown_table(
                result["tables"].get("c3r_by_uncertainty_bin", []),
                [
                    "c3r_raw_syndrome_uncertainty_bin",
                    "cases",
                    "c2_B_count",
                    "c3r_block_count",
                    "c3r_block_rate_given_c2_B",
                    "c3r_harmful_block_precision",
                    "c3r_harmful_switch_recall",
                    "c3r_beneficial_switch_block_rate",
                    "c3r_net_intervention_gain",
                ],
            ),
            "",
            "## Figures",
            "",
            *[f"- `{name}`: `{path}`" for name, path in result["figures"].items()],
            "",
        ]
    )


def run(
    *,
    codes: Sequence[str],
    state_configs: dict[str, str],
    kinds_by_code: dict[str, Sequence[str]],
    noise_families: Sequence[str],
    strengths: Sequence[float],
    depths: Sequence[int],
    seeds: Sequence[int],
    observation_ratios: Sequence[float],
    fidelity_margin: float,
    logical_success_threshold: float,
    c2_cfg: PolicyScoreConfig,
    c3_cfg: PolicyScoreConfig,
    c3r_cfg: C3RPolicyConfig | None = None,
    syndrome_obs_cfg: SyndromeObservationConfig,
    c1_objective_tol: float,
    c1_tie_break_requires_syndrome_consistent: bool,
    regime_safety_tolerance: float,
    regime_gain_tolerance: float,
    experiment_config: str,
    output_stem: str,
    plot_prefix: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ratio in observation_ratios:
        ratio_cfg = SyndromeObservationConfig(
            observation_ratio=float(ratio),
            noise_prob=float(syndrome_obs_cfg.noise_prob),
            ambiguity_level=float(syndrome_obs_cfg.ambiguity_level),
            measurement_error_prob=float(syndrome_obs_cfg.measurement_error_prob),
            reset_error_prob=float(syndrome_obs_cfg.reset_error_prob),
            consistency_threshold=float(syndrome_obs_cfg.consistency_threshold),
        )
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
            syndrome_obs_cfg=ratio_cfg,
            c1_objective_tol=c1_objective_tol,
            c1_tie_break_requires_syndrome_consistent=c1_tie_break_requires_syndrome_consistent,
            experiment_config=experiment_config,
            output_stem=f"{output_stem}_r{ratio:.2f}",
        )
        rows.extend(dict(row) for row in baseline["rows"])

    by_regime_cell = _annotate_regime_cells(
        _group_rows(
            rows,
            keys=(
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "syndrome_observation_ratio",
            ),
        ),
        safety_tolerance=float(regime_safety_tolerance),
        gain_tolerance=float(regime_gain_tolerance),
    )
    tables = {
        "by_syndrome_obs_ratio": _group_rows(rows, keys=("syndrome_observation_ratio",)),
        "by_code_and_obs_ratio": _group_rows(rows, keys=("code_family", "syndrome_observation_ratio")),
        "by_code_noise_and_obs_ratio": _group_rows(
            rows, keys=("code_family", "noise_family", "syndrome_observation_ratio")
        ),
        "by_regime_cell": by_regime_cell,
        "preferred_policy_counts_by_ratio": _preferred_policy_counts_by_ratio(by_regime_cell),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
    }
    figures = {
        "preferred_policy_counts": str(
            _save_preferred_policy_counts(
                tables["preferred_policy_counts_by_ratio"],
                filename=f"{plot_prefix}_preferred_policy_counts.png",
            )
        ),
        "false_safe_fidelity_vs_ratio": str(
            _save_metric_by_ratio(
                tables["by_syndrome_obs_ratio"],
                field_template="false_safe_fidelity_rate_{policy}",
                ylabel="False-safe fidelity rate",
                title="Partial Syndrome: Fidelity-Based False-Safe vs Observation Ratio",
                filename=f"{plot_prefix}_false_safe_fidelity_vs_ratio.png",
            )
        ),
        "fid_gain_vs_ratio": str(
            _save_metric_by_ratio(
                tables["by_syndrome_obs_ratio"],
                field_template="fid_gain_{policy}_mean",
                ylabel="Mean fidelity gain",
                title="Partial Syndrome: Fidelity Gain vs Observation Ratio",
                filename=f"{plot_prefix}_fid_gain_vs_ratio.png",
            )
        ),
    }
    result = {
        "grid": {
            "codes": [str(code) for code in codes],
            "noise_families": [str(family) for family in noise_families],
            "strengths": [float(value) for value in strengths],
            "depths": [int(value) for value in depths],
            "seeds": [int(value) for value in seeds],
            "observation_ratios": [float(value) for value in observation_ratios],
        },
        "policies": {
            "c2": c2_cfg.__dict__,
            "c3": c3_cfg.__dict__,
            "c3r": (c3r_cfg or C3RPolicyConfig()).__dict__,
            "c1": {
                "objective_tol": float(c1_objective_tol),
                "tie_break_requires_syndrome_consistent": bool(c1_tie_break_requires_syndrome_consistent),
            },
        },
        "syndrome_observation_base": syndrome_obs_cfg.__dict__,
        "overall": _aggregate_rows(rows),
        "rows": rows,
        "tables": tables,
        "figures": figures,
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the partial-syndrome pilot for hybrid C1/C2/C3/C3R.")
    parser.add_argument("--config", default="experiment/partial_syndrome_baseline.yaml")
    parser.add_argument("--codes", default=None)
    parser.add_argument("--noise-families", default=None)
    parser.add_argument("--strengths", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--observation-ratios", default=None)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--plot-prefix", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(experiment_config=args.config)
    cfg = dict(config.get("hybrid_c123", {}))
    partial_cfg = dict(config.get("partial_syndrome", {}))
    codes = _parse_csv(args.codes, str) or list(cfg.get("codes", DEFAULT_CODES))
    state_configs = {str(key): str(value) for key, value in dict(cfg.get("state_configs", {})).items()}
    kinds_by_code = {
        str(key): [str(item) for item in value]
        for key, value in dict(cfg.get("kinds_by_code", {})).items()
        if isinstance(value, (list, tuple))
    }
    noise_families = _parse_csv(args.noise_families, str) or [
        str(item) for item in partial_cfg.get("noise_families", cfg.get("noise_families", []))
    ]
    strengths = _parse_csv(args.strengths, float) or [
        float(item) for item in partial_cfg.get("strengths", cfg.get("strengths", []))
    ]
    depths = _parse_csv(args.depths, int) or [
        int(item) for item in partial_cfg.get("depths", cfg.get("depths", []))
    ]
    seeds = _parse_csv(args.seeds, int) or [
        int(item) for item in partial_cfg.get("seeds", cfg.get("seeds", []))
    ]
    observation_ratios = _parse_csv(args.observation_ratios, float) or [
        float(item) for item in partial_cfg.get("observation_ratios", [1.0, 0.75, 0.5, 0.25])
    ]
    c2 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c2", {})).items()})
    c3 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c3", {})).items()})
    c3r = C3RPolicyConfig(**{key: float(value) for key, value in dict(cfg.get("c3r", {})).items()})
    syndrome_obs = SyndromeObservationConfig(
        **{key: float(value) for key, value in dict(cfg.get("syndrome_observation", {})).items()}
    )
    c1_cfg = dict(cfg.get("c1", {}))
    stem = resolve_output_stem(config, "partial_syndrome_baseline", args.output_stem)
    plot_prefix = args.plot_prefix or stem
    result = run(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        noise_families=noise_families,
        strengths=strengths,
        depths=depths,
        seeds=seeds,
        observation_ratios=observation_ratios,
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
        regime_safety_tolerance=float(partial_cfg.get("safety_tolerance", 0.02)),
        regime_gain_tolerance=float(partial_cfg.get("gain_tolerance", 0.005)),
        experiment_config=args.config,
        output_stem=stem,
        plot_prefix=plot_prefix,
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

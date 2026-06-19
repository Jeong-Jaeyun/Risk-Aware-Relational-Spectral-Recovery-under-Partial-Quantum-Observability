"""Ambiguity x measurement/reset focused regime-map for hybrid C1/C2/C3/C3R."""
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
from .common import RESULT_ROOT, load_config, progress_iter, resolve_output_stem, to_serializable, write_json_result
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


def _preferred_policy_counts_by_combo(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for cell in cells:
        key = (
            float(cell["syndrome_ambiguity_level"]),
            float(cell["measurement_error_prob"]),
        )
        grouped.setdefault(key, []).append(dict(cell))
    rows: list[dict[str, Any]] = []
    for ambiguity_level, measurement_reset_prob in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        subset = grouped[(ambiguity_level, measurement_reset_prob)]
        counts = {"C1": 0, "C2": 0, "C3": 0, "C3R": 0}
        for cell in subset:
            counts[str(cell["preferred_policy"])] += 1
        total = len(subset)
        dominant = max(counts, key=counts.get)
        rows.append(
            {
                "syndrome_ambiguity_level": float(ambiguity_level),
                "measurement_reset_prob": float(measurement_reset_prob),
                "cases": int(total),
                "dominant_policy": str(dominant),
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


def _save_dominant_policy_heatmap(
    rows: Sequence[dict[str, Any]],
    *,
    ambiguity_levels: Sequence[float],
    measurement_reset_probs: Sequence[float],
    filename: str,
) -> Path:
    path = ensure_plot_root() / filename
    ambiguity_values = sorted({float(value) for value in ambiguity_levels})
    meas_values = sorted({float(value) for value in measurement_reset_probs})
    data = np.full((len(ambiguity_values), len(meas_values)), np.nan, dtype=float)
    policies = ("C1", "C2", "C3", "C3R")
    value_map = {policy: float(index) for index, policy in enumerate(policies)}
    for row in rows:
        a_idx = ambiguity_values.index(float(row["syndrome_ambiguity_level"]))
        m_idx = meas_values.index(float(row["measurement_reset_prob"]))
        data[a_idx, m_idx] = value_map[str(row["dominant_policy"])]
    cmap = matplotlib.colors.ListedColormap(["#54A24B", "#E45756", "#72B7B2", "#B279A2"])
    norm = matplotlib.colors.BoundaryNorm([index - 0.5 for index in range(len(policies) + 1)], cmap.N)
    fig, ax = plt.subplots(figsize=(max(6, len(meas_values) * 1.6), max(4, len(ambiguity_values) * 0.9)))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(meas_values)), [f"{value:.2f}" for value in meas_values])
    ax.set_yticks(np.arange(len(ambiguity_values)), [f"{value:.2f}" for value in ambiguity_values])
    ax.set_xlabel("Measurement/reset error probability")
    ax.set_ylabel("Syndrome ambiguity level")
    ax.set_title("Ambiguity + Measurement/Reset: Dominant Preferred Policy")
    cbar = fig.colorbar(im, ax=ax, ticks=[float(index) for index in range(len(policies))])
    cbar.ax.set_yticklabels(list(policies))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_policy_rate_heatmap(
    rows: Sequence[dict[str, Any]],
    *,
    ambiguity_levels: Sequence[float],
    measurement_reset_probs: Sequence[float],
    policy: str,
    filename: str,
) -> Path:
    path = ensure_plot_root() / filename
    ambiguity_values = sorted({float(value) for value in ambiguity_levels})
    meas_values = sorted({float(value) for value in measurement_reset_probs})
    data = np.zeros((len(ambiguity_values), len(meas_values)), dtype=float)
    for row in rows:
        a_idx = ambiguity_values.index(float(row["syndrome_ambiguity_level"]))
        m_idx = meas_values.index(float(row["measurement_reset_prob"]))
        data[a_idx, m_idx] = float(row[f"{policy}_rate"])
    fig, ax = plt.subplots(figsize=(max(6, len(meas_values) * 1.6), max(4, len(ambiguity_values) * 0.9)))
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0.0, vmax=max(1.0e-12, float(np.max(data))))
    ax.set_xticks(np.arange(len(meas_values)), [f"{value:.2f}" for value in meas_values])
    ax.set_yticks(np.arange(len(ambiguity_values)), [f"{value:.2f}" for value in ambiguity_values])
    ax.set_xlabel("Measurement/reset error probability")
    ax.set_ylabel("Syndrome ambiguity level")
    ax.set_title(f"Ambiguity + Measurement/Reset: {policy} Preferred-Rate Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{policy} preferred rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Ambiguity + Measurement/Reset Syndrome Regime Map",
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
            "## By Ambiguity Level And Measurement/Reset Error",
            "",
            _markdown_table(
                result["tables"]["by_ambiguity_and_measurement_reset"],
                [
                    "syndrome_ambiguity_level",
                    "measurement_error_prob",
                    "reset_error_prob",
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
            "## Preferred Policy Counts By Combined Stress Setting",
            "",
            _markdown_table(
                result["tables"]["preferred_policy_counts_by_combo"],
                [
                    "syndrome_ambiguity_level",
                    "measurement_reset_prob",
                    "cases",
                    "dominant_policy",
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
    ambiguity_levels: Sequence[float],
    measurement_reset_probs: Sequence[float],
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
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    combos = [
        (float(ambiguity_level), float(measurement_reset_prob))
        for ambiguity_level in ambiguity_levels
        for measurement_reset_prob in measurement_reset_probs
    ]
    for ambiguity_level, measurement_reset_prob in progress_iter(
        combos,
        total=len(combos),
        desc=f"{output_stem}: ambiguity/measurement combos",
        unit="combo",
    ):
        stress_cfg = SyndromeObservationConfig(
            observation_ratio=float(syndrome_obs_cfg.observation_ratio),
            noise_prob=float(syndrome_obs_cfg.noise_prob),
            ambiguity_level=float(ambiguity_level),
            measurement_error_prob=float(measurement_reset_prob),
            reset_error_prob=float(measurement_reset_prob),
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
            syndrome_obs_cfg=stress_cfg,
            c1_objective_tol=c1_objective_tol,
            c1_tie_break_requires_syndrome_consistent=c1_tie_break_requires_syndrome_consistent,
            experiment_config=experiment_config,
            output_stem=f"{output_stem}_a{ambiguity_level:.2f}_m{measurement_reset_prob:.2f}",
            max_workers=max_workers,
            resume=resume,
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
                "syndrome_ambiguity_level",
                "measurement_error_prob",
            ),
        ),
        safety_tolerance=float(regime_safety_tolerance),
        gain_tolerance=float(regime_gain_tolerance),
    )
    tables = {
        "by_ambiguity_level": _group_rows(rows, keys=("syndrome_ambiguity_level",)),
        "by_measurement_reset_prob": _group_rows(rows, keys=("measurement_error_prob",)),
        "by_ambiguity_and_measurement_reset": _group_rows(
            rows, keys=("syndrome_ambiguity_level", "measurement_error_prob", "reset_error_prob")
        ),
        "by_regime_cell": by_regime_cell,
        "preferred_policy_counts_by_combo": _preferred_policy_counts_by_combo(by_regime_cell),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
    }
    figures = {
        "dominant_policy_heatmap": str(
            _save_dominant_policy_heatmap(
                tables["preferred_policy_counts_by_combo"],
                ambiguity_levels=ambiguity_levels,
                measurement_reset_probs=measurement_reset_probs,
                filename=f"{plot_prefix}_preferred_policy_map.png",
            )
        ),
        "c3_rate_heatmap": str(
            _save_policy_rate_heatmap(
                tables["preferred_policy_counts_by_combo"],
                ambiguity_levels=ambiguity_levels,
                measurement_reset_probs=measurement_reset_probs,
                policy="C3",
                filename=f"{plot_prefix}_c3_rate_heatmap.png",
            )
        ),
        "c3r_rate_heatmap": str(
            _save_policy_rate_heatmap(
                tables["preferred_policy_counts_by_combo"],
                ambiguity_levels=ambiguity_levels,
                measurement_reset_probs=measurement_reset_probs,
                policy="C3R",
                filename=f"{plot_prefix}_c3r_rate_heatmap.png",
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
            "ambiguity_levels": [float(value) for value in ambiguity_levels],
            "measurement_reset_probs": [float(value) for value in measurement_reset_probs],
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
    parser = argparse.ArgumentParser(description="Run the ambiguity + measurement/reset syndrome regime map.")
    parser.add_argument("--config", default="experiment/ambiguity_measurement_syndrome_regime_map.yaml")
    parser.add_argument("--codes", default=None)
    parser.add_argument("--noise-families", default=None)
    parser.add_argument("--strengths", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--ambiguity-levels", default=None)
    parser.add_argument("--measurement-reset-probs", default=None)
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
    stress_cfg = dict(config.get("ambiguity_measurement_syndrome", {}))
    codes = _parse_csv(args.codes, str) or list(cfg.get("codes", DEFAULT_CODES))
    state_configs = {str(key): str(value) for key, value in dict(cfg.get("state_configs", {})).items()}
    kinds_by_code = {
        str(key): [str(item) for item in value]
        for key, value in dict(cfg.get("kinds_by_code", {})).items()
        if isinstance(value, (list, tuple))
    }
    noise_families = _parse_csv(args.noise_families, str) or [
        str(item) for item in stress_cfg.get("noise_families", cfg.get("noise_families", []))
    ]
    strengths = _parse_csv(args.strengths, float) or [
        float(item) for item in stress_cfg.get("strengths", cfg.get("strengths", []))
    ]
    depths = _parse_csv(args.depths, int) or [
        int(item) for item in stress_cfg.get("depths", cfg.get("depths", []))
    ]
    seeds = _parse_csv(args.seeds, int) or [
        int(item) for item in stress_cfg.get("seeds", cfg.get("seeds", []))
    ]
    ambiguity_levels = _parse_csv(args.ambiguity_levels, float) or [
        float(item) for item in stress_cfg.get("ambiguity_levels", [0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    measurement_reset_probs = _parse_csv(args.measurement_reset_probs, float) or [
        float(item) for item in stress_cfg.get("measurement_reset_probs", [0.0, 0.03, 0.05, 0.10])
    ]
    c2 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c2", {})).items()})
    c3 = PolicyScoreConfig(**{key: float(value) for key, value in dict(cfg.get("c3", {})).items()})
    c3r = C3RPolicyConfig(**{key: float(value) for key, value in dict(cfg.get("c3r", {})).items()})
    syndrome_obs = SyndromeObservationConfig(
        **{key: float(value) for key, value in dict(cfg.get("syndrome_observation", {})).items()}
    )
    c1_cfg = dict(cfg.get("c1", {}))
    stem = resolve_output_stem(config, "ambiguity_measurement_syndrome_regime_map", args.output_stem)
    plot_prefix = args.plot_prefix or stem
    result = run(
        codes=codes,
        state_configs=state_configs,
        kinds_by_code=kinds_by_code,
        noise_families=noise_families,
        strengths=strengths,
        depths=depths,
        seeds=seeds,
        ambiguity_levels=ambiguity_levels,
        measurement_reset_probs=measurement_reset_probs,
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
        regime_safety_tolerance=float(stress_cfg.get("safety_tolerance", 0.02)),
        regime_gain_tolerance=float(stress_cfg.get("gain_tolerance", 0.005)),
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

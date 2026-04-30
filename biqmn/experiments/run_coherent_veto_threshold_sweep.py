"""Threshold sweep for coherent-regime detector / veto rules.

The goal is not to promote relational recovery B as the final selector in
coherent regimes. Instead, this sweep evaluates whether the trajectory score is
useful as:

* V1 soft veto: flag structural risk but keep A
* V2 hard veto: abstain when structural risk is too large
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coherent_veto_common import (
    ensure_plot_root,
    enrich_rows,
    quantile_thresholds,
    select_operating_point,
    veto_metrics,
)
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_coherent_veto_analysis import PAIR_ORDER, _subset
from .run_encoded_qec_baseline import _markdown_table, _write_csv


def _sweep_rows(
    rows: Sequence[dict[str, Any]],
    *,
    quantiles: Sequence[float],
) -> list[dict[str, Any]]:
    sweep: list[dict[str, Any]] = []
    thresholds = quantile_thresholds(rows, quantiles)
    scopes = [("overall", None, None), *[(f"{code}/{family}", code, family) for code, family in PAIR_ORDER]]
    for q, threshold in thresholds:
        for scope, code_type, noise_family in scopes:
            subset = list(rows) if code_type is None else _subset(rows, code_type=code_type, noise_family=noise_family)
            for mode in ("V1", "V2"):
                metrics = veto_metrics(subset, threshold=float(threshold), mode=mode)
                metrics["threshold_quantile"] = float(q)
                metrics["scope"] = scope
                metrics["code_type"] = code_type
                metrics["noise_family"] = noise_family
                sweep.append(metrics)
    return sweep


def _false_safe_comparison(
    rows: Sequence[dict[str, Any]],
    operating_points: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    comparison = []
    scopes = [("overall", None, None), *[(f"{code}/{family}", code, family) for code, family in PAIR_ORDER]]
    for scope, code_type, noise_family in scopes:
        subset = list(rows) if code_type is None else _subset(rows, code_type=code_type, noise_family=noise_family)
        base_false_safe = float(sum(1.0 for row in subset if bool(row["false_safe_flag_A"])) / len(subset)) if subset else 0.0
        row = {
            "scope": scope,
            "code_type": code_type,
            "noise_family": noise_family,
            "cases": len(subset),
            "false_safe_rate_A": base_false_safe,
        }
        v1 = operating_points.get("V1")
        v2 = operating_points.get("V2")
        if v1 is not None:
            v1_metrics = veto_metrics(subset, threshold=float(v1["threshold"]), mode="V1")
            row["V1_threshold_quantile"] = float(v1["threshold_quantile"])
            row["V1_false_safe_after_flag"] = float(v1_metrics["false_safe_rate_after_veto"])
            row["V1_risky_case_capture_rate"] = float(v1_metrics["risky_case_capture_rate"])
            row["V1_safe_case_retention_rate"] = float(v1_metrics["safe_case_retention_rate"])
        if v2 is not None:
            v2_metrics = veto_metrics(subset, threshold=float(v2["threshold"]), mode="V2")
            row["V2_threshold_quantile"] = float(v2["threshold_quantile"])
            row["V2_false_safe_after_abstain"] = float(v2_metrics["accepted_false_safe_rate"])
            row["V2_abstain_rate"] = float(v2_metrics["abstain_rate"])
            row["V2_risky_case_capture_rate"] = float(v2_metrics["risky_case_capture_rate"])
            row["V2_safe_case_retention_rate"] = float(v2_metrics["safe_case_retention_rate"])
        comparison.append(row)
    return comparison


def _save_threshold_sweep_figure(rows: Sequence[dict[str, Any]], stem: str) -> Path:
    plot_root = ensure_plot_root()
    path = plot_root / f"{stem}_threshold_sweep.png"
    overall_v1 = [row for row in rows if str(row["scope"]) == "overall" and str(row["mode"]) == "V1"]
    overall_v2 = [row for row in rows if str(row["scope"]) == "overall" and str(row["mode"]) == "V2"]
    overall_v1.sort(key=lambda row: float(row["threshold_quantile"]))
    overall_v2.sort(key=lambda row: float(row["threshold_quantile"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mode, subset in zip(axes, ("V1", "V2"), (overall_v1, overall_v2)):
        xs = [float(row["threshold_quantile"]) for row in subset]
        ax.plot(xs, [float(row["risky_case_capture_rate"]) for row in subset], marker="o", label="risky capture")
        ax.plot(xs, [float(row["safe_case_retention_rate"]) for row in subset], marker="o", label="safe retention")
        ax.plot(xs, [float(row["abstain_rate"]) for row in subset], marker="o", label="abstain")
        ax.set_title(f"Figure B. Threshold Sweep ({mode})")
        ax.set_xlabel("Threshold quantile")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_false_safe_figure(rows: Sequence[dict[str, Any]], stem: str) -> Path:
    plot_root = ensure_plot_root()
    path = plot_root / f"{stem}_false_safe_before_after.png"
    labels = [str(row["scope"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width, [float(row["false_safe_rate_A"]) for row in rows], width=width, label="A baseline", color="tab:blue")
    ax.bar(
        x,
        [float(row.get("V1_false_safe_after_flag", 0.0)) for row in rows],
        width=width,
        label="V1 residual false-safe",
        color="tab:orange",
    )
    ax.bar(
        x + width,
        [float(row.get("V2_false_safe_after_abstain", 0.0)) for row in rows],
        width=width,
        label="V2 accepted false-safe",
        color="tab:green",
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title("Figure C. False-Safe Before / After Veto")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Coherent Veto Threshold Sweep",
        "",
        "## Operating Points",
        "",
        _markdown_table(result["tables"]["operating_points"], [
            "mode",
            "scope",
            "threshold_quantile",
            "threshold",
            "risky_case_capture_rate",
            "safe_case_retention_rate",
            "abstain_rate",
            "false_safe_rate_after_veto",
            "accepted_false_safe_rate",
            "selection_score",
        ]),
        "",
        "## Overall Threshold Sweep",
        "",
        _markdown_table(result["tables"]["overall_threshold_sweep"], [
            "mode",
            "threshold_quantile",
            "threshold",
            "flag_rate",
            "risky_case_capture_rate",
            "safe_case_retention_rate",
            "abstain_rate",
            "false_safe_rate_after_veto",
            "accepted_false_safe_rate",
        ]),
        "",
        "## False-Safe Comparison",
        "",
        _markdown_table(result["tables"]["false_safe_comparison"], [
            "scope",
            "cases",
            "false_safe_rate_A",
            "V1_threshold_quantile",
            "V1_false_safe_after_flag",
            "V2_threshold_quantile",
            "V2_false_safe_after_abstain",
            "V2_abstain_rate",
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
    rows: Sequence[dict[str, Any]],
    quantiles: Sequence[float],
    fidelity_margin: float = 0.01,
    negative_gain_eps: float = 0.0,
    score_field: str = "clean_observed_distance",
    plot_stem: str = "coherent_veto_threshold_sweep",
) -> dict[str, Any]:
    enriched = enrich_rows(
        rows,
        fidelity_margin=float(fidelity_margin),
        score_field=score_field,
        negative_gain_eps=float(negative_gain_eps),
    )
    sweep_rows = _sweep_rows(enriched, quantiles=quantiles)
    operating_points = {
        "V1": select_operating_point(
            [row for row in sweep_rows if str(row["scope"]) == "overall"],
            mode="V1",
        ),
        "V2": select_operating_point(
            [row for row in sweep_rows if str(row["scope"]) == "overall"],
            mode="V2",
        ),
    }
    operating_rows = [row for row in operating_points.values() if row is not None]
    false_safe_rows = _false_safe_comparison(enriched, operating_points)
    threshold_path = _save_threshold_sweep_figure(sweep_rows, plot_stem)
    false_safe_path = _save_false_safe_figure(false_safe_rows, plot_stem)
    result = {
        "selection": {
            "quantiles": [float(item) for item in quantiles],
            "fidelity_margin": float(fidelity_margin),
            "negative_gain_eps": float(negative_gain_eps),
            "score_field": score_field,
            "operating_point_heuristic": {
                "V1": "maximize risky_capture + safe_retention - false_safe_after_flag",
                "V2": "maximize risky_capture + safe_retention - abstain - accepted_false_safe",
            },
        },
        "rows": sweep_rows,
        "tables": {
            "operating_points": operating_rows,
            "overall_threshold_sweep": [
                row for row in sweep_rows if str(row["scope"]) == "overall"
            ],
            "by_pair_threshold_sweep": [
                row for row in sweep_rows if str(row["scope"]) != "overall"
            ],
            "false_safe_comparison": false_safe_rows,
        },
        "figures": {
            "figure_b_threshold_sweep": str(threshold_path),
            "figure_c_false_safe_before_after": str(false_safe_path),
        },
    }
    result["markdown"] = _build_markdown(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep coherent detector/veto thresholds and export figures.")
    parser.add_argument("--config", default="experiment/coherent_veto_analysis.yaml")
    parser.add_argument("--source-stem", default=None)
    parser.add_argument("--quantiles", default=None)
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--fidelity-margin", type=float, default=None)
    parser.add_argument("--negative-gain-eps", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(experiment_config=args.config)
    veto_cfg = dict(base_config.get("coherent_veto", {}))
    if args.quantiles:
        quantiles = [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
    else:
        quantiles = [float(item) for item in veto_cfg.get("quantiles", [0.70, 0.80, 0.90, 0.95])]

    from .coherent_veto_common import load_result_rows

    source_rows = load_result_rows(str(args.source_stem or veto_cfg.get("source_stem", "encoded_coherent_validation")))
    stem_base = args.output_stem or "coherent_veto_threshold_sweep"
    result = run(
        rows=source_rows,
        quantiles=quantiles,
        fidelity_margin=float(args.fidelity_margin or veto_cfg.get("fidelity_margin", 0.01)),
        negative_gain_eps=float(args.negative_gain_eps or veto_cfg.get("negative_gain_eps", 0.0)),
        score_field=str(args.score_field or veto_cfg.get("score_field", "clean_observed_distance")),
        plot_stem=stem_base,
    )
    json_path = write_json_result(result, stem_base)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem_base}.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem_base}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem_base}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["tables"]["operating_points"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

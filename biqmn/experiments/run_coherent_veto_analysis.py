"""Coherent-regime veto analysis and figure export.

This runner consumes the coherent-only encoded validation payload and reframes
it in the way outlined in ``LetsDoThis.md``:

* A remains the default recovery baseline
* the relational criterion is interpreted as a detector / veto signal
* coherent failure cases are surfaced explicitly
* paper-facing figures are emitted alongside CSV / JSON / Markdown summaries
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
    group_by,
    load_result_rows,
    select_negative_gain_cases,
    summarize_rows,
)
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_encoded_qec_baseline import _markdown_table, _write_csv


PAIR_ORDER = (
    ("bitflip", "coherent_x"),
    ("phaseflip", "coherent_z"),
)


def _subset(rows: Sequence[dict[str, Any]], *, code_type: str, noise_family: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("code_type")) == str(code_type)
        and str(row.get("noise_family")) == str(noise_family)
    ]


def _summary_row(rows: Sequence[dict[str, Any]], *, code_type: str | None = None, noise_family: str | None = None) -> dict[str, Any]:
    row = summarize_rows(rows)
    if code_type is not None:
        row["code_type"] = str(code_type)
    if noise_family is not None:
        row["noise_family"] = str(noise_family)
    return row


def _group_summary(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        grouped.setdefault(group_key, []).append(dict(row))
    out: list[dict[str, Any]] = []
    for group_key in sorted(grouped.keys()):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(summarize_rows(grouped[group_key]))
        out.append(entry)
    return out


def _project_case(row: dict[str, Any], *, tag: str) -> dict[str, Any]:
    return {
        "tag": tag,
        "experiment_id": row["experiment_id"],
        "code_type": row["code_type"],
        "noise_family": row["noise_family"],
        "noise_strength": row["noise_strength"],
        "noise_depth": row["noise_depth"],
        "seed": row["seed"],
        "syndrome": row["syndrome"],
        "reason_C": row["reason_C"],
        "traj_inconsistency_score": row["traj_inconsistency_score"],
        "fidelity_before": row["fidelity_before"],
        "fidelity_after_A": row["fidelity_after_A"],
        "fidelity_after_B": row["fidelity_after_B"],
        "fidelity_after_C": row["fidelity_after_C"],
        "gain_A": row["gain_A"],
        "gain_B": row["gain_B"],
        "gain_C": row["gain_C"],
        "harmful_relational_gap": row["harmful_relational_gap"],
        "admissible_A": row["admissible_A"],
        "admissible_B": row["admissible_B"],
        "false_safe_flag_A": row["false_safe_flag_A"],
        "decision_disagreement_AB": row["decision_disagreement_AB"],
    }


def _build_casebook(rows: Sequence[dict[str, Any]], *, case_limit: int) -> dict[str, list[dict[str, Any]]]:
    phase_negative = select_negative_gain_cases(rows, family="coherent_z", code="phaseflip", limit=case_limit)
    phase_high_score = sorted(
        _subset(rows, code_type="phaseflip", noise_family="coherent_z"),
        key=lambda row: (
            -float(row["traj_inconsistency_score"]),
            float(row["gain_B"]),
        ),
    )[: int(case_limit)]
    bitflip_agreement = sorted(
        [
            row for row in _subset(rows, code_type="bitflip", noise_family="coherent_x")
            if abs(float(row["gain_B"]) - float(row["gain_A"])) <= 1.0e-9
        ],
        key=lambda row: -float(row["traj_inconsistency_score"]),
    )[: int(case_limit)]
    syndrome_false_safe = sorted(
        [row for row in rows if bool(row["false_safe_flag_A"])],
        key=lambda row: (
            -float(row["traj_inconsistency_score"]),
            float(row["gain_B"]),
        ),
    )[: int(case_limit)]
    return {
        "negative_gain_phase_coherent": [_project_case(row, tag="negative_gain_phase_coherent") for row in phase_negative],
        "high_risk_phase_coherent": [_project_case(row, tag="high_risk_phase_coherent") for row in phase_high_score],
        "bitflip_agreement_reference": [_project_case(row, tag="bitflip_agreement_reference") for row in bitflip_agreement],
        "syndrome_false_safe_examples": [_project_case(row, tag="syndrome_false_safe_examples") for row in syndrome_false_safe],
    }


def _save_score_vs_gain_figure(rows: Sequence[dict[str, Any]], stem: str) -> Path:
    plot_root = ensure_plot_root()
    path = plot_root / f"{stem}_score_vs_gain_B.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (code_type, noise_family) in zip(axes, PAIR_ORDER):
        subset = _subset(rows, code_type=code_type, noise_family=noise_family)
        xs = [float(row["traj_inconsistency_score"]) for row in subset]
        ys = [float(row["gain_B"]) for row in subset]
        colors = ["tab:red" if bool(row["negative_gain_B"]) else "tab:blue" for row in subset]
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=20, edgecolors="none")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"{code_type} / {noise_family}")
        ax.set_xlabel("Trajectory inconsistency score")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("fid_gain_B")
    fig.suptitle("Figure A. Inconsistency Score vs Relational Gain")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_failure_cases_figure(cases: Sequence[dict[str, Any]], stem: str) -> Path:
    plot_root = ensure_plot_root()
    path = plot_root / f"{stem}_representative_failures.png"
    focus = list(cases[:3])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not focus:
        ax.text(0.5, 0.5, "No representative failure cases found.", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [
            f"s={case['noise_strength']:.2f}, d={int(case['noise_depth'])}, seed={int(case['seed'])}"
            for case in focus
        ]
        y = np.arange(len(focus))
        width = 0.22
        ax.barh(y - width, [float(case["gain_A"]) for case in focus], height=width, label="A", color="tab:blue")
        ax.barh(y, [float(case["gain_B"]) for case in focus], height=width, label="B", color="tab:red")
        ax.barh(y + width, [float(case["gain_C"]) for case in focus], height=width, label="C", color="tab:green")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Fidelity gain")
        ax.set_title("Figure D. Representative Failure Cases")
        ax.legend()
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Coherent Veto Analysis",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "score_mean",
            "gain_B_mean",
            "negative_gain_rate_B",
            "false_safe_rate_A",
            "false_safe_rate_B",
            "decision_disagreement_rate_AB",
            "corr_score_vs_gain_B",
            "corr_score_vs_harmful_gap",
        ]),
        "",
        "## By Pair",
        "",
        _markdown_table(result["tables"]["by_pair"], [
            "code_type",
            "noise_family",
            "cases",
            "score_mean",
            "gain_B_mean",
            "negative_gain_rate_B",
            "false_safe_rate_A",
            "decision_disagreement_rate_AB",
            "corr_score_vs_gain_B",
        ]),
        "",
        "## By Noise Strength",
        "",
        _markdown_table(result["tables"]["by_noise_strength"], [
            "noise_strength",
            "cases",
            "score_mean",
            "gain_B_mean",
            "negative_gain_rate_B",
            "false_safe_rate_A",
            "decision_disagreement_rate_AB",
        ]),
        "",
        "## By Noise Depth",
        "",
        _markdown_table(result["tables"]["by_noise_depth"], [
            "noise_depth",
            "cases",
            "score_mean",
            "gain_B_mean",
            "negative_gain_rate_B",
            "false_safe_rate_A",
            "decision_disagreement_rate_AB",
        ]),
        "",
        "## Figures",
        "",
    ]
    for name, path in result["figures"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend([
        "",
        "## Negative-Gain Phase-Coherent Cases",
        "",
    ])
    if result["negative_gain_cases"]:
        lines.append(
            _markdown_table(result["negative_gain_cases"], [
                "experiment_id",
                "noise_strength",
                "noise_depth",
                "seed",
                "traj_inconsistency_score",
                "gain_A",
                "gain_B",
                "gain_C",
                "reason_C",
            ])
        )
    else:
        lines.append("No negative-gain coherent-z cases found.")
    return "\n".join(lines) + "\n"


def _build_negative_cases_markdown(cases: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Coherent Negative-Gain Cases",
        "",
    ]
    if cases:
        lines.append(
            _markdown_table(list(cases), [
                "experiment_id",
                "code_type",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "seed",
                "traj_inconsistency_score",
                "gain_A",
                "gain_B",
                "gain_C",
                "admissible_A",
                "admissible_B",
                "reason_C",
            ])
        )
    else:
        lines.append("No cases matched.")
    return "\n".join(lines) + "\n"


def _build_casebook_markdown(casebook: dict[str, list[dict[str, Any]]]) -> str:
    title_map = {
        "negative_gain_phase_coherent": "Negative-Gain Phase-Coherent Cases",
        "high_risk_phase_coherent": "High-Risk Phase-Coherent Cases",
        "bitflip_agreement_reference": "Bitflip Agreement Reference",
        "syndrome_false_safe_examples": "Syndrome False-Safe Examples",
    }
    lines = ["# Coherent Veto Casebook", ""]
    for key, rows in casebook.items():
        lines.extend([f"## {title_map.get(key, key)}", ""])
        if rows:
            lines.append(
                _markdown_table(rows, [
                    "experiment_id",
                    "code_type",
                    "noise_family",
                    "noise_strength",
                    "noise_depth",
                    "seed",
                    "traj_inconsistency_score",
                    "gain_A",
                    "gain_B",
                    "gain_C",
                    "reason_C",
                ])
            )
        else:
            lines.append("No cases matched.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    source_stem: str = "encoded_coherent_validation",
    score_field: str = "clean_observed_distance",
    fidelity_margin: float = 0.01,
    negative_gain_eps: float = 0.0,
    case_limit: int = 6,
    plot_stem: str = "coherent_veto_analysis",
) -> dict[str, Any]:
    source_rows = list(rows) if rows is not None else load_result_rows(source_stem)
    enriched = enrich_rows(
        source_rows,
        fidelity_margin=float(fidelity_margin),
        score_field=score_field,
        negative_gain_eps=float(negative_gain_eps),
    )
    overall = summarize_rows(enriched)
    by_pair = []
    for code_type, noise_family in PAIR_ORDER:
        subset = _subset(enriched, code_type=code_type, noise_family=noise_family)
        if subset:
            by_pair.append(_summary_row(subset, code_type=code_type, noise_family=noise_family))
    by_noise_strength = _group_summary(enriched, keys=("noise_strength",))
    by_noise_depth = _group_summary(enriched, keys=("noise_depth",))
    negative_cases = [
        _project_case(row, tag="negative_gain_phase_coherent")
        for row in select_negative_gain_cases(
            enriched,
            family="coherent_z",
            code="phaseflip",
            limit=int(case_limit),
        )
    ]
    casebook = _build_casebook(enriched, case_limit=int(case_limit))
    score_vs_gain_path = _save_score_vs_gain_figure(enriched, plot_stem)
    failure_cases_path = _save_failure_cases_figure(negative_cases, plot_stem)
    result = {
        "source_stem": source_stem,
        "selection": {
            "score_field": score_field,
            "fidelity_margin": float(fidelity_margin),
            "negative_gain_eps": float(negative_gain_eps),
            "case_limit": int(case_limit),
        },
        "overall": overall,
        "tables": {
            "by_pair": by_pair,
            "by_noise_strength": by_noise_strength,
            "by_noise_depth": by_noise_depth,
        },
        "rows": enriched,
        "negative_gain_cases": negative_cases,
        "casebook": casebook,
        "figures": {
            "figure_a_score_vs_gain_B": str(score_vs_gain_path),
            "figure_d_representative_failures": str(failure_cases_path),
        },
    }
    result["markdown"] = _build_summary_markdown(result)
    result["negative_gain_markdown"] = _build_negative_cases_markdown(negative_cases)
    result["casebook_markdown"] = _build_casebook_markdown(casebook)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze coherent-regime veto signals and export figures.")
    parser.add_argument("--config", default="experiment/coherent_veto_analysis.yaml")
    parser.add_argument("--source-stem", default=None)
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--fidelity-margin", type=float, default=None)
    parser.add_argument("--negative-gain-eps", type=float, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(experiment_config=args.config)
    veto_cfg = dict(base_config.get("coherent_veto", {}))
    stem_base = args.output_stem or "coherent_veto_analysis"
    result = run(
        source_stem=str(args.source_stem or veto_cfg.get("source_stem", "encoded_coherent_validation")),
        score_field=str(args.score_field or veto_cfg.get("score_field", "clean_observed_distance")),
        fidelity_margin=float(args.fidelity_margin or veto_cfg.get("fidelity_margin", 0.01)),
        negative_gain_eps=float(args.negative_gain_eps or veto_cfg.get("negative_gain_eps", 0.0)),
        case_limit=int(args.case_limit or veto_cfg.get("case_limit", 6)),
        plot_stem=stem_base,
    )
    summary_json_path = write_json_result(result, f"{stem_base}_summary")
    negative_json_path = write_json_result(
        {
            "source_stem": result["source_stem"],
            "negative_gain_cases": result["negative_gain_cases"],
        },
        "coherent_negative_gain_cases",
    )
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem_base}_raw.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem_base}_{name}.csv", rows)
    _write_csv(tables_dir / "coherent_negative_gain_cases.csv", result["negative_gain_cases"])
    markdown_path = tables_dir / f"{stem_base}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    negative_md_path = tables_dir / "coherent_negative_gain_cases.md"
    negative_md_path.write_text(result["negative_gain_markdown"], encoding="utf-8")
    casebook_md_path = tables_dir / "coherent_veto_casebook.md"
    casebook_md_path.write_text(result["casebook_markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={summary_json_path}")
    print(f"saved_negative_json={negative_json_path}")
    print(f"saved_markdown={markdown_path}")
    print(f"saved_negative_markdown={negative_md_path}")
    print(f"saved_casebook_markdown={casebook_md_path}")


if __name__ == "__main__":
    main()

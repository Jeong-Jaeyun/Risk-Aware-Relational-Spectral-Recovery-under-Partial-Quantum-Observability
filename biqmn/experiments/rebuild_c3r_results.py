from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .common import RESULT_ROOT, to_serializable
from .run_encoded_qec_baseline import _write_csv
from .run_hybrid_c123_baseline import (
    _aggregate_rows,
    _c3r_uncertainty_bin_rows,
    _group_rows,
    _reason_rows,
    enrich_c3r_row,
)
from .run_hybrid_c123_regime_map import (
    _annotate_regime_cells,
    _build_markdown as _build_hybrid_markdown,
    _group_preferred_policy,
    _reason_summary,
)
from .run_partial_syndrome_baseline import (
    _build_markdown as _build_partial_markdown,
    _preferred_policy_counts_by_ratio,
)
from .run_noisy_syndrome_baseline import (
    _build_markdown as _build_noisy_markdown,
    _preferred_policy_counts_by_noise,
)
from .run_partial_noisy_syndrome_regime_map import (
    _build_markdown as _build_partial_noisy_markdown,
    _preferred_policy_counts_by_combo as _preferred_policy_counts_by_partial_noisy_combo,
)
from .run_ambiguity_measurement_syndrome_regime_map import (
    _build_markdown as _build_ambiguity_markdown,
    _preferred_policy_counts_by_combo as _preferred_policy_counts_by_ambiguity_combo,
)


DEFAULT_STEMS = (
    "hybrid_c123_regime_map_c3r_260427_seed10",
    "partial_syndrome_c3r_260427_seed10",
    "noisy_syndrome_c3r_260427_seed10",
    "partial_noisy_syndrome_c3r_260427_seed10",
    "ambiguity_measurement_c3r_260427_seed10",
)


def _infer_kind(stem: str, payload: dict[str, Any]) -> str:
    grid = dict(payload.get("grid", {}))
    if "ambiguity_levels" in grid:
        return "ambiguity"
    if "observation_ratios" in grid and "syndrome_noise_probs" in grid:
        return "partial_noisy"
    if "observation_ratios" in grid:
        return "partial"
    if "syndrome_noise_probs" in grid:
        return "noisy"
    if "hybrid_c123_regime_map" in stem:
        return "hybrid"
    raise ValueError(f"Cannot infer C3R result kind for {stem!r}")


def _figures_for_existing(payload: dict[str, Any]) -> dict[str, Any]:
    figures = payload.get("figures")
    return dict(figures) if isinstance(figures, dict) else {}


def _rebuild_hybrid(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selection = dict(payload.get("selection_rules", {}))
    safety_tolerance = float(selection.get("regime_safety_tolerance", 0.02))
    gain_tolerance = float(selection.get("regime_gain_tolerance", 0.005))
    by_regime_cell = _annotate_regime_cells(
        _group_rows(rows, keys=("code_family", "noise_family", "noise_strength", "noise_depth")),
        safety_tolerance=safety_tolerance,
        gain_tolerance=gain_tolerance,
    )
    tables = {
        "by_code": _group_rows(rows, keys=("code_family",)),
        "by_noise_family": _group_rows(rows, keys=("noise_family",)),
        "by_noise_strength": _group_rows(rows, keys=("noise_strength",)),
        "by_noise_depth": _group_rows(rows, keys=("noise_depth",)),
        "by_code_and_noise_family": _group_rows(rows, keys=("code_family", "noise_family")),
        "by_syndrome_obs_ratio": _group_rows(rows, keys=("syndrome_observation_ratio",)),
        "by_code_and_obs_ratio": _group_rows(rows, keys=("code_family", "syndrome_observation_ratio")),
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
    result = dict(payload)
    result.update({"overall": _aggregate_rows(rows), "rows": rows, "tables": tables})
    result["figures"] = _figures_for_existing(payload)
    result["markdown"] = _build_hybrid_markdown(result)
    return result


def _rebuild_partial(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        safety_tolerance=0.02,
        gain_tolerance=0.005,
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
    result = dict(payload)
    result.update({"overall": _aggregate_rows(rows), "rows": rows, "tables": tables})
    result["figures"] = _figures_for_existing(payload)
    result["markdown"] = _build_partial_markdown(result)
    return result


def _rebuild_noisy(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime_cell = _annotate_regime_cells(
        _group_rows(
            rows,
            keys=(
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "syndrome_noise_prob",
            ),
        ),
        safety_tolerance=0.02,
        gain_tolerance=0.005,
    )
    tables = {
        "by_syndrome_noise_prob": _group_rows(rows, keys=("syndrome_noise_prob",)),
        "by_code_and_syndrome_noise": _group_rows(rows, keys=("code_family", "syndrome_noise_prob")),
        "by_code_noise_and_syndrome_noise": _group_rows(
            rows, keys=("code_family", "noise_family", "syndrome_noise_prob")
        ),
        "by_regime_cell": by_regime_cell,
        "preferred_policy_counts_by_noise": _preferred_policy_counts_by_noise(by_regime_cell),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
    }
    result = dict(payload)
    result.update({"overall": _aggregate_rows(rows), "rows": rows, "tables": tables})
    result["figures"] = _figures_for_existing(payload)
    result["markdown"] = _build_noisy_markdown(result)
    return result


def _rebuild_partial_noisy(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime_cell = _annotate_regime_cells(
        _group_rows(
            rows,
            keys=(
                "code_family",
                "noise_family",
                "noise_strength",
                "noise_depth",
                "syndrome_observation_ratio",
                "syndrome_noise_prob",
            ),
        ),
        safety_tolerance=0.02,
        gain_tolerance=0.005,
    )
    tables = {
        "by_syndrome_obs_ratio": _group_rows(rows, keys=("syndrome_observation_ratio",)),
        "by_syndrome_noise_prob": _group_rows(rows, keys=("syndrome_noise_prob",)),
        "by_obs_ratio_and_noise_prob": _group_rows(
            rows, keys=("syndrome_observation_ratio", "syndrome_noise_prob")
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
        "preferred_policy_counts_by_combo": _preferred_policy_counts_by_partial_noisy_combo(by_regime_cell),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
    }
    result = dict(payload)
    result.update({"overall": _aggregate_rows(rows), "rows": rows, "tables": tables})
    result["figures"] = _figures_for_existing(payload)
    result["markdown"] = _build_partial_noisy_markdown(result)
    return result


def _rebuild_ambiguity(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        safety_tolerance=0.02,
        gain_tolerance=0.005,
    )
    tables = {
        "by_ambiguity_level": _group_rows(rows, keys=("syndrome_ambiguity_level",)),
        "by_measurement_reset_prob": _group_rows(rows, keys=("measurement_error_prob",)),
        "by_ambiguity_and_measurement_reset": _group_rows(
            rows,
            keys=("syndrome_ambiguity_level", "measurement_error_prob", "reset_error_prob"),
        ),
        "by_regime_cell": by_regime_cell,
        "preferred_policy_counts_by_combo": _preferred_policy_counts_by_ambiguity_combo(by_regime_cell),
        "c3r_by_uncertainty_bin": _c3r_uncertainty_bin_rows(rows),
    }
    result = dict(payload)
    result.update({"overall": _aggregate_rows(rows), "rows": rows, "tables": tables})
    result["figures"] = _figures_for_existing(payload)
    result["markdown"] = _build_ambiguity_markdown(result)
    return result


def rebuild_payload(stem: str, payload: dict[str, Any]) -> dict[str, Any]:
    fidelity_margin = float(payload.get("grid", {}).get("fidelity_margin", 0.01))
    rows = [enrich_c3r_row(row, fidelity_margin=fidelity_margin) for row in payload["rows"]]
    kind = _infer_kind(stem, payload)
    if kind == "hybrid":
        return _rebuild_hybrid(payload, rows)
    if kind == "partial":
        return _rebuild_partial(payload, rows)
    if kind == "noisy":
        return _rebuild_noisy(payload, rows)
    if kind == "partial_noisy":
        return _rebuild_partial_noisy(payload, rows)
    if kind == "ambiguity":
        return _rebuild_ambiguity(payload, rows)
    raise AssertionError(kind)


def _write_result(stem: str, result: dict[str, Any]) -> None:
    raw_path = RESULT_ROOT / "raw" / f"{stem}.json"
    raw_path.write_text(
        json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tables_dir = RESULT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / f"{stem}_raw.csv", result["rows"])
    for name, rows in result["tables"].items():
        if isinstance(rows, list):
            _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    (tables_dir / f"{stem}.md").write_text(result["markdown"], encoding="utf-8")


def rebuild_stems(stems: Sequence[str]) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    for stem in stems:
        raw_path = RESULT_ROOT / "raw" / f"{stem}.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        result = rebuild_payload(stem, payload)
        _write_result(stem, result)
        rebuilt.append({"stem": stem, "overall": result["overall"]})
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild C3R result tables from existing raw rows.")
    parser.add_argument(
        "--stems",
        default=",".join(DEFAULT_STEMS),
        help="Comma-separated raw result stems under results/raw, without .json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stems = [item.strip() for item in str(args.stems).split(",") if item.strip()]
    rebuilt = rebuild_stems(stems)
    print(json.dumps(to_serializable(rebuilt), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

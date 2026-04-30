"""Threshold sweep for the coherent V1 detector branch."""
from __future__ import annotations

import argparse
import json

from .coherent_veto_common import load_result_rows
from .common import RESULT_ROOT, load_config, to_serializable, write_json_result
from .run_coherent_veto_threshold_sweep import run as run_veto_threshold_sweep
from .run_encoded_qec_baseline import _write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep coherent V1 thresholds.")
    parser.add_argument("--config", default="experiment/coherent_branch.yaml")
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
    config = load_config(experiment_config=args.config)
    sweep_cfg = dict(config.get("coherent_threshold_sweep", {}))
    quantiles = (
        [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
        if args.quantiles
        else [float(item) for item in sweep_cfg.get("quantiles", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85])]
    )
    source_rows = load_result_rows(str(args.source_stem or sweep_cfg.get("source_stem", "encoded_coherent_validation")))
    stem = args.output_stem or "coherent_threshold_sweep"
    result = run_veto_threshold_sweep(
        rows=source_rows,
        quantiles=quantiles,
        fidelity_margin=float(args.fidelity_margin or sweep_cfg.get("fidelity_margin", 0.01)),
        negative_gain_eps=float(args.negative_gain_eps or sweep_cfg.get("negative_gain_eps", 0.0)),
        score_field=str(args.score_field or sweep_cfg.get("score_field", "clean_observed_distance")),
        plot_stem=stem,
    )
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / "coherent_threshold_sweep.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / "coherent_threshold_sweep.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["tables"]["operating_points"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

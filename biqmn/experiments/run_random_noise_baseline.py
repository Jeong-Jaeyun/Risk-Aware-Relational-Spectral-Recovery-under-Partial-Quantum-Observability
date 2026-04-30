from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from .common import RESULT_ROOT, load_config, resolve_output_stem, to_serializable, write_json_result
from .run_detection_metrics import run as run_detection_metrics
from .run_recovery_objective import run as run_recovery_objective


def _parse_csv_list(raw: str | None, caster) -> list[Any] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []
    return [caster(item) for item in items]


def _sample_schedule(
    rng: np.random.Generator,
    *,
    n_system: int,
    kinds: Sequence[str],
    min_steps: int,
    max_steps: int,
    p_min: float,
    p_max: float,
) -> list[dict[str, Any]]:
    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError("Random-noise baseline requires 0 < min_steps <= max_steps.")
    if not kinds:
        raise ValueError("Random-noise baseline requires at least one noise kind.")
    steps = int(rng.integers(min_steps, max_steps + 1))
    schedule = []
    for _ in range(steps):
        schedule.append({
            "kind": str(rng.choice(kinds)),
            "qubit": int(rng.integers(0, n_system)),
            "p": float(rng.uniform(p_min, p_max)),
        })
    return schedule


def _schedule_signature(schedule: Sequence[dict[str, Any]]) -> str:
    kinds = [str(step["kind"]) for step in schedule]
    unique = sorted(set(kinds))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


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


def _mode_label(values: Sequence[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    label, count = counts.most_common(1)[0]
    return label, float(count / len(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "simulation_backend": None,
            "mean_schedule_length": None,
            "mean_total_strength": None,
            "mixed_schedule_rate": 0.0,
            "mean_trajectory_distance": None,
            "mean_detection_score": None,
            "mean_detection_score_max": None,
            "observed_admissible_rate": 0.0,
            "recovered_admissible_rate": 0.0,
            "recovery_nonworsen_rate": 0.0,
            "stage2_candidate_improvement_rate": 0.0,
            "final_stage_label": None,
            "final_stage_rate": 0.0,
        }

    backend_label, _ = _mode_label([str(row["simulation_backend"]) for row in rows])
    final_stage_label, final_stage_rate = _mode_label([str(row["final_stage"]) for row in rows])
    return {
        "cases": len(rows),
        "simulation_backend": backend_label,
        "mean_schedule_length": _mean([float(row["schedule_length"]) for row in rows]),
        "mean_total_strength": _mean([float(row["total_strength"]) for row in rows]),
        "mixed_schedule_rate": float(
            sum(1.0 if str(row["schedule_signature"]) == "mixed" else 0.0 for row in rows) / len(rows)
        ),
        "mean_trajectory_distance": _mean([float(row["trajectory_distance"]) for row in rows]),
        "mean_detection_score": _mean([float(row["score_mean"]) for row in rows]),
        "mean_detection_score_max": _mean([float(row["score_max"]) for row in rows]),
        "observed_admissible_rate": float(
            sum(1.0 if bool(row["observed_admissible"]) else 0.0 for row in rows) / len(rows)
        ),
        "recovered_admissible_rate": float(
            sum(1.0 if bool(row["recovered_admissible"]) else 0.0 for row in rows) / len(rows)
        ),
        "recovery_nonworsen_rate": float(
            sum(1.0 if bool(row["recovery_nonworsen"]) else 0.0 for row in rows) / len(rows)
        ),
        "stage2_candidate_improvement_rate": float(
            sum(
                1.0
                if float(row["stage2_candidate_objective_gain_vs_stage1"]) > 1.0e-9
                else 0.0
                for row in rows
            ) / len(rows)
        ),
        "final_stage_label": final_stage_label,
        "final_stage_rate": final_stage_rate,
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


def _build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Random Noise Baseline",
        "",
        "## Overall",
        "",
        _markdown_table([result["overall"]], [
            "cases",
            "simulation_backend",
            "mean_schedule_length",
            "mean_total_strength",
            "mixed_schedule_rate",
            "mean_trajectory_distance",
            "mean_detection_score",
            "observed_admissible_rate",
            "recovered_admissible_rate",
            "recovery_nonworsen_rate",
            "final_stage_label",
            "final_stage_rate",
        ]),
        "",
        "## By Schedule Signature",
        "",
        _markdown_table(result["tables"]["by_schedule_signature"], [
            "schedule_signature",
            "cases",
            "mean_schedule_length",
            "mean_total_strength",
            "mean_trajectory_distance",
            "mean_detection_score",
            "observed_admissible_rate",
            "recovered_admissible_rate",
            "recovery_nonworsen_rate",
        ]),
        "",
        "## By Schedule Length",
        "",
        _markdown_table(result["tables"]["by_schedule_length"], [
            "schedule_length",
            "cases",
            "mean_total_strength",
            "mean_trajectory_distance",
            "mean_detection_score",
            "observed_admissible_rate",
            "recovered_admissible_rate",
            "recovery_nonworsen_rate",
        ]),
    ]
    return "\n".join(lines) + "\n"


def run(
    base_config: dict[str, Any],
    *,
    n_samples: int,
    seed: int,
    min_steps: int,
    max_steps: int,
    kinds: Sequence[str],
    p_min: float,
    p_max: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_system = int(base_config.get("system", {}).get("n_qubits", 2))
    rows = []
    for index in range(int(n_samples)):
        schedule = _sample_schedule(
            rng,
            n_system=n_system,
            kinds=kinds,
            min_steps=int(min_steps),
            max_steps=int(max_steps),
            p_min=float(p_min),
            p_max=float(p_max),
        )
        config = deepcopy(base_config)
        config.setdefault("noise", {})["schedule"] = schedule
        detection = run_detection_metrics(config)
        recovery = run_recovery_objective(config)
        row = {
            "sample_index": int(index),
            "schedule_signature": _schedule_signature(schedule),
            "schedule_length": len(schedule),
            "total_strength": float(sum(float(step["p"]) for step in schedule)),
            "max_strength": float(max(float(step["p"]) for step in schedule)),
            "schedule": schedule,
            "simulation_backend": str(
                detection["meta"]["noisy"].get("simulation_backend", "unknown")
            ),
            "trajectory_distance": float(detection["summary"]["trajectory_distance"]),
            "score_mean": float(detection["summary"]["score_mean"]),
            "score_max": float(detection["summary"]["score_max"]),
            "observed_admissible": bool(recovery["summary"]["observed_admissible"]),
            "recovered_admissible": bool(recovery["summary"]["recovered_admissible"]),
            "clean_to_observed_distance": float(recovery["summary"]["clean_to_observed_distance"]),
            "clean_to_recovered_distance": float(recovery["summary"]["clean_to_recovered_distance"]),
            "recovery_nonworsen": bool(
                float(recovery["summary"]["clean_to_recovered_distance"])
                <= float(recovery["summary"]["clean_to_observed_distance"]) + 1.0e-12
            ),
            "stage1_best_label": str(recovery["summary"]["stage1_best_label"]),
            "reference_anchor_label": str(recovery["summary"]["reference_anchor_label"]),
            "stage2_candidate_objective_gain_vs_stage1": float(
                recovery["summary"]["stage2_candidate_objective_gain_vs_stage1"]
            ),
            "stage2_applied": bool(recovery["summary"]["stage2_applied"]),
            "final_stage": str(recovery["summary"]["final_stage"]),
        }
        rows.append(row)

    tables = {
        "by_schedule_signature": _group_rows(rows, keys=("schedule_signature",)),
        "by_schedule_length": _group_rows(rows, keys=("schedule_length",)),
    }
    result = {
        "grid": {
            "n_samples": int(n_samples),
            "seed": int(seed),
            "min_steps": int(min_steps),
            "max_steps": int(max_steps),
            "kinds": [str(item) for item in kinds],
            "p_min": float(p_min),
            "p_max": float(p_max),
        },
        "overall": _aggregate_rows(rows),
        "tables": tables,
        "rows": rows,
    }
    result["markdown"] = _build_markdown_report(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a qiskit-aer random-noise baseline over sampled CPTP schedules."
    )
    parser.add_argument("--config", default="experiment/random_noise_baseline.yaml")
    parser.add_argument("--state-config", default="states/null_dynamic.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--kinds", default=None)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(
        experiment_config=args.config,
        state_config=args.state_config,
    )
    random_cfg = dict(base_config.get("random_noise", {}))
    result = run(
        base_config,
        n_samples=int(args.n_samples or random_cfg.get("n_samples", 24)),
        seed=int(args.seed or random_cfg.get("seed", 21)),
        min_steps=int(args.min_steps or random_cfg.get("min_steps", 1)),
        max_steps=int(args.max_steps or random_cfg.get("max_steps", 3)),
        kinds=_parse_csv_list(args.kinds, str) or list(
            random_cfg.get(
                "kinds",
                [
                    "bitflip",
                    "phaseflip",
                    "dephasing",
                    "depolarizing",
                    "amplitude_damping",
                    "coherent_x",
                    "coherent_z",
                ],
            )
        ),
        p_min=float(args.p_min or random_cfg.get("p_min", 0.02)),
        p_max=float(args.p_max or random_cfg.get("p_max", 0.16)),
    )
    stem = resolve_output_stem(base_config, "random_noise_baseline", args.output_stem)
    json_path = write_json_result(result, stem)
    tables_dir = RESULT_ROOT / "tables"
    _write_csv(tables_dir / f"{stem}_rows.csv", result["rows"])
    for name, rows in result["tables"].items():
        _write_csv(tables_dir / f"{stem}_{name}.csv", rows)
    markdown_path = tables_dir / f"{stem}.md"
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(json.dumps(to_serializable(result["overall"]), indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_markdown={markdown_path}")


if __name__ == "__main__":
    main()

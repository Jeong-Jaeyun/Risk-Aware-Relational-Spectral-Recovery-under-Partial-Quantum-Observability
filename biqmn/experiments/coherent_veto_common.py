from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .common import RESULT_ROOT


PLOT_ROOT = RESULT_ROOT / "plots"


def load_result_rows(stem: str) -> list[dict[str, Any]]:
    path = RESULT_ROOT / "raw" / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result payload: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Result payload {path} does not contain a 'rows' list.")
    return [normalize_row(row) for row in rows]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _as_float(value: Any) -> float:
    return float(value)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    float_fields = [
        "noise_strength",
        "noise_depth",
        "fidelity_before",
        "fidelity_after_A",
        "fidelity_after_B",
        "fidelity_after_C",
        "gain_A",
        "gain_B",
        "gain_C",
        "clean_observed_distance",
        "traj_dist_A",
        "traj_dist_B",
        "objective_A",
        "objective_B",
        "objective_C",
        "hybrid_objective_gain_B_vs_A",
        "syndrome_mean_no_error",
    ]
    bool_fields = [
        "admissible_A",
        "admissible_B",
        "recovered_C_admissible",
        "syndrome_consistent",
        "trajectory_inconsistent",
        "syndrome_consistent_but_trajectory_inconsistent",
        "hybrid_use_relational",
        "A_recovery_nonworsen",
        "B_recovery_nonworsen",
        "C_recovery_nonworsen",
    ]
    for key in float_fields:
        if key in out:
            out[key] = _as_float(out[key])
    for key in bool_fields:
        if key in out:
            out[key] = _as_bool(out[key])
    return out


def enrich_rows(
    rows: Sequence[dict[str, Any]],
    *,
    fidelity_margin: float = 0.01,
    score_field: str = "clean_observed_distance",
    negative_gain_eps: float = 0.0,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = normalize_row(row)
        score = float(item[score_field])
        gain_a = float(item["gain_A"])
        gain_b = float(item["gain_B"])
        item["traj_inconsistency_score"] = score
        item["negative_gain_B"] = bool(gain_b < -float(negative_gain_eps))
        item["decision_disagreement_AB"] = bool(
            item["admissible_A"] != item["admissible_B"]
            or abs(float(item["fidelity_after_B"]) - float(item["fidelity_after_A"])) > float(fidelity_margin)
        )
        item["false_safe_flag_A"] = bool(item["syndrome_consistent_but_trajectory_inconsistent"])
        item["false_safe_flag_B"] = bool(item["negative_gain_B"] and item["trajectory_inconsistent"])
        item["risky_case"] = bool(item["negative_gain_B"])
        item["safe_case"] = bool(not item["risky_case"])
        item["harmful_relational_gap"] = float(gain_a - gain_b)
        enriched.append(item)
    return enriched


def rate(rows: Sequence[dict[str, Any]], predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return float(sum(1.0 if predicate(row) else 0.0 for row in rows) / len(rows))


def safe_rate(num: int, den: int) -> float:
    return 0.0 if den <= 0 else float(num / den)


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    dx = [x - x_mean for x in xs]
    dy = [y - y_mean for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = sum(a * a for a in dx) ** 0.5
    den_y = sum(b * b for b in dy) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return None
    return float(num / (den_x * den_y))


def group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"cases": 0}
    scores = [float(row["traj_inconsistency_score"]) for row in rows]
    gains_b = [float(row["gain_B"]) for row in rows]
    return {
        "cases": len(rows),
        "score_mean": float(mean(scores)),
        "score_max": float(max(scores)),
        "score_min": float(min(scores)),
        "gain_B_mean": float(mean(gains_b)),
        "negative_gain_rate_B": rate(rows, lambda row: bool(row["negative_gain_B"])),
        "false_safe_rate_A": rate(rows, lambda row: bool(row["false_safe_flag_A"])),
        "false_safe_rate_B": rate(rows, lambda row: bool(row["false_safe_flag_B"])),
        "decision_disagreement_rate_AB": rate(rows, lambda row: bool(row["decision_disagreement_AB"])),
        "admissible_A_rate": rate(rows, lambda row: bool(row["admissible_A"])),
        "admissible_B_rate": rate(rows, lambda row: bool(row["admissible_B"])),
        "corr_score_vs_gain_B": pearson_corr(scores, gains_b),
        "corr_score_vs_harmful_gap": pearson_corr(
            scores,
            [float(row["harmful_relational_gap"]) for row in rows],
        ),
    }


def select_negative_gain_cases(
    rows: Sequence[dict[str, Any]],
    *,
    family: str = "coherent_z",
    code: str = "phaseflip",
    limit: int = 6,
) -> list[dict[str, Any]]:
    subset = [
        dict(row)
        for row in rows
        if str(row["noise_family"]) == str(family)
        and str(row["code_type"]) == str(code)
        and bool(row["negative_gain_B"])
    ]
    subset.sort(
        key=lambda row: (
            float(row["gain_B"]),
            -float(row["traj_inconsistency_score"]),
            int(row["seed"]),
        )
    )
    return subset[: int(limit)]


def ensure_plot_root() -> Path:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    return PLOT_ROOT


def quantile_thresholds(rows: Sequence[dict[str, Any]], quantiles: Sequence[float]) -> list[tuple[float, float]]:
    import numpy as np

    values = np.asarray([float(row["traj_inconsistency_score"]) for row in rows], dtype=float)
    if values.size == 0:
        return []
    out: list[tuple[float, float]] = []
    for q in quantiles:
        qq = float(q)
        out.append((qq, float(np.quantile(values, qq))))
    return out


def veto_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    mode: str,
) -> dict[str, Any]:
    rows = list(rows)
    flagged = [bool(float(row["traj_inconsistency_score"]) >= float(threshold)) for row in rows]
    risky_rows = [row for row in rows if bool(row["risky_case"])]
    safe_rows = [row for row in rows if bool(row["safe_case"])]
    risky_flagged = sum(
        1 for row, is_flagged in zip(rows, flagged) if bool(row["risky_case"]) and is_flagged
    )
    safe_unflagged = sum(
        1 for row, is_flagged in zip(rows, flagged) if bool(row["safe_case"]) and not is_flagged
    )
    abstain_rate = 0.0 if mode == "V1" else float(sum(1 for value in flagged if value) / len(rows)) if rows else 0.0
    residual_false_safe = sum(
        1
        for row, is_flagged in zip(rows, flagged)
        if bool(row["false_safe_flag_A"]) and not is_flagged
    )
    accepted_rows = sum(1 for is_flagged in flagged if not is_flagged)
    return {
        "cases": len(rows),
        "threshold": float(threshold),
        "mode": mode,
        "flag_rate": float(sum(1 for value in flagged if value) / len(rows)) if rows else 0.0,
        "abstain_rate": abstain_rate,
        "false_safe_rate_A": rate(rows, lambda row: bool(row["false_safe_flag_A"])),
        "false_safe_rate_after_veto": safe_rate(residual_false_safe, len(rows)),
        "accepted_false_safe_rate": safe_rate(residual_false_safe, accepted_rows),
        "risky_case_rate": rate(rows, lambda row: bool(row["risky_case"])),
        "risky_case_capture_rate": safe_rate(risky_flagged, len(risky_rows)),
        "safe_case_retention_rate": safe_rate(safe_unflagged, len(safe_rows)),
        "negative_gain_rate_B": rate(rows, lambda row: bool(row["negative_gain_B"])),
    }


def select_operating_point(rows: Sequence[dict[str, Any]], *, mode: str) -> dict[str, Any] | None:
    candidates = [dict(row) for row in rows if str(row.get("mode")) == str(mode)]
    if not candidates:
        return None

    def score(entry: dict[str, Any]) -> float:
        base = (
            float(entry["risky_case_capture_rate"])
            + float(entry["safe_case_retention_rate"])
            - float(entry["false_safe_rate_after_veto"])
        )
        if str(mode) == "V2":
            base -= float(entry["abstain_rate"])
            base -= float(entry["accepted_false_safe_rate"])
        return base

    best = max(
        candidates,
        key=lambda entry: (
            score(entry),
            -float(entry["threshold"]),
            -float(entry["threshold_quantile"]),
        ),
    )
    best["selection_score"] = score(best)
    return best

"""Post-hoc linkage audit for Phase-2 recovery decisions.

This script does not run new simulations.  It re-aggregates row-level Phase-2
results to connect syndrome ambiguity, the C3R uncertainty gate, switch
intervention classes, and failure-boundary/logical-success outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype


ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "biqmn" / "results" / "tables"

INPUTS = {
    "clean": TABLE_DIR / "hybrid_c123_regime_map_c3r_phase2_seed10_raw.csv",
    "noisy": TABLE_DIR / "noisy_syndrome_c3r_phase2_seed10_raw.csv",
    "partial": TABLE_DIR / "partial_syndrome_c3r_phase2_seed10_raw.csv",
    "partial_noisy": TABLE_DIR / "partial_noisy_syndrome_c3r_phase2_seed10_raw.csv",
    "ambiguity_measurement": TABLE_DIR
    / "ambiguity_measurement_c3r_phase2_seed10_raw.csv",
}

OUTPUT_PREFIX = TABLE_DIR / "phase2_failure_linkage"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "b"})


def safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    """Compute ROC AUC without sklearn, using average ranks for ties."""
    y = np.asarray(y_true).astype(bool)
    values = np.asarray(score, dtype=float)
    finite = np.isfinite(values)
    y = y[finite]
    values = values[finite]
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum = ranks[y].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def summarize(group: pd.DataFrame) -> pd.Series:
    c2_switch = as_bool(group["c3r_gate_c2_switch"])
    blocked = as_bool(group["c3r_blocks_c2_switch_flag"])
    harmful = as_bool(group["c3r_harmful_c2_switch_flag"])
    beneficial = as_bool(group["c3r_beneficial_c2_switch_flag"])
    prevented = as_bool(group["c3r_prevented_harmful_switch_flag"])
    missed = as_bool(group["c3r_missed_beneficial_switch_flag"])
    logical_c2 = as_bool(group["logical_success_C2"])
    logical_c3r = as_bool(group["logical_success_C3R"])
    true_fail_c2 = as_bool(group["true_failure_boundary_flag_C2"])
    true_fail_c3r = as_bool(group["true_failure_boundary_flag_C3R"])
    true_fail_b = as_bool(group["true_failure_boundary_flag_B"])
    logical_a = as_bool(group["logical_success_A"])
    logical_b = as_bool(group["logical_success_B"])
    fidelity_loss = group["fid_gain_B"] < group["fid_gain_A"]
    logical_loss = logical_a & ~logical_b

    return pd.Series(
        {
            "cases": len(group),
            "uncertainty_mean": group["c3r_raw_syndrome_uncertainty"].mean(),
            "corruption_mean": group["syndrome_corruption_rate"].mean(),
            "information_loss_mean": group["syndrome_information_loss"].mean(),
            "c2_B_rate": c2_switch.mean(),
            "c3r_B_rate": group["candidate_C3R"].astype(str).eq("B").mean(),
            "block_rate_given_c2B": safe_rate(blocked.sum(), c2_switch.sum()),
            "harmful_switch_rate_given_c2B": safe_rate(harmful.sum(), c2_switch.sum()),
            "beneficial_switch_rate_given_c2B": safe_rate(
                beneficial.sum(), c2_switch.sum()
            ),
            "neutral_switch_rate_given_c2B": safe_rate(
                (c2_switch & ~(harmful | beneficial)).sum(), c2_switch.sum()
            ),
            "c2_true_failure_rate_given_c2B": safe_rate(
                (c2_switch & true_fail_c2).sum(), c2_switch.sum()
            ),
            "B_true_failure_rate_given_c2B": safe_rate(
                (c2_switch & true_fail_b).sum(), c2_switch.sum()
            ),
            "B_fidelity_loss_rate_given_c2B": safe_rate(
                (c2_switch & fidelity_loss).sum(), c2_switch.sum()
            ),
            "B_logical_loss_rate_given_c2B": safe_rate(
                (c2_switch & logical_loss).sum(), c2_switch.sum()
            ),
            "harmful_recall": safe_rate(prevented.sum(), harmful.sum()),
            "beneficial_retention": 1.0 - safe_rate(missed.sum(), beneficial.sum()),
            "harmful_block_precision": safe_rate(prevented.sum(), blocked.sum()),
            "prevented_loss_sum": group["c3r_prevented_loss"].sum(),
            "missed_gain_sum": group["c3r_missed_gain"].sum(),
            "net_intervention_gain": group["c3r_net_intervention_gain"].sum(),
            "prevented_loss_per_1000": safe_rate(
                1000.0 * group["c3r_prevented_loss"].sum(), len(group)
            ),
            "missed_gain_per_1000": safe_rate(
                1000.0 * group["c3r_missed_gain"].sum(), len(group)
            ),
            "net_gain_per_1000": safe_rate(
                1000.0 * group["c3r_net_intervention_gain"].sum(), len(group)
            ),
            "logical_success_C2": logical_c2.mean(),
            "logical_success_C3R": logical_c3r.mean(),
            "delta_logical_success": logical_c3r.mean() - logical_c2.mean(),
            "true_failure_C2": true_fail_c2.mean(),
            "true_failure_C3R": true_fail_c3r.mean(),
            "delta_true_failure": true_fail_c3r.mean() - true_fail_c2.mean(),
            "oracle_regret_C2": group["oracle_regret_C2"].mean(),
            "oracle_regret_C3R": group["oracle_regret_C3R"].mean(),
            "score_margin_mean": group["c3r_score_margin"].mean(),
            "structural_margin_mean": group["c3r_structural_margin"].mean(),
            "traj_delta_B_minus_A_mean": (
                group["traj_distance_B"] - group["traj_distance_A"]
            ).mean(),
            "objective_delta_B_minus_A_mean": (
                group["objective_B"] - group["objective_A"]
            ).mean(),
            "fid_gain_B_minus_A_mean": (group["fid_gain_B"] - group["fid_gain_A"]).mean(),
        }
    )


def group_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = list(columns)
    if not columns:
        return summarize(df).to_frame().T
    return df.groupby(columns, dropna=False, observed=True).apply(summarize).reset_index()


def assign_switch_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    blocked = as_bool(out["c3r_blocks_c2_switch_flag"])
    harmful = as_bool(out["c3r_harmful_c2_switch_flag"])
    beneficial = as_bool(out["c3r_beneficial_c2_switch_flag"])
    c2_switch = as_bool(out["c3r_gate_c2_switch"])
    prevented = as_bool(out["c3r_prevented_harmful_switch_flag"])
    missed = as_bool(out["c3r_missed_beneficial_switch_flag"])

    out["switch_class"] = np.select(
        [
            prevented,
            missed,
            harmful & ~blocked,
            beneficial & ~blocked,
            c2_switch,
        ],
        [
            "prevented_harmful",
            "missed_beneficial",
            "unblocked_harmful",
            "retained_beneficial",
            "other_c2_switch",
        ],
        default="no_c2_switch",
    )
    return out


def predictive_auc_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    feature_getters = {
        "raw_uncertainty": lambda g: g["c3r_raw_syndrome_uncertainty"],
        "structural_margin": lambda g: g["c3r_structural_margin"],
        "score_margin": lambda g: g["c3r_score_margin"],
        "trajectory_delta_B_minus_A": lambda g: g["traj_distance_B"]
        - g["traj_distance_A"],
        "objective_delta_B_minus_A": lambda g: g["objective_B"] - g["objective_A"],
    }
    target_getters = {
        "c3r_block": lambda g: as_bool(g["c3r_blocks_c2_switch_flag"]),
        "harmful_switch": lambda g: as_bool(g["c3r_harmful_c2_switch_flag"]),
        "beneficial_switch": lambda g: as_bool(g["c3r_beneficial_c2_switch_flag"]),
        "c2_true_failure": lambda g: as_bool(g["true_failure_boundary_flag_C2"]),
        "prevented_harmful": lambda g: as_bool(g["c3r_prevented_harmful_switch_flag"]),
        "missed_beneficial": lambda g: as_bool(g["c3r_missed_beneficial_switch_flag"]),
    }

    for regime, frame in frames.items():
        c2_switch_rows = frame[as_bool(frame["c3r_gate_c2_switch"])].copy()
        if c2_switch_rows.empty:
            continue
        for target_name, target_getter in target_getters.items():
            target = target_getter(c2_switch_rows)
            positives = int(target.sum())
            negatives = int((~target).sum())
            for feature_name, feature_getter in feature_getters.items():
                auc = roc_auc(target, feature_getter(c2_switch_rows))
                rows.append(
                    {
                        "regime": regime,
                        "target": target_name,
                        "feature": feature_name,
                        "positive_count": positives,
                        "negative_count": negatives,
                        "auc_high_value": auc,
                        "directional_auc": max(auc, 1.0 - auc)
                        if np.isfinite(auc)
                        else float("nan"),
                        "positive_direction": "high"
                        if np.isfinite(auc) and auc >= 0.5
                        else "low",
                    }
                )
    return pd.DataFrame(rows)


def replay_policy(frame: pd.DataFrame, policy: str, allow_b: pd.Series) -> dict[str, float]:
    c2_switch = as_bool(frame["c3r_gate_c2_switch"])
    allow_b = allow_b.astype(bool)
    block = c2_switch & ~allow_b
    harmful = as_bool(frame["c3r_harmful_c2_switch_flag"])
    beneficial = as_bool(frame["c3r_beneficial_c2_switch_flag"])

    logical = np.where(
        allow_b, as_bool(frame["logical_success_B"]), as_bool(frame["logical_success_A"])
    )
    true_failure = np.where(
        allow_b,
        as_bool(frame["true_failure_boundary_flag_B"]),
        as_bool(frame["true_failure_boundary_flag_A"]),
    )
    intervention_delta = np.where(block, frame["fid_gain_A"] - frame["fid_gain_B"], 0.0)
    prevented_loss = np.where(block, np.maximum(frame["fid_gain_A"] - frame["fid_gain_B"], 0), 0.0)
    missed_gain = np.where(block, np.maximum(frame["fid_gain_B"] - frame["fid_gain_A"], 0), 0.0)

    return {
        "policy": policy,
        "cases": len(frame),
        "chosen_B_rate": allow_b.mean(),
        "block_rate_given_c2B": safe_rate(block.sum(), c2_switch.sum()),
        "harmful_recall": safe_rate((block & harmful).sum(), harmful.sum()),
        "beneficial_retention": 1.0
        - safe_rate((block & beneficial).sum(), beneficial.sum()),
        "logical_success": float(np.mean(logical)),
        "true_failure": float(np.mean(true_failure)),
        "prevented_loss_sum": float(np.sum(prevented_loss)),
        "missed_gain_sum": float(np.sum(missed_gain)),
        "net_intervention_gain": float(np.sum(intervention_delta)),
    }


def candidate_replay_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in frames.items():
        policies = {
            "A only": pd.Series(False, index=frame.index),
            "B only": pd.Series(True, index=frame.index),
            "C2 replay": as_bool(frame["c3r_gate_c2_switch"]),
            "C3R full": frame["candidate_C3R"].astype(str).eq("B"),
        }
        for policy, choose_b in policies.items():
            result = replay_policy(frame, policy, choose_b)
            result["regime"] = regime
            rows.append(result)
    columns = ["regime"] + [c for c in rows[0].keys() if c != "regime"]
    return pd.DataFrame(rows)[columns]


def gate_replay_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in frames.items():
        c2_switch = as_bool(frame["c3r_gate_c2_switch"])
        score_gate = as_bool(frame["c3r_gate_score_margin"])
        leave_a_gate = as_bool(frame["c3r_gate_leave_A"])
        b_safe_gate = as_bool(frame["c3r_gate_B_safe"])
        uncertainty_gate = as_bool(frame["c3r_gate_uncertainty"])
        full_gate = as_bool(frame["c3r_allow_B"])

        policies = {
            "C2 replay": c2_switch,
            "C3R full": c2_switch & full_gate,
            "no uncertainty gate": c2_switch & score_gate & leave_a_gate & b_safe_gate,
            "uncertainty only": c2_switch & uncertainty_gate,
            "score gate only": c2_switch & score_gate,
        }
        for policy, allow_b in policies.items():
            result = replay_policy(frame, policy, allow_b)
            result["regime"] = regime
            result["score_gate_pass_given_c2B"] = safe_rate(
                (score_gate & c2_switch).sum(), c2_switch.sum()
            )
            result["leave_A_gate_pass_given_c2B"] = safe_rate(
                (leave_a_gate & c2_switch).sum(), c2_switch.sum()
            )
            result["B_safe_gate_pass_given_c2B"] = safe_rate(
                (b_safe_gate & c2_switch).sum(), c2_switch.sum()
            )
            result["uncertainty_gate_pass_given_c2B"] = safe_rate(
                (uncertainty_gate & c2_switch).sum(), c2_switch.sum()
            )
            rows.append(result)
    columns = ["regime"] + [c for c in rows[0].keys() if c != "regime"]
    return pd.DataFrame(rows)[columns]


def c2_switch_taxonomy_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in frames.items():
        c2_switch = as_bool(frame["c3r_gate_c2_switch"])
        subset = frame[c2_switch].copy()
        if subset.empty:
            rows.append(
                {
                    "regime": regime,
                    "c2_B_count": 0,
                    "harmful_count": 0,
                    "beneficial_count": 0,
                    "neutral_count": 0,
                    "harmful_rate": 0.0,
                    "beneficial_rate": 0.0,
                    "neutral_rate": 0.0,
                    "true_failure_rate": 0.0,
                    "B_fidelity_loss_rate": 0.0,
                    "B_logical_loss_rate": 0.0,
                    "fid_gain_B_minus_A_mean": 0.0,
                }
            )
            continue
        harmful = as_bool(subset["c3r_harmful_c2_switch_flag"])
        beneficial = as_bool(subset["c3r_beneficial_c2_switch_flag"])
        neutral = ~(harmful | beneficial)
        fidelity_loss = subset["fid_gain_B"] < subset["fid_gain_A"]
        logical_loss = as_bool(subset["logical_success_A"]) & ~as_bool(
            subset["logical_success_B"]
        )
        true_failure = as_bool(subset["true_failure_boundary_flag_C2"])
        rows.append(
            {
                "regime": regime,
                "c2_B_count": len(subset),
                "harmful_count": int(harmful.sum()),
                "beneficial_count": int(beneficial.sum()),
                "neutral_count": int(neutral.sum()),
                "harmful_rate": harmful.mean(),
                "beneficial_rate": beneficial.mean(),
                "neutral_rate": neutral.mean(),
                "true_failure_rate": true_failure.mean(),
                "B_fidelity_loss_rate": fidelity_loss.mean(),
                "B_logical_loss_rate": logical_loss.mean(),
                "fid_gain_B_minus_A_mean": (
                    subset["fid_gain_B"] - subset["fid_gain_A"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def per_code_mechanism_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for regime, frame in frames.items():
        summary = group_summary(frame, ["code_family"])
        summary.insert(0, "regime", regime)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def uncertainty_calibration_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bins = [-1e-12, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, np.inf]
    labels = ["[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "[1,1.25)", "[1.25,1.5)", "[1.5,+)"]
    parts = []
    for regime, frame in frames.items():
        copy = frame.copy()
        copy["uncertainty_bin"] = pd.cut(
            copy["c3r_raw_syndrome_uncertainty"],
            bins=bins,
            labels=labels,
            right=False,
        )
        summary = group_summary(copy, ["uncertainty_bin"])
        summary = summary[summary["cases"] > 0].copy()
        summary.insert(0, "regime", regime)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def benefit_cost_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in frames.items():
        summary = summarize(frame).to_dict()
        rows.append(
            {
                "regime": regime,
                "cases": int(summary["cases"]),
                "c2_B_rate": summary["c2_B_rate"],
                "c3r_B_rate": summary["c3r_B_rate"],
                "block_rate_given_c2B": summary["block_rate_given_c2B"],
                "prevented_loss_sum": summary["prevented_loss_sum"],
                "missed_gain_sum": summary["missed_gain_sum"],
                "net_intervention_gain": summary["net_intervention_gain"],
                "prevented_loss_per_1000": summary["prevented_loss_per_1000"],
                "missed_gain_per_1000": summary["missed_gain_per_1000"],
                "net_gain_per_1000": summary["net_gain_per_1000"],
                "delta_logical_success": summary["delta_logical_success"],
                "delta_true_failure": summary["delta_true_failure"],
            }
        )
    return pd.DataFrame(rows)


def seed_mechanism_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for regime, frame in frames.items():
        for seed, group in frame.groupby("seed", dropna=False):
            c2_switch = as_bool(group["c3r_gate_c2_switch"])
            score_gate = as_bool(group["c3r_gate_score_margin"])
            leave_a_gate = as_bool(group["c3r_gate_leave_A"])
            b_safe_gate = as_bool(group["c3r_gate_B_safe"])
            uncertainty_gate = as_bool(group["c3r_gate_uncertainty"])
            full_gate = as_bool(group["c3r_allow_B"])

            no_uncertainty_choice = c2_switch & score_gate & leave_a_gate & b_safe_gate
            uncertainty_only_choice = c2_switch & uncertainty_gate
            c2_choice = c2_switch
            c3r_choice = c2_switch & full_gate

            summary = summarize(group).to_dict()
            seed_rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "cases": int(summary["cases"]),
                    "c2_B_rate": summary["c2_B_rate"],
                    "c3r_B_rate": summary["c3r_B_rate"],
                    "block_rate_given_c2B": summary["block_rate_given_c2B"],
                    "harmful_recall": summary["harmful_recall"],
                    "beneficial_retention": summary["beneficial_retention"],
                    "delta_logical_success": summary["delta_logical_success"],
                    "delta_true_failure": summary["delta_true_failure"],
                    "net_intervention_gain": summary["net_intervention_gain"],
                    "no_uncertainty_equals_C2": bool(
                        np.array_equal(no_uncertainty_choice.to_numpy(), c2_choice.to_numpy())
                    ),
                    "uncertainty_only_equals_C3R": bool(
                        np.array_equal(
                            uncertainty_only_choice.to_numpy(), c3r_choice.to_numpy()
                        )
                    ),
                }
            )
    seed_table = pd.DataFrame(seed_rows)
    consistency_rows = []
    for regime, group in seed_table.groupby("regime", dropna=False):
        consistency_rows.append(
            {
                "regime": regime,
                "seeds": len(group),
                "positive_block_seeds": int((group["block_rate_given_c2B"] > 0).sum()),
                "positive_harmful_recall_seeds": int((group["harmful_recall"] > 0).sum()),
                "positive_delta_logical_success_seeds": int(
                    (group["delta_logical_success"] > 0).sum()
                ),
                "negative_delta_true_failure_seeds": int(
                    (group["delta_true_failure"] < 0).sum()
                ),
                "positive_net_gain_seeds": int(
                    (group["net_intervention_gain"] > 0).sum()
                ),
                "no_uncertainty_equals_C2_seeds": int(
                    group["no_uncertainty_equals_C2"].sum()
                ),
                "uncertainty_only_equals_C3R_seeds": int(
                    group["uncertainty_only_equals_C3R"].sum()
                ),
                "mean_delta_logical_success": group["delta_logical_success"].mean(),
                "mean_delta_true_failure": group["delta_true_failure"].mean(),
                "mean_net_intervention_gain": group["net_intervention_gain"].mean(),
            }
        )
    return seed_table, pd.DataFrame(consistency_rows)


def table_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if col == "cases":
            formatted[col] = formatted[col].astype(int).astype(str)
        elif is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:.{float_digits}f}")
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    frames = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    ambiguity = frames["ambiguity_measurement"]

    by_ambiguity = group_summary(ambiguity, ["syndrome_ambiguity_level"])
    by_measurement = group_summary(ambiguity, ["measurement_error_prob"])

    switch_classes = group_summary(
        assign_switch_classes(ambiguity), ["switch_class"]
    ).sort_values("cases", ascending=False)

    binned = []
    bins = [-1e-12, 0.25, 0.5, 0.75, 1.0, np.inf]
    labels = ["[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "[1,+)"]
    for regime, frame in frames.items():
        copy = frame.copy()
        copy["uncertainty_bin"] = pd.cut(
            copy["c3r_raw_syndrome_uncertainty"],
            bins=bins,
            labels=labels,
            right=False,
        )
        summary = group_summary(copy, ["uncertainty_bin"])
        summary.insert(0, "regime", regime)
        binned.append(summary)
    uncertainty_bins = pd.concat(binned, ignore_index=True)

    cell = group_summary(
        ambiguity, ["syndrome_ambiguity_level", "measurement_error_prob"]
    )
    drivers = [
        "syndrome_ambiguity_level",
        "measurement_error_prob",
        "uncertainty_mean",
        "corruption_mean",
    ]
    outcomes = [
        "block_rate_given_c2B",
        "harmful_recall",
        "beneficial_retention",
        "net_intervention_gain",
        "delta_logical_success",
        "delta_true_failure",
    ]
    correlations = (
        cell[drivers + outcomes]
        .corr(method="spearman")
        .loc[drivers, outcomes]
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    seed_mechanism, seed_consistency = seed_mechanism_tables(frames)

    outputs = {
        "by_ambiguity_level": by_ambiguity,
        "by_measurement_error": by_measurement,
        "switch_classes": switch_classes,
        "uncertainty_bins": uncertainty_bins,
        "cell_spearman_correlations": correlations,
        "predictive_auc": predictive_auc_table(frames),
        "gate_replay": gate_replay_table(frames),
        "candidate_replay": candidate_replay_table(frames),
        "c2_switch_taxonomy": c2_switch_taxonomy_table(frames),
        "per_code_mechanism": per_code_mechanism_table(frames),
        "uncertainty_calibration": uncertainty_calibration_table(frames),
        "benefit_cost_by_regime": benefit_cost_table(frames),
        "seed_mechanism": seed_mechanism,
        "seed_consistency": seed_consistency,
    }
    for suffix, df in outputs.items():
        df.to_csv(f"{OUTPUT_PREFIX}_{suffix}.csv", index=False)

    report_sections = [
        "# Phase-2 Failure-Linkage Audit",
        "",
        "This is a post-hoc row-level aggregation. It links ambiguity and",
        "measurement/reset stress to the C3R uncertainty gate, switch classes,",
        "and failure-boundary outcomes. It does not introduce new simulations.",
        "",
        "## By Ambiguity Level",
        "",
        table_to_markdown(
            by_ambiguity[
                [
                    "syndrome_ambiguity_level",
                    "cases",
                    "uncertainty_mean",
                    "corruption_mean",
                    "block_rate_given_c2B",
                    "harmful_recall",
                    "beneficial_retention",
                    "delta_logical_success",
                    "delta_true_failure",
                    "prevented_loss_sum",
                    "missed_gain_sum",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## Switch Classes in Ambiguity plus Measurement/Reset",
        "",
        table_to_markdown(
            switch_classes[
                [
                    "switch_class",
                    "cases",
                    "uncertainty_mean",
                    "score_margin_mean",
                    "structural_margin_mean",
                    "delta_logical_success",
                    "delta_true_failure",
                    "prevented_loss_sum",
                    "missed_gain_sum",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## Uncertainty Bins Across Active Regimes",
        "",
        table_to_markdown(
            uncertainty_bins[uncertainty_bins["cases"] > 0][
                [
                    "regime",
                    "uncertainty_bin",
                    "cases",
                    "block_rate_given_c2B",
                    "harmful_recall",
                    "beneficial_retention",
                    "delta_logical_success",
                    "delta_true_failure",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## Spearman Correlations Across Ambiguity/Measurement Cells",
        "",
        table_to_markdown(correlations),
        "",
        "## Predictive AUC Audit",
        "",
        "AUC values use the feature's high-value direction directly. Values",
        "near 0.5 are close to chance in that direction.",
        "",
        table_to_markdown(
            outputs["predictive_auc"][
                outputs["predictive_auc"]["regime"].isin(
                    ["partial_noisy", "ambiguity_measurement"]
                )
                & outputs["predictive_auc"]["target"].isin(
                    [
                        "c3r_block",
                        "c2_true_failure",
                        "prevented_harmful",
                        "missed_beneficial",
                    ]
                )
                & outputs["predictive_auc"]["feature"].isin(
                    ["raw_uncertainty", "structural_margin", "score_margin"]
                )
            ][
                [
                    "regime",
                    "target",
                    "feature",
                    "positive_count",
                    "negative_count",
                    "auc_high_value",
                    "directional_auc",
                    "positive_direction",
                ]
            ]
        ),
        "",
        "## Gate Replay Audit",
        "",
        table_to_markdown(
            outputs["gate_replay"][
                outputs["gate_replay"]["regime"].isin(
                    ["partial_noisy", "ambiguity_measurement"]
                )
                & outputs["gate_replay"]["policy"].isin(
                    [
                        "C2 replay",
                        "C3R full",
                        "no uncertainty gate",
                        "uncertainty only",
                    ]
                )
            ][
                [
                    "regime",
                    "policy",
                    "chosen_B_rate",
                    "block_rate_given_c2B",
                    "harmful_recall",
                    "beneficial_retention",
                    "logical_success",
                    "true_failure",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## Candidate Replay Audit",
        "",
        table_to_markdown(
            outputs["candidate_replay"][
                outputs["candidate_replay"]["regime"].isin(
                    ["partial", "partial_noisy", "ambiguity_measurement"]
                )
                & outputs["candidate_replay"]["policy"].isin(
                    ["A only", "B only", "C2 replay", "C3R full"]
                )
            ][
                [
                    "regime",
                    "policy",
                    "chosen_B_rate",
                    "logical_success",
                    "true_failure",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## C2 Failure-Mode Taxonomy",
        "",
        table_to_markdown(
            outputs["c2_switch_taxonomy"][
                [
                    "regime",
                    "c2_B_count",
                    "harmful_rate",
                    "beneficial_rate",
                    "neutral_rate",
                    "true_failure_rate",
                    "B_fidelity_loss_rate",
                    "B_logical_loss_rate",
                    "fid_gain_B_minus_A_mean",
                ]
            ]
        ),
        "",
        "## Per-Code Mechanism Split",
        "",
        table_to_markdown(
            outputs["per_code_mechanism"][
                outputs["per_code_mechanism"]["regime"].isin(
                    ["partial", "partial_noisy", "ambiguity_measurement"]
                )
            ][
                [
                    "regime",
                    "code_family",
                    "cases",
                    "c2_B_rate",
                    "c3r_B_rate",
                    "block_rate_given_c2B",
                    "harmful_recall",
                    "beneficial_retention",
                    "delta_logical_success",
                    "delta_true_failure",
                    "net_intervention_gain",
                ]
            ]
        ),
        "",
        "## Benefit-Cost Decomposition by Regime",
        "",
        table_to_markdown(
            outputs["benefit_cost_by_regime"][
                [
                    "regime",
                    "cases",
                    "block_rate_given_c2B",
                    "prevented_loss_sum",
                    "missed_gain_sum",
                    "net_intervention_gain",
                    "delta_logical_success",
                    "delta_true_failure",
                ]
            ]
        ),
        "",
        "## Seed-Level Mechanism Consistency",
        "",
        table_to_markdown(
            outputs["seed_consistency"][
                [
                    "regime",
                    "seeds",
                    "positive_block_seeds",
                    "positive_harmful_recall_seeds",
                    "positive_delta_logical_success_seeds",
                    "negative_delta_true_failure_seeds",
                    "positive_net_gain_seeds",
                    "no_uncertainty_equals_C2_seeds",
                    "uncertainty_only_equals_C3R_seeds",
                    "mean_delta_logical_success",
                    "mean_delta_true_failure",
                    "mean_net_intervention_gain",
                ]
            ]
        ),
        "",
    ]
    Path(f"{OUTPUT_PREFIX}.md").write_text("\n".join(report_sections), encoding="utf-8")

    print(f"wrote {OUTPUT_PREFIX}_*.csv")
    print(f"wrote {OUTPUT_PREFIX}.md")


if __name__ == "__main__":
    main()

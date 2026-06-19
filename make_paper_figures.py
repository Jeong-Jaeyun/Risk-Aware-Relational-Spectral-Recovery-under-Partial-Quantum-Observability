from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "biqmn" / "results" / "tables"
RAW = ROOT / "biqmn" / "results" / "raw"
FIG_DIR = ROOT / "template" / "figure"
SN_FIG_DIR = ROOT / "template" / "springer_Nature" / "figure"
FIG_DIRS = (FIG_DIR, SN_FIG_DIR)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


def _savefig(fig: plt.Figure, filename: str, *, springer_main: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename)
    if springer_main:
        SN_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(SN_FIG_DIR / filename)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def regime_overview() -> None:
    stems = [
        ("Clean", "hybrid_c123_regime_map_c3r_phase2_seed10"),
        ("Partial", "partial_syndrome_c3r_phase2_seed10"),
        ("Noisy", "noisy_syndrome_c3r_phase2_seed10"),
        ("Partial+noisy", "partial_noisy_syndrome_c3r_phase2_seed10"),
        ("Ambiguity", "ambiguity_measurement_c3r_phase2_seed10"),
    ]
    rows = []
    for label, stem in stems:
        overall = _read_json(RAW / f"{stem}.json")["overall"]
        rows.append({
            "label": label,
            "cases": int(overall["cases"]),
            "ls_c2": float(overall["logical_success_rate_C2"]),
            "ls_c3r": float(overall["logical_success_rate_C3R"]),
            "chosen_c2": float(overall["chosen_B_rate_C2"]),
            "chosen_c3r": float(overall["chosen_B_rate_C3R"]),
            "block_given_c2b": float(overall.get("c3r_block_rate_given_c2_B", 0.0)),
            "harm_recall": float(overall.get("c3r_harmful_switch_recall", 0.0)),
        })

    x = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.9, 3.1), constrained_layout=True)
    ax.bar(x - width / 2, [r["ls_c2"] for r in rows], width, label="C2", color="#4C78A8")
    ax.bar(x + width / 2, [r["ls_c3r"] for r in rows], width, label="C3R", color="#E45756")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Logical success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Recovery success across stress regimes")
    _savefig(fig, "fig_regime_logical_success.pdf", springer_main=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    ax.bar(x - width / 2, [r["chosen_c2"] for r in rows], width, label="C2 selects B", color="#72B7B2")
    ax.bar(x + width / 2, [r["chosen_c3r"] for r in rows], width, label="C3R selects B", color="#F58518")
    ax.plot(x, [r["block_given_c2b"] for r in rows], "o-", color="#B279A2", label="C3R block | C2 B")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Switching and robust-gate activity")
    _savefig(fig, "fig_regime_switching.pdf", springer_main=True)
    plt.close(fig)


def partial_by_code() -> None:
    rows = _read_csv(TABLES / "partial_syndrome_c3r_phase2_seed10_by_code_and_obs_ratio.csv")
    codes = ["perfect5", "steane7", "shor9"]
    labels = {"perfect5": "Perfect", "steane7": "Steane", "shor9": "Shor"}
    colors = {"perfect5": "#4C78A8", "steane7": "#F58518", "shor9": "#E45756"}
    ratios = sorted({_f(row, "syndrome_observation_ratio") for row in rows})

    fig, ax = plt.subplots(figsize=(6.5, 3.35), constrained_layout=True)
    for code in codes:
        subset = [row for row in rows if row["code_family"] == code]
        by_ratio = {round(_f(row, "syndrome_observation_ratio"), 8): row for row in subset}
        y_c2 = [_f(by_ratio[round(r, 8)], "logical_success_rate_C2") for r in ratios]
        y_c3r = [_f(by_ratio[round(r, 8)], "logical_success_rate_C3R") for r in ratios]
        ax.plot(ratios, y_c2, "o-", color=colors[code], linewidth=1.6)
        ax.plot(
            ratios,
            y_c3r,
            "s--",
            color=colors[code],
            linewidth=1.6,
        )
    ax.set_xlabel("Observed syndrome fraction")
    ax.set_ylabel("Logical success rate")
    ax.set_ylim(-0.03, 1.05)
    ax.invert_xaxis()
    ax.grid(axis="y", alpha=0.25)
    code_handles = [
        Line2D([0], [0], color=colors[code], marker="o", linestyle="-", label=labels[code])
        for code in codes
    ]
    ax.legend(handles=code_handles, frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="center left")
    ax.set_title("Partial-syndrome logical success by code")
    _savefig(fig, "fig_partial_by_code.pdf", springer_main=True)
    plt.close(fig)


def structure_sensitivity() -> None:
    rows = _read_csv(TABLES / "partial_syndrome_c3r_phase2_seed10_by_code_and_obs_ratio.csv")
    codes = ["perfect5", "steane7", "shor9"]
    labels = {
        "perfect5": "Perfect\n[[5,1,3]]",
        "steane7": "Steane\n[[7,1,3]]",
        "shor9": "Shor\n[[9,1,3]]",
    }
    # Stabilizer check-weight profiles used in the implementation.
    check_weights = {
        "perfect5": [4, 4, 4, 4],
        "steane7": [4, 4, 4, 4, 4, 4],
        "shor9": [2, 2, 2, 2, 2, 2, 6, 6],
    }
    layer_imbalance = {"perfect5": 0.0, "steane7": 0.0, "shor9": 0.5}
    weight_bins = [2, 4, 6]
    weight_colors = {2: "#72B7B2", 4: "#4C78A8", 6: "#F58518"}
    x = np.arange(len(codes))

    fig, ax = plt.subplots(figsize=(5.2, 3.1), constrained_layout=True)
    bottom = np.zeros(len(codes))
    for weight in weight_bins:
        values = [
            sum(1 for w in check_weights[code] if w == weight) / len(check_weights[code])
            for code in codes
        ]
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=f"weight {weight}",
            color=weight_colors[weight],
            width=0.62,
        )
        bottom += np.array(values)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[code] for code in codes])
    ax.set_ylabel("Fraction of stabilizer checks")
    ax.set_title("Implemented stabilizer check-weight profile")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for idx, code in enumerate(codes):
        variance = np.var(check_weights[code])
        ax.text(
            idx,
            0.06,
            f"$I_w$={variance:.1f}\n$I_\\ell$={layer_imbalance[code]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="white" if code != "shor9" else "black",
        )
    _savefig(fig, "fig_structure_profile.pdf")
    plt.close(fig)

    by_code_ratio = {
        (row["code_family"], round(_f(row, "syndrome_observation_ratio"), 8)): row
        for row in rows
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.1), constrained_layout=True)
    width = 0.28
    for offset, ratio, color in [(-width / 2, 0.25, "#E45756"), (width / 2, 0.50, "#54A24B")]:
        delta = []
        for code in codes:
            row = by_code_ratio[(code, ratio)]
            delta.append(_f(row, "logical_success_rate_C3R") - _f(row, "logical_success_rate_C2"))
        ax.bar(x + offset, delta, width, label=f"observed fraction {ratio:.2f}", color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[code] for code in codes])
    ax.set_ylabel(r"$\Delta$ logical success (C3R - C2)")
    ax.set_title("Partial-syndrome rescue by code")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, "fig_structure_sensitivity.pdf")
    plt.close(fig)


def intervention_quality() -> None:
    rows = [
        ("Partial+noisy", _read_json(RAW / "partial_noisy_syndrome_c3r_phase2_seed10.json")["overall"]),
        ("Ambiguity", _read_json(RAW / "ambiguity_measurement_c3r_phase2_seed10.json")["overall"]),
    ]
    labels = [label for label, _ in rows]
    x = np.arange(len(rows))
    width = 0.18
    metrics = [
        ("block | C2 B", "c3r_block_rate_given_c2_B", "#4C78A8"),
        ("harm recall", "c3r_harmful_switch_recall", "#E45756"),
        ("harm precision", "c3r_harmful_block_precision", "#B279A2"),
        ("beneficial retained", "c3r_beneficial_switch_retention", "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.1), constrained_layout=True)
    for idx, (name, key, color) in enumerate(metrics):
        ax.bar(
            x + (idx - 1.5) * width,
            [float(row[key]) for _, row in rows],
            width,
            label=name,
            color=color,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Robust-gate intervention rates")
    _savefig(fig, "fig_intervention_rates.pdf")
    plt.close(fig)

    gains = [float(row["c3r_net_intervention_gain"]) for _, row in rows]
    colors = ["#54A24B" if gain >= 0 else "#E45756" for gain in gains]
    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    ax.bar(x, gains, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net intervention gain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Benefit--cost balance")
    gain_span = max(gains) - min(gains)
    pad = max(1.0, 0.12 * gain_span)
    ax.set_ylim(min(gains) - pad, max(gains) + pad)
    for i, gain in enumerate(gains):
        ax.text(
            i,
            gain + (0.35 * pad if gain >= 0 else -0.35 * pad),
            f"{gain:.1f}",
            ha="center",
            va="center",
            fontsize=8,
        )

    _savefig(fig, "fig_intervention_gain.pdf")
    plt.close(fig)


def oracle_regret() -> None:
    stems = [
        ("Clean", "hybrid_c123_regime_map_c3r_phase2_seed10"),
        ("Partial", "partial_syndrome_c3r_phase2_seed10"),
        ("Noisy", "noisy_syndrome_c3r_phase2_seed10"),
        ("Partial+noisy", "partial_noisy_syndrome_c3r_phase2_seed10"),
        ("Ambiguity", "ambiguity_measurement_c3r_phase2_seed10"),
    ]
    rows = [(label, _read_json(RAW / f"{stem}.json")["overall"]) for label, stem in stems]
    labels = [label for label, _ in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.6, 3.25), constrained_layout=True)
    series = [
        ("C2 mean", "oracle_regret_mean_C2", "#4C78A8", "o", "-"),
        ("C3R mean", "oracle_regret_mean_C3R", "#E45756", "s", "-"),
        ("C2 CVaR95", "oracle_regret_cvar95_C2", "#4C78A8", "^", "--"),
        ("C3R CVaR95", "oracle_regret_cvar95_C3R", "#E45756", "D", "--"),
    ]
    for name, key, color, marker, linestyle in series:
        ax.plot(
            x,
            [float(row[key]) for _, row in rows],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=name,
            linewidth=1.6,
        )
    ax.set_ylabel("Oracle regret")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Mean and tail regret against the best two-candidate choice")

    _savefig(fig, "fig_oracle_regret.pdf")
    plt.close(fig)


def ambiguity_tradeoff() -> None:
    rows = _read_csv(TABLES / "ambiguity_measurement_c3r_phase2_seed10_by_ambiguity_and_measurement_reset.csv")
    measurement = 0.10
    subset = [row for row in rows if abs(_f(row, "measurement_error_prob") - measurement) < 1e-9]
    subset = sorted(subset, key=lambda row: _f(row, "syndrome_ambiguity_level"))
    x = [_f(row, "syndrome_ambiguity_level") for row in subset]

    fig, ax = plt.subplots(figsize=(4.7, 3.0), constrained_layout=True)
    ax.plot(x, [_f(row, "chosen_B_rate_C2") for row in subset], "o-", label="C2 selects B", color="#4C78A8")
    ax.plot(x, [_f(row, "chosen_B_rate_C3R") for row in subset], "s-", label="C3R selects B", color="#E45756")
    ax.plot(x, [_f(row, "c3r_block_rate_given_c2_B") for row in subset], "^-", label="C3R block | C2 B", color="#B279A2")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Syndrome ambiguity level")
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Ambiguity-driven switching control")
    _savefig(fig, "fig_ambiguity_tradeoff.pdf", springer_main=True)
    plt.close(fig)


def phase_diagram() -> None:
    sources = [
        ("Clean", "hybrid_c123_regime_map_c3r_phase2_seed10_by_regime_cell.csv", "o", "#4C78A8"),
        ("Partial", "partial_syndrome_c3r_phase2_seed10_by_syndrome_obs_ratio.csv", "s", "#54A24B"),
        ("Noisy", "noisy_syndrome_c3r_phase2_seed10_by_syndrome_noise_prob.csv", "^", "#F58518"),
        ("Partial+noisy", "partial_noisy_syndrome_c3r_phase2_seed10_by_obs_ratio_and_noise_prob.csv", "D", "#B279A2"),
        ("Ambiguity", "ambiguity_measurement_c3r_phase2_seed10_by_ambiguity_and_measurement_reset.csv", "P", "#E45756"),
    ]
    points = []
    for regime, filename, marker, edge_color in sources:
        rows = _read_csv(TABLES / filename)
        if regime == "Clean":
            # The clean regime-cell table contains many identical uncertainty/corruption points.
            rows = [rows[0]]
        for row in rows:
            cases = max(_f(row, "cases"), 1.0)
            points.append({
                "regime": regime,
                "marker": marker,
                "edge_color": edge_color,
                "uncertainty": _f(row, "c3r_syndrome_uncertainty_mean"),
                "corruption": _f(row, "syndrome_corruption_rate"),
                "gain_per_1000": 1000.0 * _f(row, "c3r_net_intervention_gain") / cases,
            })

    vmax = max(abs(p["gain_per_1000"]) for p in points)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.9, 3.85), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")

    positive_zone = mpatches.Ellipse(
        (0.985, 0.835),
        width=0.24,
        height=0.24,
        angle=0,
        facecolor="#54A24B",
        edgecolor="#2C7A3F",
        linewidth=1.2,
        linestyle="--",
        alpha=0.10,
        zorder=0,
    )
    negative_zone = mpatches.Ellipse(
        (0.965, 0.845),
        width=0.32,
        height=0.48,
        angle=-10,
        facecolor="#E45756",
        edgecolor="#B23A3A",
        linewidth=1.2,
        linestyle=":",
        alpha=0.08,
        zorder=0,
    )
    ax.add_patch(positive_zone)
    ax.add_patch(negative_zone)

    for regime, _, marker, edge_color in sources:
        subset = [p for p in points if p["regime"] == regime]
        sc = ax.scatter(
            [p["uncertainty"] for p in subset],
            [p["corruption"] for p in subset],
            c=[p["gain_per_1000"] for p in subset],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=82 if regime in {"Partial", "Partial+noisy"} else 60,
            edgecolors=edge_color,
            linewidths=1.0,
            label=regime,
            zorder=3,
        )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Syndrome ambiguity/uncertainty score")
    ax.set_ylabel("Syndrome corruption rate")
    ax.grid(alpha=0.22)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        loc="lower right",
        borderpad=0.35,
        handlelength=1.1,
        labelspacing=0.35,
    )
    ax.set_title("Empirical recovery operating map")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Net intervention gain per 1000 cases")

    ax.annotate(
        "positive intervention",
        xy=(0.985, 0.825),
        xytext=(0.43, 0.93),
        ha="left",
        va="top",
        color="#2C7A3F",
        arrowprops={"arrowstyle": "->", "color": "#2C7A3F", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#B8D9BE", "alpha": 0.92},
        zorder=4,
    )
    ax.annotate(
        "conservative negative-gain",
        xy=(0.91, 0.65),
        xytext=(0.36, 0.58),
        ha="left",
        va="center",
        color="#B23A3A",
        arrowprops={"arrowstyle": "->", "color": "#B23A3A", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E6B0B0", "alpha": 0.92},
        zorder=4,
    )
    ax.annotate(
        "inactive controls",
        xy=(0.05, 0.06),
        xytext=(0.18, 0.18),
        ha="left",
        va="center",
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#CCCCCC", "alpha": 0.90},
        zorder=4,
    )

    _savefig(fig, "fig_phase_diagram.pdf", springer_main=True)
    plt.close(fig)


def main() -> None:
    for fig_dir in FIG_DIRS:
        fig_dir.mkdir(parents=True, exist_ok=True)
    _style()
    regime_overview()
    partial_by_code()
    structure_sensitivity()
    intervention_quality()
    oracle_regret()
    ambiguity_tradeoff()
    phase_diagram()
    print("Wrote figures to " + ", ".join(str(path) for path in FIG_DIRS))


if __name__ == "__main__":
    main()

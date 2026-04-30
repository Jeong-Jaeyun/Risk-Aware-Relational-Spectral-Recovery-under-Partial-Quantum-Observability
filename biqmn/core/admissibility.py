"""Admissibility penalties Φ_lap / Φ_smooth / Φ_ref / Φ_clock.

Implements the admissible-set definition of §10.7–10.8 of TheFirstThoery.tex.
All penalties are non-negative; admissibility is declared when each penalty
is below its calibrated threshold.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .clock_consistency import clock_penalty
from .laplacian import laplacian_validity_penalty
from .trajectory import (
    SpectralTrajectory,
    trajectory_distance_squared,
    trajectory_smoothness_penalty,
)


def phi_lap(traj: SpectralTrajectory) -> float:
    """Section 10.7 discrete sum of Laplacian consistency penalties."""
    if not traj.laplacians:
        return 0.0
    violations = [laplacian_validity_penalty(L)["total"] for L in traj.laplacians]
    return float(np.sum(violations))


def phi_smooth(traj: SpectralTrajectory) -> float:
    return trajectory_smoothness_penalty(traj)


def phi_ref(traj: SpectralTrajectory,
            ref_bank: Sequence[SpectralTrajectory]) -> float:
    """Section 10.7 nearest-reference trajectory distance squared."""
    if not ref_bank:
        return 0.0
    dists = [trajectory_distance_squared(traj, r) for r in ref_bank]
    return float(min(dists))


def phi_clock(traj: SpectralTrajectory,
              Hc: np.ndarray,
              psi0_clock: np.ndarray,
              eta: Optional[float] = None) -> float:
    del eta
    return clock_penalty(traj, Hc, psi0_clock)["phi"]


def calibrate_thresholds(reference_trajs: Sequence[SpectralTrajectory],
                         Hc: np.ndarray,
                         psi0_clock: np.ndarray,
                         kappa: float = 3.0,
                         eta: Optional[float] = None) -> dict:
    """μ + κ·σ thresholds for each Φ on a reference bank.

    Φ_ref is calibrated by holding out one trajectory and computing
    distance-to-rest for every member (leave-one-out).
    """
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("Admissibility calibration kappa must be finite and non-negative.")
    phis_lap: List[float] = []
    phis_smooth: List[float] = []
    phis_clock: List[float] = []
    phis_ref: List[float] = []
    for i, t in enumerate(reference_trajs):
        phis_lap.append(phi_lap(t))
        phis_smooth.append(phi_smooth(t))
        phis_clock.append(phi_clock(t, Hc, psi0_clock, eta))
        rest = list(reference_trajs[:i]) + list(reference_trajs[i + 1:])
        phis_ref.append(phi_ref(t, rest))

    def summ(vals):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "threshold": 0.0, "phis": []}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "threshold": float(arr.mean() + kappa * arr.std()),
            "phis": arr.tolist(),
        }

    return {
        "phi_lap": summ(phis_lap),
        "phi_smooth": summ(phis_smooth),
        "phi_ref": summ(phis_ref),
        "phi_clock": summ(phis_clock),
        "kappa": float(kappa),
        "n_ref": len(reference_trajs),
    }


def admissibility_report(traj: SpectralTrajectory,
                         ref_bank: Sequence[SpectralTrajectory],
                         Hc: np.ndarray,
                         psi0_clock: np.ndarray,
                         thresholds: dict,
                         eta: Optional[float] = None) -> dict:
    """Return penalty values and pass/fail decision per penalty."""
    required = ("phi_lap", "phi_smooth", "phi_ref", "phi_clock")
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise ValueError(
            f"Admissibility thresholds missing required keys: {', '.join(missing)}."
        )
    penalties = {
        "phi_lap": phi_lap(traj),
        "phi_smooth": phi_smooth(traj),
        "phi_ref": phi_ref(traj, ref_bank),
        "phi_clock": phi_clock(traj, Hc, psi0_clock, eta),
    }
    decisions = {k: bool(v <= thresholds.get(k, np.inf))
                 for k, v in penalties.items()}
    margins = {k: float(thresholds[k] - penalties[k]) for k in penalties}
    return {
        "penalties": penalties,
        "thresholds": {k: float(thresholds[k]) for k in penalties},
        "margins": margins,
        "decisions": decisions,
        "admissible": bool(all(decisions.values())),
    }

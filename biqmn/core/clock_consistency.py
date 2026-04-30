"""Clock consistency penalty from Section 11 of TheFirstThoery.tex.

Definitions
-----------
d_resp(r, r+1)      = spectral_response(Lambda(tau_r), Lambda(tau_{r+1}))
d_clock_geom(r,r+1) = 1 - |<tau_r|tau_{r+1}>|^2
C_clock(r)          = d_resp / (d_clock_geom + eps)
Phi_clock           = sum_r C_clock(r)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .clock import clock_geom_distance
from .trajectory import SpectralTrajectory, spectral_response


def clock_consistency_ratio(resp: float,
                            d_clock_geom: float,
                            eps: float = 1e-8) -> float:
    if eps <= 0.0:
        raise ValueError("Clock consistency regularizer eps must be positive.")
    if not np.isfinite(resp) or not np.isfinite(d_clock_geom):
        raise ValueError("Clock consistency inputs must be finite.")
    if resp < 0.0:
        raise ValueError("Clock response must be non-negative.")
    if d_clock_geom < 0.0:
        raise ValueError("Clock geometric distance must be non-negative.")
    return float(resp / (d_clock_geom + eps))


def _slice_distances(traj: SpectralTrajectory,
                     Hc: np.ndarray,
                     psi0_clock: np.ndarray):
    m = len(traj)
    resps = np.zeros(max(m - 1, 0))
    geoms = np.zeros(max(m - 1, 0))
    for r in range(m - 1):
        resps[r] = spectral_response(traj.spectra[r], traj.spectra[r + 1])
        geoms[r] = clock_geom_distance(
            Hc,
            float(traj.tau_grid[r]),
            float(traj.tau_grid[r + 1]),
            psi0_clock,
        )
    return resps, geoms


def clock_penalty(traj: SpectralTrajectory,
                  Hc: np.ndarray,
                  psi0_clock: np.ndarray,
                  eta: Optional[float] = None) -> dict:
    del eta
    m = len(traj)
    if m < 2:
        return {"phi": 0.0, "resp": [], "geom": [], "ratios": []}
    resps, geoms = _slice_distances(traj, Hc, psi0_clock)
    ratios = np.asarray(
        [clock_consistency_ratio(resp, geom) for resp, geom in zip(resps, geoms)],
        dtype=float,
    )
    return {
        "phi": float(np.sum(ratios)),
        "resp": resps.tolist(),
        "geom": geoms.tolist(),
        "ratios": ratios.tolist(),
    }


def calibrate_clock_eta(reference_trajs: Sequence[SpectralTrajectory],
                        Hc: np.ndarray,
                        psi0_clock: np.ndarray,
                        kappa: float = 3.0) -> dict:
    """Calibrate the admissibility threshold eta = mean(Phi_clock) + kappa * std."""
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("Clock calibration kappa must be a finite non-negative value.")
    phis: List[float] = []
    for traj in reference_trajs:
        phis.append(clock_penalty(traj, Hc, psi0_clock)["phi"])
    if not phis:
        return {
            "eta": 0.0,
            "phi_mean": 0.0,
            "phi_std": 0.0,
            "phi_threshold": 0.0,
            "n_ref": 0,
        }
    phis_arr = np.asarray(phis, dtype=float)
    eta = float(phis_arr.mean() + kappa * phis_arr.std())
    return {
        "eta": eta,
        "phis": phis_arr.tolist(),
        "phi_mean": float(phis_arr.mean()),
        "phi_std": float(phis_arr.std()),
        "phi_threshold": eta,
        "n_ref": len(phis),
    }

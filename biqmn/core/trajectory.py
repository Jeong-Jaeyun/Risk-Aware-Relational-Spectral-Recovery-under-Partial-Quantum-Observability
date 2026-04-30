"""Relational spectral trajectory {(τ_r, ρ_S(τ_r), L(τ_r), Λ(τ_r))}.

The SpectralTrajectory dataclass is the primary Layer-2 object.  All downstream
penalties (smoothness, clock consistency, admissibility, recovery) operate on
instances of this class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .graph_mapping import adjacency_from_reduced_density
from .laplacian import graph_laplacian, ordered_spectrum


@dataclass
class SpectralTrajectory:
    """Relational family indexed by the internal clock label τ."""
    tau_grid: np.ndarray
    densities: List[np.ndarray]            # ρ_S(τ_r)
    adjacency: List[np.ndarray]            # W(τ_r)
    laplacians: List[np.ndarray]           # L(τ_r)
    spectra: List[np.ndarray]              # Λ(τ_r), ascending
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tau_grid = np.asarray(self.tau_grid, dtype=float).reshape(-1)
        n_slices = int(self.tau_grid.size)
        for name, values in (
            ("densities", self.densities),
            ("adjacency", self.adjacency),
            ("laplacians", self.laplacians),
            ("spectra", self.spectra),
        ):
            if len(values) != n_slices:
                raise ValueError(
                    f"Trajectory {name} has length {len(values)}, expected {n_slices}."
                )
        if n_slices > 1:
            dtau = np.diff(self.tau_grid)
            if np.any(~np.isfinite(dtau)):
                raise ValueError("Trajectory tau_grid must be finite.")
            if np.any(dtau <= 0.0):
                raise ValueError("Trajectory tau_grid must be strictly increasing.")
        spec_sizes = {np.asarray(spec).reshape(-1).size for spec in self.spectra}
        if len(spec_sizes) > 1:
            raise ValueError("All trajectory spectra must have the same length.")

    def __len__(self) -> int:
        return int(np.asarray(self.tau_grid).size)

    @property
    def n_nodes(self) -> int:
        return 0 if not self.laplacians else int(self.laplacians[0].shape[0])

    def copy(self) -> "SpectralTrajectory":
        return SpectralTrajectory(
            tau_grid=np.array(self.tau_grid, copy=True),
            densities=[d.copy() for d in self.densities],
            adjacency=[w.copy() for w in self.adjacency],
            laplacians=[l.copy() for l in self.laplacians],
            spectra=[s.copy() for s in self.spectra],
            meta=dict(self.meta),
        )


def build_spectral_trajectory(relative_family: Sequence[np.ndarray],
                              n_system: int,
                              tau_grid: Sequence[float],
                              mapping_cfg: Optional[dict] = None) -> SpectralTrajectory:
    """Construct (τ, ρ, W, L, Λ) tuples from a Page-Wootters relative family."""
    cfg = mapping_cfg or {}
    mode = cfg.get("mode", "coherence_abs")
    Ws: List[np.ndarray] = []
    Ls: List[np.ndarray] = []
    spectra: List[np.ndarray] = []
    for rho in relative_family:
        W = adjacency_from_reduced_density(rho, n_system, mode=mode)
        L = graph_laplacian(W)
        Ws.append(W)
        Ls.append(L)
        spectra.append(ordered_spectrum(L))
    return SpectralTrajectory(
        tau_grid=np.asarray(tau_grid, dtype=float),
        densities=list(relative_family),
        adjacency=Ws,
        laplacians=Ls,
        spectra=spectra,
        meta={"mapping_mode": mode, "n_system": n_system},
    )


def spectral_response(lam_a: np.ndarray, lam_b: np.ndarray) -> float:
    """Paper's d_resp: sorted-eigenvalue ℓ² distance."""
    a = np.sort(np.asarray(lam_a).real)
    b = np.sort(np.asarray(lam_b).real)
    if a.size != b.size:
        raise ValueError(
            f"Spectral response requires equal dimensions, got {a.size} and {b.size}."
        )
    return float(np.linalg.norm(a - b))


def _validated_weights(m: int, weights: Optional[np.ndarray]) -> np.ndarray:
    if m == 0:
        return np.zeros(0, dtype=float)
    if weights is None:
        return np.ones(m, dtype=float) / m
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != m:
        raise ValueError(f"Expected {m} trajectory weights, got {w.size}.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("Trajectory weights must sum to a positive value.")
    return w / total


def trajectory_distance_squared(traj_a: SpectralTrajectory,
                                traj_b: SpectralTrajectory,
                                weights: Optional[np.ndarray] = None) -> float:
    """Discrete trajectory metric sum_r w_r ||Lambda_a - Lambda_b||_2^2."""
    if len(traj_a) != len(traj_b):
        raise ValueError("Trajectory distance requires equal slice counts.")
    if not np.allclose(traj_a.tau_grid, traj_b.tau_grid):
        raise ValueError("Trajectory distance requires matching tau grids.")
    m = len(traj_a)
    if m == 0:
        return 0.0
    w = _validated_weights(m, weights)
    total = 0.0
    for r in range(m):
        d = spectral_response(traj_a.spectra[r], traj_b.spectra[r])
        total += w[r] * (d ** 2)
    return float(total)


def trajectory_distance(traj_a: SpectralTrajectory,
                        traj_b: SpectralTrajectory,
                        weights: Optional[np.ndarray] = None) -> float:
    """Discrete trajectory metric sqrt(sum_r w_r ||Lambda_a - Lambda_b||_2^2)."""
    return float(np.sqrt(max(0.0, trajectory_distance_squared(traj_a, traj_b, weights))))


def trajectory_smoothness_penalty(traj: SpectralTrajectory) -> float:
    """Section 10 discrete smoothness penalty sum_r ||Delta Lambda_r||_2^2 / Delta tau_r^2.

    Smooth families give small values; discontinuous ones blow up.
    """
    m = len(traj)
    if m < 2:
        return 0.0
    total = 0.0
    for r in range(m - 1):
        d = spectral_response(traj.spectra[r], traj.spectra[r + 1])
        dtau = float(traj.tau_grid[r + 1] - traj.tau_grid[r])
        if dtau <= 0.0:
            raise ValueError("Trajectory tau_grid must be strictly increasing.")
        total += (d ** 2) / (dtau ** 2)
    return float(total)


def trajectory_arc_length(traj: SpectralTrajectory) -> float:
    m = len(traj)
    if m < 2:
        return 0.0
    total = 0.0
    for r in range(m - 1):
        total += spectral_response(traj.spectra[r], traj.spectra[r + 1])
    return float(total)

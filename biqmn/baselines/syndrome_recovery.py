"""Baseline A: classical syndrome-based recovery for the [[3,1,1]] repetition
codes, lifted to the Page-Wootters trajectory framework.

Given a noisy SpectralTrajectory (whose per-τ density is a 3-qubit data-register
state), this module runs the circuit-level syndrome-recovery channel on every
slice and returns the recovered trajectory plus aggregated syndrome / logical
fidelity statistics. The recovered slices are re-packaged as a SpectralTrajectory
so downstream admissibility/trajectory-distance code operates on them unchanged.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.encoding import (
    apply_syndrome_recovery,
    logical_state,
    syndrome_probabilities,
)
from ..core.graph_mapping import adjacency_from_reduced_density
from ..core.laplacian import graph_laplacian, ordered_spectrum
from ..core.trajectory import SpectralTrajectory


def logical_state_at_tau(code: str, amplitudes: Sequence[complex]) -> np.ndarray:
    """Convenience wrapper: ideal pure logical ket for the given code/amplitudes."""
    return logical_state(code, amplitudes)


def logical_fidelity(rho_data: np.ndarray, psi_L: np.ndarray) -> float:
    """Return ⟨ψ_L|ρ|ψ_L⟩ (real part clamped to [0, 1])."""
    rho = np.asarray(rho_data, dtype=complex)
    psi = np.asarray(psi_L, dtype=complex).reshape(-1)
    val = float(np.real(np.vdot(psi, rho @ psi)))
    return float(max(0.0, min(1.0, val)))


def _retarget_density_to_trajectory(rho_data: np.ndarray,
                                    n_system: int,
                                    mapping_mode: str) -> dict:
    W = adjacency_from_reduced_density(rho_data, n_system, mode=mapping_mode)
    L = graph_laplacian(W)
    return {
        "density": rho_data,
        "adjacency": W,
        "laplacian": L,
        "spectrum": ordered_spectrum(L),
    }


def apply_syndrome_recovery_to_trajectory(noisy_traj: SpectralTrajectory,
                                          code: str) -> SpectralTrajectory:
    """Return a new SpectralTrajectory whose per-τ densities are the syndrome-
    recovered versions of ``noisy_traj``'s densities.

    The τ grid, mapping mode, and graph construction match ``noisy_traj`` so the
    result is a drop-in replacement usable by the existing admissibility /
    trajectory-distance machinery.
    """
    n_system = int(noisy_traj.meta.get("n_system", 3))
    if n_system != 3:
        raise ValueError(
            "Syndrome recovery expects a 3-qubit system trajectory (n_system=3), "
            f"got n_system={n_system}."
        )
    mapping_mode = str(noisy_traj.meta.get("mapping_mode", "coherence_abs"))
    densities: list[np.ndarray] = []
    adjacency: list[np.ndarray] = []
    laplacians: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    for rho in noisy_traj.densities:
        rho_rec = apply_syndrome_recovery(np.asarray(rho, dtype=complex), code)
        built = _retarget_density_to_trajectory(rho_rec, n_system, mapping_mode)
        densities.append(built["density"])
        adjacency.append(built["adjacency"])
        laplacians.append(built["laplacian"])
        spectra.append(built["spectrum"])
    meta = dict(noisy_traj.meta)
    meta.update({
        "recovery_baseline": "syndrome",
        "recovery_code": str(code),
    })
    return SpectralTrajectory(
        tau_grid=np.array(noisy_traj.tau_grid, copy=True),
        densities=densities,
        adjacency=adjacency,
        laplacians=laplacians,
        spectra=spectra,
        meta=meta,
    )


def collect_syndrome_statistics(noisy_traj: SpectralTrajectory,
                                code: str) -> dict:
    """Per-slice syndrome probabilities + aggregate no-error rate."""
    per_slice = []
    for idx, rho in enumerate(noisy_traj.densities):
        sp = syndrome_probabilities(np.asarray(rho, dtype=complex), code)
        per_slice.append({
            "tau_index": int(idx),
            "probabilities": {str(key): float(val) for key, val in sp["probabilities"].items()},
            "no_error_probability": float(sp["no_error_probability"]),
        })
    mean_no_error = float(np.mean([item["no_error_probability"] for item in per_slice]))
    min_no_error = float(np.min([item["no_error_probability"] for item in per_slice]))
    return {
        "per_slice": per_slice,
        "mean_no_error_probability": mean_no_error,
        "min_no_error_probability": min_no_error,
    }


def trajectory_logical_fidelity(traj: SpectralTrajectory,
                                clean_traj: SpectralTrajectory) -> dict:
    """Per-τ ⟨ψ_clean(τ)|ρ(τ)|ψ_clean(τ)⟩ using the clean trajectory's pure
    density as the reference logical ket at each slice.

    The clean trajectory carries ρ_clean(τ) which for the encoded mode is the
    pure-state projector |ψ_L(τ)⟩⟨ψ_L(τ)|; we extract the top eigenvector as
    the reference ket.
    """
    if len(traj) != len(clean_traj):
        raise ValueError("Trajectories must share the same τ grid.")
    values: list[float] = []
    for rho, rho_clean in zip(traj.densities, clean_traj.densities):
        rho_clean_h = 0.5 * (np.asarray(rho_clean) + np.asarray(rho_clean).conj().T)
        vals, vecs = np.linalg.eigh(rho_clean_h)
        psi_L = vecs[:, int(np.argmax(vals.real))]
        values.append(logical_fidelity(rho, psi_L))
    values_arr = np.asarray(values, dtype=float)
    return {
        "per_slice": values_arr.tolist(),
        "mean": float(values_arr.mean()) if values_arr.size else 0.0,
        "min": float(values_arr.min()) if values_arr.size else 0.0,
        "max": float(values_arr.max()) if values_arr.size else 0.0,
    }

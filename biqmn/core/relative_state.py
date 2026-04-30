"""Page-Wootters conditional / relative state.

For a global density ρ on H_C ⊗ H_S and a clock projector P = |τ⟩⟨τ|_C,
    ρ_S(τ) = Tr_C[(P ⊗ I_S) ρ] / Tr[(P ⊗ I_S) ρ].
For a pure global state |Ψ⟩, the unnormalized conditional ket is ⟨τ|_C |Ψ⟩.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .clock import clock_state
from .metrics import fidelity


def _tensor_reshape(rho: np.ndarray, dim_c: int, dim_s: int) -> np.ndarray:
    return rho.reshape(dim_c, dim_s, dim_c, dim_s)


def partial_trace_clock(rho: np.ndarray,
                        dim_clock: int,
                        dim_system: int,
                        clock_proj: np.ndarray) -> np.ndarray:
    """Compute Tr_C[(P ⊗ I_S) ρ] for an arbitrary clock-side operator P.

    Parameters
    ----------
    rho        : (d_C·d_S, d_C·d_S) density matrix
    dim_clock  : d_C
    dim_system : d_S
    clock_proj : (d_C, d_C) operator acting on the clock register

    Returns
    -------
    (d_S, d_S) matrix (not normalized).
    """
    t = _tensor_reshape(rho, dim_clock, dim_system)
    tmp = np.einsum("ac,cbde->abde", clock_proj, t)
    return np.einsum("abae->be", tmp)


def slice_probability(global_rho: np.ndarray,
                      clock_ket: np.ndarray,
                      dim_clock: int,
                      dim_system: int) -> float:
    """Born weight of the clock event |tau><tau|_C for one relational slice."""
    proj = np.outer(clock_ket, clock_ket.conj())
    sigma = partial_trace_clock(global_rho, dim_clock, dim_system, proj)
    sigma = 0.5 * (sigma + sigma.conj().T)
    return float(np.trace(sigma).real)


def relative_state_density(global_rho: np.ndarray,
                           clock_ket: np.ndarray,
                           dim_clock: int,
                           dim_system: int,
                           normalize: bool = True,
                           eps: float = 1e-15,
                           allow_undefined: bool = False) -> np.ndarray:
    """ρ_S(τ) from a clock ket |τ⟩_C and the global density.

    When `normalize=False` the unnormalised conditional operator is returned; its
    trace is the probability of observing the clock in |τ⟩_C.
    """
    proj = np.outer(clock_ket, clock_ket.conj())
    sigma = partial_trace_clock(global_rho, dim_clock, dim_system, proj)
    sigma = 0.5 * (sigma + sigma.conj().T)  # Hermitise
    if not normalize:
        return sigma
    tr = float(np.trace(sigma).real)
    if tr < eps:
        if allow_undefined:
            return np.zeros_like(sigma)
        raise ValueError(
            "Clock-conditioned slice is undefined because its probability "
            f"is below eps={eps:.1e}."
        )
    return sigma / tr


def relative_state_pure(global_state: np.ndarray,
                        clock_ket: np.ndarray,
                        dim_clock: int,
                        dim_system: int,
                        normalize: bool = True) -> np.ndarray:
    """Pure-state branch: ⟨τ|_C |Ψ⟩.

    Uses |Ψ⟩ reshaped as a d_C × d_S matrix.  Result lives in H_S.
    """
    psi = global_state.reshape(dim_clock, dim_system)
    v = clock_ket.conj() @ psi
    if not normalize:
        return v
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def generate_tau_grid(Hc: np.ndarray,
                      tau_min: float,
                      tau_max: float,
                      n_tau: int,
                      scale_by_norm: bool = False) -> np.ndarray:
    """Build a τ grid.

    When `scale_by_norm=True`, the interval is rescaled by 1/‖H_C‖ so that the
    effective resolution matches Lemma 4.1 (Δτ ≳ 1/‖H_C‖).
    """
    grid = np.linspace(float(tau_min), float(tau_max), int(n_tau))
    if scale_by_norm:
        norm = float(np.linalg.norm(Hc, ord=2))
        if norm > 0:
            grid = grid / norm
    return grid


def build_relative_family(global_rho: np.ndarray,
                          Hc: np.ndarray,
                          tau_grid: Sequence[float],
                          psi0_clock: np.ndarray,
                          dim_clock: int,
                          dim_system: int) -> List[np.ndarray]:
    """Materialise {ρ_S(τ_r)} for every τ on the grid."""
    family: List[np.ndarray] = []
    for idx, tau in enumerate(tau_grid):
        ket = clock_state(Hc, float(tau), psi0_clock)
        try:
            family.append(
                relative_state_density(global_rho, ket, dim_clock, dim_system)
            )
        except ValueError as exc:
            prob = slice_probability(global_rho, ket, dim_clock, dim_system)
            raise ValueError(
                f"Undefined relative slice at index={idx}, tau={float(tau):.6g}, "
                f"probability={prob:.3e}."
            ) from exc
    return family


def check_slice_sanity(rho_s: np.ndarray, tol: float = 1e-8) -> dict:
    """Trace / hermiticity / positivity health-check for one slice."""
    trace = float(np.trace(rho_s).real)
    herm_err = float(np.linalg.norm(rho_s - rho_s.conj().T))
    eigs = np.linalg.eigvalsh(0.5 * (rho_s + rho_s.conj().T)).real
    min_eig = float(eigs.min())
    return {
        "trace": trace,
        "trace_err": abs(trace - 1.0),
        "hermiticity_err": herm_err,
        "min_eig": min_eig,
        "psd_violation": max(0.0, -min_eig),
        "is_valid": abs(trace - 1.0) < tol and herm_err < tol and min_eig > -tol,
    }


def adjacent_fidelity(relative_family: Sequence[np.ndarray]) -> np.ndarray:
    """Uhlmann fidelity between adjacent relative density slices."""
    out = np.zeros(len(relative_family) - 1)
    for r in range(len(relative_family) - 1):
        out[r] = fidelity(relative_family[r], relative_family[r + 1])
    return out

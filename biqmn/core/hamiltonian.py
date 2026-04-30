"""Clock, system, and total Hamiltonians.

Implements the construction required by Proposition 1.1 of TheFirstThoery.tex:
the null space of H_tot = H_C ⊗ I + I ⊗ H_S is non-empty iff
Spec(H_C) ∩ Spec(-H_S) ≠ ∅.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.linalg import eigh

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_list(ops: Iterable[np.ndarray]) -> np.ndarray:
    ops = list(ops)
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_site(n: int, site: int, pauli: np.ndarray) -> np.ndarray:
    """Embed a single-qubit operator on `site` inside an `n`-qubit register."""
    ops = [PAULI_I] * n
    ops[site] = pauli
    return _kron_list(ops)


def two_site(n: int, i: int, j: int, p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    ops = [PAULI_I] * n
    ops[i] = p_i
    ops[j] = p_j
    return _kron_list(ops)


def _broadcast(x, length: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.size == 1:
        arr = np.full(length, arr.item())
    if arr.size != length:
        raise ValueError(f"Expected length-{length} parameter, got {arr.size}")
    return arr


def build_transverse_ising(n: int, params: dict) -> np.ndarray:
    """Transverse-field Ising chain with tunable biases.

    H = Σ_i hx_i X_i + Σ_i hz_i Z_i + Σ_<ij> J_i Z_i Z_{i+1}

    params keys (optional):
        hx : float | list[float] - transverse field (default 1.0)
        hz : float | list[float] - longitudinal bias  (default 0.0)
        J  : float | list[float] - nearest-neighbor ZZ coupling (default 0.0)
    """
    if n <= 0:
        raise ValueError("Hamiltonian requires at least one qubit.")
    hx = _broadcast(params.get("hx", 1.0), n)
    hz = _broadcast(params.get("hz", 0.0), n)
    J = _broadcast(params.get("J", 0.0), max(n - 1, 1))

    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        H += hx[i] * single_site(n, i, PAULI_X)
        H += hz[i] * single_site(n, i, PAULI_Z)
    for i in range(n - 1):
        H += J[i] * two_site(n, i, i + 1, PAULI_Z, PAULI_Z)
    # Ensure Hermitian numerically
    return 0.5 * (H + H.conj().T)


def build_clock_hamiltonian(n_clock: int, params: dict) -> np.ndarray:
    """Clock Hamiltonian (defaults to transverse-Ising form)."""
    return build_transverse_ising(n_clock, params)


def build_system_hamiltonian(n_system: int, params: dict) -> np.ndarray:
    """System Hamiltonian (defaults to transverse-Ising form)."""
    return build_transverse_ising(n_system, params)


def build_total_hamiltonian(Hc: np.ndarray, Hs: np.ndarray) -> np.ndarray:
    """H_tot = H_C ⊗ I_S + I_C ⊗ H_S (clock index is the leftmost in kron)."""
    dc = Hc.shape[0]
    ds = Hs.shape[0]
    return np.kron(Hc, np.eye(ds, dtype=complex)) + np.kron(np.eye(dc, dtype=complex), Hs)


def check_nullspace(Htot: np.ndarray, tol: float = 1e-9) -> dict:
    """Detect the null space of H_tot.

    Returns a dict with:
        dim           : dimension of kernel
        eigvals       : full eigenspectrum (ascending)
        null_basis    : orthonormal basis of Ker(H_tot)  (d_tot × dim)
        min_abs_eig   : smallest |eigenvalue|
        gap           : distance from zero to the nearest non-null level
    """
    H = 0.5 * (Htot + Htot.conj().T)
    eigvals, eigvecs = eigh(H)
    mask = np.abs(eigvals) < tol
    null_basis = eigvecs[:, mask]
    nonnull = np.abs(eigvals)[~mask]
    return {
        "dim": int(mask.sum()),
        "eigvals": eigvals,
        "null_basis": null_basis,
        "min_abs_eig": float(np.min(np.abs(eigvals))),
        "gap": float(nonnull.min()) if nonnull.size else float("inf"),
    }


def clock_energy_variance(state: np.ndarray, Hc: np.ndarray) -> float:
    """(ΔH_C)² = ⟨H_C²⟩ - ⟨H_C⟩² for a pure clock state."""
    s = state / (np.linalg.norm(state) + 1e-30)
    mean = np.vdot(s, Hc @ s).real
    sq = np.vdot(s, Hc @ (Hc @ s)).real
    return float(max(0.0, sq - mean * mean))


def clock_operator_norm(Hc: np.ndarray) -> float:
    """‖H_C‖₂ — used for Lemma 4.1 time-resolution bound Δt ≳ 1/‖H_C‖."""
    return float(np.linalg.norm(Hc, ord=2))


def min_energy_gap(H: np.ndarray) -> float:
    """Smallest non-zero gap in the spectrum — used for Lemma 4.2 recurrence T ≲ 2π/ΔE_min."""
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (H + H.conj().T)).real)
    diffs = np.diff(eigs)
    pos = diffs[diffs > 1e-12]
    return float(pos.min()) if pos.size else 0.0


def align_clock_system_spectra(Hc: np.ndarray,
                               Hs: np.ndarray,
                               tol: float = 1e-9) -> Optional[float]:
    """Return a shift δ such that Spec(Hc) ∩ Spec(-(Hs + δ I)) ≠ ∅, or None.

    Shifting H_S by +δ I does not affect the system dynamics up to a global phase,
    and is the cheapest way to engineer a non-trivial null space.
    """
    sc = np.linalg.eigvalsh(0.5 * (Hc + Hc.conj().T)).real
    ss = np.linalg.eigvalsh(0.5 * (Hs + Hs.conj().T)).real
    # Want  e_c = -(e_s + δ)  ⇒  δ = -e_c - e_s
    candidates = sorted({-float(a) - float(b) for a in sc for b in ss})
    # Prefer small-magnitude shifts
    candidates.sort(key=lambda x: abs(x))
    for delta in candidates:
        overlap = any(
            any(abs(a - (-(b + delta))) < tol for b in ss) for a in sc
        )
        if overlap:
            return float(delta)
    return None

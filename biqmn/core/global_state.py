"""Global timeless state |Ψ⟩ (Wheeler-DeWitt / Page-Wootters).

The kernel of H_tot is the admissible configuration space for the relational
theory.  This module exposes (a) the kernel-based construction and
(b) hand-crafted entangled test states useful for debugging slicing.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def build_global_null_state(Htot: np.ndarray,
                            mode: str = "ground_null",
                            tol: float = 1e-9,
                            coeffs: np.ndarray | None = None) -> np.ndarray:
    """Select a vector from (approximately) Ker(H_tot).

    Modes
    -----
    ground_null : lowest-|eigenvalue| eigenvector (the unique state if the kernel
                  is 1-dimensional; otherwise the first basis vector).
    uniform_null : equal-amplitude superposition over all kernel basis vectors.
    custom       : combine kernel basis vectors using the provided `coeffs`.
    lowest_abs   : plain lowest-|eigenvalue| vector, even if it is not exactly null
                   (falls back when the exact kernel is empty).
    """
    H = 0.5 * (Htot + Htot.conj().T)
    eigvals, eigvecs = eigh(H)
    mask = np.abs(eigvals) < tol
    null_basis = eigvecs[:, mask]

    if mode == "lowest_abs" or not mask.any():
        idx = int(np.argmin(np.abs(eigvals)))
        v = eigvecs[:, idx]
        return v / np.linalg.norm(v)

    if mode == "ground_null":
        v = null_basis[:, 0]
    elif mode == "uniform_null":
        c = np.ones(null_basis.shape[1], dtype=complex) / np.sqrt(null_basis.shape[1])
        v = null_basis @ c
    elif mode == "custom":
        if coeffs is None or coeffs.size != null_basis.shape[1]:
            raise ValueError("mode='custom' requires coeffs of length dim(Ker)")
        v = null_basis @ coeffs.astype(complex)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return v / (np.linalg.norm(v) + 1e-30)


def build_manual_entangled_state(n_clock: int,
                                 n_system: int,
                                 recipe: str = "uniform_bell") -> np.ndarray:
    """Construct hand-built clock–system entangled states for debugging.

    Recipes
    -------
    uniform_bell : Σ_k |k⟩_C ⊗ |k⟩_S (truncated to min(d_C, d_S)).
    ghz_like     : (|0⟩_C|0⟩_S + |d_C-1⟩_C|d_S-1⟩_S) / √2.
    product_plus : (|+⟩_C)^{n_c} ⊗ (|+⟩_S)^{n_s}.
    """
    dc = 2 ** n_clock
    ds = 2 ** n_system
    dtot = dc * ds

    if recipe == "uniform_bell":
        d = min(dc, ds)
        v = np.zeros(dtot, dtype=complex)
        for k in range(d):
            v[k * ds + k] = 1.0
    elif recipe == "ghz_like":
        v = np.zeros(dtot, dtype=complex)
        v[0] = 1.0
        v[(dc - 1) * ds + (ds - 1)] = 1.0
    elif recipe == "product_plus":
        plus = np.ones(2, dtype=complex) / np.sqrt(2.0)
        clock = plus.copy()
        for _ in range(n_clock - 1):
            clock = np.kron(clock, plus)
        sys = plus.copy()
        for _ in range(n_system - 1):
            sys = np.kron(sys, plus)
        v = np.kron(clock, sys)
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

    return v / (np.linalg.norm(v) + 1e-30)


def build_encoded_entangled_state(n_clock: int,
                                  n_system: int,
                                  code: str,
                                  amplitudes: np.ndarray | None = None) -> np.ndarray:
    """Global clock-logical-Bell state for a 3-qubit repetition code.

    For a 1-qubit clock the state is
        |Ψ⟩ = c_0 |0⟩_C ⊗ |0_L⟩_S + c_1 |1⟩_C ⊗ |1_L⟩_S
    where (|0_L⟩, |1_L⟩) is the encoding basis for `code` ∈ {bitflip, phaseflip}.
    """
    from .encoding import logical_basis  # local import avoids circularity

    if int(n_clock) != 1:
        raise NotImplementedError(
            "Encoded repetition-code preparation currently supports a 1-qubit clock."
        )
    if int(n_system) != 3:
        raise ValueError(
            f"[[3,1,1]] code requires n_system=3, got {n_system}."
        )
    psi_0L, psi_1L = logical_basis(code)
    if amplitudes is None:
        coeffs = np.array([1.0, 1.0], dtype=complex)
    else:
        coeffs = np.asarray(amplitudes, dtype=complex).reshape(-1)
    if coeffs.size != 2:
        raise ValueError(
            f"Expected two logical amplitudes for a 1-qubit clock, got {coeffs.size}."
        )
    ds = 2 ** int(n_system)
    state = np.zeros(2 * ds, dtype=complex)
    state[0:ds] = coeffs[0] * psi_0L
    state[ds:2 * ds] = coeffs[1] * psi_1L
    norm = float(np.linalg.norm(state))
    if norm < 1.0e-14:
        raise ValueError("Encoded amplitudes collapse to the zero vector.")
    return state / norm


def to_density_matrix(state: np.ndarray) -> np.ndarray:
    s = state.reshape(-1, 1).astype(complex)
    rho = s @ s.conj().T
    # Symmetrize (numerical safety)
    return 0.5 * (rho + rho.conj().T)


def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)

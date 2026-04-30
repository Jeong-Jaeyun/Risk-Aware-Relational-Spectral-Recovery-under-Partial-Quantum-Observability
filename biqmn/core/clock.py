"""Clock states and geometric distances on the clock Hilbert space.

|τ⟩_C = U(τ)|ψ₀⟩_C = exp(−i H_C τ) |ψ₀⟩_C (Stone's theorem; §2 of TheFirstThoery.tex).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _validated_clock_state_inputs(Hc: np.ndarray,
                                  psi0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(Hc, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Clock Hamiltonian must be a square matrix.")
    if not np.allclose(H, H.conj().T):
        raise ValueError("Clock Hamiltonian must be Hermitian.")
    v = np.asarray(psi0, dtype=complex).reshape(-1)
    if v.size != H.shape[0]:
        raise ValueError(
            f"Clock initial state has dimension {v.size}, expected {H.shape[0]}."
        )
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("Clock initial state must have non-zero norm.")
    return H, v / norm


def clock_state(Hc: np.ndarray, tau: float, psi0: np.ndarray) -> np.ndarray:
    H, v0 = _validated_clock_state_inputs(Hc, psi0)
    U = expm(-1j * H * float(tau))
    v = U @ v0
    return v / (np.linalg.norm(v) + 1e-30)


def clock_overlap(Hc: np.ndarray, tau_a: float, tau_b: float, psi0: np.ndarray) -> complex:
    a = clock_state(Hc, tau_a, psi0)
    b = clock_state(Hc, tau_b, psi0)
    return complex(np.vdot(a, b))


def clock_geom_distance(Hc: np.ndarray, tau_a: float, tau_b: float,
                        psi0: np.ndarray) -> float:
    """Clock-label distinguishability 1 - |<tau_a|tau_b>|^2 from Section 11."""
    ov = clock_overlap(Hc, tau_a, tau_b, psi0)
    return float(max(0.0, 1.0 - abs(ov) ** 2))


def clock_bures_distance(Hc: np.ndarray, tau_a: float, tau_b: float,
                         psi0: np.ndarray) -> float:
    """Bures distance √(2(1 − |⟨τ_a|τ_b⟩|)) on pure states."""
    ov = clock_overlap(Hc, tau_a, tau_b, psi0)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - abs(ov)))))


def default_clock_initial_state(n_clock: int, kind: str = "plus") -> np.ndarray:
    """Convenient initial clock kets |ψ₀⟩_C."""
    if int(n_clock) <= 0:
        raise ValueError("Clock register must contain at least one qubit.")
    dim = 2 ** n_clock
    if kind == "plus":
        v = np.ones(dim, dtype=complex) / np.sqrt(dim)
    elif kind == "ground":
        v = np.zeros(dim, dtype=complex)
        v[0] = 1.0
    elif kind == "random":
        rng = np.random.default_rng()
        v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        v /= np.linalg.norm(v)
    else:
        raise ValueError(f"Unknown clock initial kind: {kind}")
    return v

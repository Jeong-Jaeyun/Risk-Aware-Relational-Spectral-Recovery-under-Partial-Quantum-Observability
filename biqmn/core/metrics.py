"""Cross-module metric helpers.

Exposed here (as opposed to buried in spectral_density / graph_mapping) so
that experiments can import a single place for generic fidelity / distance
computations on quantum and spectral objects.
"""
from __future__ import annotations

import numpy as np


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _project_psd(matrix: np.ndarray, *, renormalize: bool = False) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_hermitian_part(matrix))
    clipped = np.clip(vals.real, 0.0, None)
    projected = vecs @ np.diag(clipped) @ vecs.conj().T
    projected = _hermitian_part(projected)
    if renormalize:
        trace = float(np.trace(projected).real)
        if trace > 0.0:
            projected = projected / trace
    return projected


def _psd_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_hermitian_part(matrix))
    sqrt_vals = np.sqrt(np.clip(vals.real, 0.0, None))
    return _hermitian_part(vecs @ np.diag(sqrt_vals) @ vecs.conj().T)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    diff = _hermitian_part(diff)
    vals = np.linalg.eigvalsh(diff).real
    return float(0.5 * np.sum(np.abs(vals)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(ρ, σ) = [Tr √(√ρ σ √ρ)]²."""
    r = _project_psd(rho, renormalize=True)
    s = _project_psd(sigma, renormalize=True)
    sqrt_r = _psd_matrix_sqrt(r)
    inner = _hermitian_part(sqrt_r @ s @ sqrt_r)
    vals = np.linalg.eigvalsh(inner).real
    amplitude = float(np.sum(np.sqrt(np.clip(vals, 0.0, None))))
    return float(max(0.0, min(1.0, amplitude ** 2)))


def pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    ov = complex(np.vdot(psi, phi))
    return float(abs(ov) ** 2)


def spectral_l2(lam_a: np.ndarray, lam_b: np.ndarray) -> float:
    a = np.sort(np.asarray(lam_a).real)
    b = np.sort(np.asarray(lam_b).real)
    m = min(a.size, b.size)
    return float(np.linalg.norm(a[:m] - b[:m]))


def spectral_linf(lam_a: np.ndarray, lam_b: np.ndarray) -> float:
    a = np.sort(np.asarray(lam_a).real)
    b = np.sort(np.asarray(lam_b).real)
    m = min(a.size, b.size)
    return float(np.max(np.abs(a[:m] - b[:m]))) if m > 0 else 0.0


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    f = fidelity(rho, sigma)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - np.sqrt(max(0.0, f))))))

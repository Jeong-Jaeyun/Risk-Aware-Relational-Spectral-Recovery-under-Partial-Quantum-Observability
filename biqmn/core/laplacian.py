"""Graph Laplacian utilities.

L = D − W, with D = diag(row_sum(W)).  For non-negative symmetric W the
Laplacian is PSD and its kernel contains the all-ones vector.  Eq.(10)--(11)
of TheFirstThoery.tex are the spectral responses used downstream.
"""
from __future__ import annotations

import numpy as np


def degree_matrix(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(axis=1))


def graph_laplacian(W: np.ndarray) -> np.ndarray:
    L = degree_matrix(W) - W
    return 0.5 * (L + L.T)


def laplacian_validity_penalty(L: np.ndarray, tol: float = 1e-8) -> dict:
    """Section 10.7 structural penalty terms for a valid combinatorial Laplacian."""
    sym_err = float(np.linalg.norm(L - L.T, ord="fro"))
    row_err = float(np.linalg.norm(L.sum(axis=1), ord=2))
    eigs = np.linalg.eigvalsh(0.5 * (L + L.T)).real
    neg_eigs = np.minimum(eigs, 0.0)
    neg_eig_l2 = float(np.linalg.norm(neg_eigs, ord=2))
    min_eig = float(eigs.min())
    offdiag = np.asarray(L, dtype=float) - np.diag(np.diag(np.asarray(L, dtype=float)))
    offdiag_positive = float(np.linalg.norm(np.maximum(offdiag, 0.0), ord="fro"))
    diag_consistency = float(np.linalg.norm(np.diag(L) + np.sum(offdiag, axis=1), ord=2))
    total = (
        (sym_err ** 2)
        + (neg_eig_l2 ** 2)
        + (row_err ** 2)
        + (offdiag_positive ** 2)
        + (diag_consistency ** 2)
    )
    return {
        "symmetry_err": sym_err,
        "rowsum_l2": row_err,
        "min_eig": min_eig,
        "negative_eig_l2": neg_eig_l2,
        "offdiag_positive_fro": offdiag_positive,
        "diag_consistency_l2": diag_consistency,
        "total": total,
        "is_valid": (
            sym_err < tol
            and row_err < tol
            and neg_eig_l2 < tol
            and offdiag_positive < tol
            and diag_consistency < tol
        ),
    }


def ordered_spectrum(L: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(0.5 * (L + L.T)).real)


def fiedler_value(L: np.ndarray) -> float:
    """Second smallest eigenvalue (algebraic connectivity)."""
    spec = ordered_spectrum(L)
    return float(spec[1]) if spec.size >= 2 else float("nan")


def cheeger_bounds(L: np.ndarray) -> dict:
    """Cheeger inequality bracket:  λ₂/2 ≤ h(G) ≤ √(2 λ₂ λ_max)."""
    spec = ordered_spectrum(L)
    if spec.size < 2:
        return {"lower": float("nan"), "upper": float("nan")}
    lam2 = float(spec[1])
    lam_max = float(spec[-1])
    return {"lower": lam2 / 2.0, "upper": float(np.sqrt(2.0 * lam2 * lam_max))}


def weyl_shift_estimate(L0: np.ndarray, L_noise: np.ndarray) -> dict:
    """Verify the Weyl inequality bracket for the (L0, L0 + ΔL) pair."""
    dL = L_noise - L0
    dL = 0.5 * (dL + dL.T)
    e_dL = np.linalg.eigvalsh(dL).real
    spec0 = ordered_spectrum(L0)
    spec1 = ordered_spectrum(L_noise)
    lmin = float(e_dL.min())
    lmax = float(e_dL.max())
    lower = spec0 + lmin
    upper = spec0 + lmax
    tight_lower = float(np.min(spec1 - lower))
    tight_upper = float(np.max(spec1 - upper))
    return {
        "delta_min": lmin,
        "delta_max": lmax,
        "slack_lower": tight_lower,  # should be ≥ 0
        "slack_upper": tight_upper,  # should be ≤ 0
        "frobenius": float(np.linalg.norm(dL, ord="fro")),
    }

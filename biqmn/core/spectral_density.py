"""Spectral density p_τ(λ) and divergence functionals.

Implements the clock-conditioned empirical spectral density, its Gaussian-KDE
approximation, and the detection functionals used in §8 (KL divergence) and
§8.6 (hybrid KL + Wasserstein) of TheFirstThoery.tex.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _grid_dx(grid: np.ndarray) -> float:
    g = np.asarray(grid).ravel()
    if g.size < 2:
        return 1.0
    return float(g[1] - g[0])


def kde_spectral_density(eigs: np.ndarray,
                         grid: np.ndarray,
                         sigma: float,
                         normalize: bool = True) -> np.ndarray:
    """Gaussian-KDE approximation of p_τ(λ) = (1/n) Σ δ(λ − λ_k)."""
    e = np.asarray(eigs).real.reshape(-1, 1)
    g = np.asarray(grid).real.reshape(1, -1)
    if sigma <= 0:
        raise ValueError("KDE bandwidth sigma must be strictly positive.")
    w = np.exp(-0.5 * ((g - e) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    p = w.mean(axis=0)
    if normalize:
        dx = _grid_dx(grid)
        z = p.sum() * dx
        if z <= 0:
            return np.zeros_like(p)
        p = p / z
    return p


def auto_bandwidth(eigs: np.ndarray, factor: float = 1.06) -> float:
    """Silverman's rule of thumb.  Guarantees σ > 0 even on tiny samples."""
    x = np.asarray(eigs).real
    if x.size < 2:
        return max(float(factor), 1e-3)
    sigma = float(np.std(x))
    return max(factor * sigma * x.size ** (-1.0 / 5.0), 1e-6)


def auto_grid(eigs_list: list[np.ndarray],
              n_grid: int = 256,
              pad_frac: float = 0.1) -> np.ndarray:
    """Common spectral grid wide enough to contain every input spectrum."""
    if not eigs_list:
        return np.linspace(-1.0, 1.0, int(n_grid))
    lo = min(float(np.min(e)) for e in eigs_list)
    hi = max(float(np.max(e)) for e in eigs_list)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return np.linspace(lo - pad_frac * span, hi + pad_frac * span, int(n_grid))


def _safe(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, None)


def kl_divergence(p: np.ndarray,
                  q: np.ndarray,
                  dx: float = 1.0,
                  eps: float = 1e-12) -> float:
    """Discrete KL divergence D_KL(p‖q) on a uniform grid (weight dx)."""
    pp = _safe(p, eps)
    qq = _safe(q, eps)
    return float(np.sum(pp * np.log(pp / qq)) * dx)


def js_divergence(p: np.ndarray,
                  q: np.ndarray,
                  dx: float = 1.0,
                  eps: float = 1e-12) -> float:
    """Symmetric Jensen–Shannon divergence (values in [0, log 2])."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, dx, eps) + 0.5 * kl_divergence(q, m, dx, eps)


def wasserstein_1(grid: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """W₁ on the real line via |CDF_p − CDF_q| (uniform grid assumed)."""
    dx = _grid_dx(grid)
    P = np.cumsum(p) * dx
    Q = np.cumsum(q) * dx
    return float(np.sum(np.abs(P - Q)) * dx)


def total_variation(p: np.ndarray, q: np.ndarray, dx: float = 1.0) -> float:
    return float(0.5 * np.sum(np.abs(p - q)) * dx)


def normalize_density(p: np.ndarray, dx: float = 1.0, eps: float = 1e-30) -> np.ndarray:
    z = float(np.sum(p) * dx)
    if z < eps:
        return np.zeros_like(p)
    return p / z


def hybrid_detect_score(p: np.ndarray,
                        q: np.ndarray,
                        grid: np.ndarray,
                        alpha: float = 0.5,
                        kl_scale: float = 1.0,
                        w1_scale: float = 1.0) -> dict:
    """α·KL_norm + (1−α)·W1_norm hybrid (cf. §8.6)."""
    dx = _grid_dx(grid)
    kl = kl_divergence(p, q, dx)
    js = js_divergence(p, q, dx)
    w1 = wasserstein_1(grid, p, q)
    tv = total_variation(p, q, dx)
    kl_norm = kl / max(kl_scale, 1e-15)
    w1_norm = w1 / max(w1_scale, 1e-15)
    score = float(alpha * kl_norm + (1.0 - alpha) * w1_norm)
    return {
        "kl": kl,
        "kl_norm": kl_norm,
        "js": js,
        "w1": w1,
        "w1_norm": w1_norm,
        "tv": tv,
        "score": score,
    }


def band_wasserstein(grid: np.ndarray,
                     p: np.ndarray,
                     q: np.ndarray,
                     band: Tuple[float, float]) -> float:
    """Localised W₁ inside a spectral band [λ_lo, λ_hi]."""
    lo, hi = band
    mask = (grid >= lo) & (grid <= hi)
    if not mask.any():
        return 0.0
    dx = _grid_dx(grid)
    P = np.cumsum(p) * dx
    Q = np.cumsum(q) * dx
    return float(np.sum(np.abs(P[mask] - Q[mask])) * dx)

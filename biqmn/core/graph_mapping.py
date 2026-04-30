"""Map reduced system densities to weighted relational graphs.

The paper leaves the pairwise relational functional open. This module
implements several concrete choices:

- ``coherence_abs``: l1 off-diagonal coherence in the computational basis
- ``coherence_abs_xbasis``: the same observable after a local Hadamard rotation
- ``mutual_info``: von-Neumann mutual-information proxy
- ``correlation``: Frobenius total-correlation proxy

All weights are non-negative, symmetric, and zero on the diagonal, so the
resulting graph Laplacian ``L = D - W`` is always PSD.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List

import numpy as np


_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
_HADAMARD_2Q = np.kron(_HADAMARD, _HADAMARD)


def _partial_trace_all_but(rho: np.ndarray, n_qubits: int, keep: Iterable[int]) -> np.ndarray:
    """Trace out qubits not in ``keep``. Qubit 0 is leftmost in kron order."""
    keep = sorted(set(int(k) for k in keep))
    if any(k < 0 or k >= n_qubits for k in keep):
        raise ValueError(f"keep={keep} out of [0,{n_qubits})")
    dims = [2] * n_qubits
    tensor = rho.reshape(dims + dims)
    current = list(range(n_qubits))
    to_trace = sorted(set(current) - set(keep), reverse=True)
    for q in to_trace:
        idx_bra = current.index(q)
        idx_ket = idx_bra + len(current)
        tensor = np.trace(tensor, axis1=idx_bra, axis2=idx_ket)
        current.remove(q)
    d_keep = 2 ** len(keep)
    return tensor.reshape(d_keep, d_keep)


def reduced_one_qubit(rho_s: np.ndarray, n_system: int, i: int) -> np.ndarray:
    return _partial_trace_all_but(rho_s, n_system, [i])


def reduced_two_qubit(rho_s: np.ndarray, n_system: int, i: int, j: int) -> np.ndarray:
    return _partial_trace_all_but(rho_s, n_system, [i, j])


def _coherence_abs_weight(rho_ij: np.ndarray) -> float:
    off = rho_ij - np.diag(np.diag(rho_ij))
    return float(np.sum(np.abs(off)))


def _coherence_abs_xbasis_weight(rho_ij: np.ndarray) -> float:
    rotated = _HADAMARD_2Q @ rho_ij @ _HADAMARD_2Q.conj().T
    return _coherence_abs_weight(rotated)


def _von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    eigs = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    eigs = eigs[eigs > eps]
    return float(-np.sum(eigs * np.log(eigs)))


def _mutual_info_weight(rho_ij: np.ndarray, rho_i: np.ndarray, rho_j: np.ndarray) -> float:
    return max(
        0.0,
        _von_neumann_entropy(rho_i)
        + _von_neumann_entropy(rho_j)
        - _von_neumann_entropy(rho_ij),
    )


def _correlation_weight(rho_ij: np.ndarray, rho_i: np.ndarray, rho_j: np.ndarray) -> float:
    return float(np.linalg.norm(rho_ij - np.kron(rho_i, rho_j), ord="fro"))


def adjacency_from_reduced_density(
    rho_s: np.ndarray,
    n_system: int,
    mode: str = "coherence_abs",
) -> np.ndarray:
    """Build the ``n_system x n_system`` symmetric weight matrix ``W``."""
    W = np.zeros((n_system, n_system), dtype=float)
    singles: List[np.ndarray] = []
    if mode in ("mutual_info", "correlation"):
        singles = [reduced_one_qubit(rho_s, n_system, i) for i in range(n_system)]
    for i, j in combinations(range(n_system), 2):
        rho_ij = reduced_two_qubit(rho_s, n_system, i, j)
        if mode == "coherence_abs":
            w = _coherence_abs_weight(rho_ij)
        elif mode == "coherence_abs_xbasis":
            w = _coherence_abs_xbasis_weight(rho_ij)
        elif mode == "mutual_info":
            w = _mutual_info_weight(rho_ij, singles[i], singles[j])
        elif mode == "correlation":
            w = _correlation_weight(rho_ij, singles[i], singles[j])
        else:
            raise ValueError(f"Unknown mapping mode: {mode!r}")
        W[i, j] = w
        W[j, i] = w
    return W


def adjacency_family(
    rho_family: Iterable[np.ndarray],
    n_system: int,
    mode: str = "coherence_abs",
) -> List[np.ndarray]:
    return [adjacency_from_reduced_density(r, n_system, mode=mode) for r in rho_family]

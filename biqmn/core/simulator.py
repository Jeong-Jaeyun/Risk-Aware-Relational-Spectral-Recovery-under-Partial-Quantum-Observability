"""Simulation backends for global density-matrix generation.

The theory code works at the matrix level after the global state has been
prepared.  This module provides two interchangeable ways to obtain that global
density matrix:

- ``linear_algebra``: direct NumPy/SciPy state -> density conversion plus
  explicit Kraus application.
- ``qiskit_aer``: prepare the state in ``AerSimulator(method="density_matrix")``
  and apply the same single-qubit channels as Kraus instructions.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .global_state import to_density_matrix
from .noise import apply_noise_schedule, kraus_ops_for_step


SUPPORTED_BACKENDS = {"linear_algebra", "qiskit_aer"}


def _symmetrize_density(rho: np.ndarray) -> np.ndarray:
    out = 0.5 * (rho + rho.conj().T)
    trace = complex(np.trace(out))
    if abs(trace) > 0.0:
        out = out / trace
    return out


def _validated_backend(name: str) -> str:
    backend = str(name).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported simulation backend {name!r}; "
            f"expected one of {sorted(SUPPORTED_BACKENDS)}."
        )
    return backend


def _global_site_to_qiskit_qubit(site_global: int, n_total: int) -> int:
    return int(n_total - 1 - site_global)

def simulate_global_density(
    state: np.ndarray,
    schedule: Iterable[dict],
    *,
    n_clock: int,
    n_system: int,
    backend: str = "linear_algebra",
) -> np.ndarray:
    """Return the global density matrix using the selected simulation backend."""
    engine = _validated_backend(backend)
    psi = np.asarray(state, dtype=complex).reshape(-1)
    noise_schedule = list(schedule)
    if engine == "linear_algebra":
        rho = to_density_matrix(psi)
        if noise_schedule:
            rho = apply_noise_schedule(rho, noise_schedule, n_clock, n_system)
        return _symmetrize_density(rho)
    return _simulate_global_density_qiskit_aer(
        psi,
        noise_schedule,
        n_clock=n_clock,
        n_system=n_system,
    )


def _simulate_global_density_qiskit_aer(
    state: np.ndarray,
    schedule: list[dict],
    *,
    n_clock: int,
    n_system: int,
) -> np.ndarray:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Kraus
        from qiskit_aer import AerSimulator
        from qiskit_aer.library import SaveDensityMatrix, SetStatevector
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "qiskit_aer backend requested, but qiskit/qiskit-aer is unavailable."
        ) from exc

    n_total = int(n_clock + n_system)
    qc = QuantumCircuit(n_total)
    qc.append(SetStatevector(state), list(range(n_total)))
    for step in schedule:
        site_global = n_clock + int(step["qubit"])
        qiskit_qubit = _global_site_to_qiskit_qubit(site_global, n_total)
        qc.append(Kraus(kraus_ops_for_step(step)).to_instruction(), [qiskit_qubit])
    qc.append(SaveDensityMatrix(n_total), list(range(n_total)))

    backend = AerSimulator(method="density_matrix")
    result = backend.run(qc).result()
    raw = result.data(0)["density_matrix"]
    rho = np.asarray(raw.data if hasattr(raw, "data") else raw, dtype=complex)
    return _symmetrize_density(rho)

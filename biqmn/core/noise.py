"""Kraus / unitary noise channels applied to a global density matrix.

All channels act on a single qubit of the system register. The global register
is ordered `(clock, system)`, so the embedded site index is `n_clock + qubit`.

Schedule convention:
- stochastic channels interpret `step["p"]` as a probability in `[0, 1]`
- coherent channels interpret `step["p"]` as a rotation angle `theta` in radians
"""
from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np

from .hamiltonian import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z


def _embed_single(op2: np.ndarray, n_total: int, site: int) -> np.ndarray:
    ops: List[np.ndarray] = [PAULI_I] * n_total
    ops[site] = op2
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _apply_kraus_on_site(
    rho: np.ndarray,
    kraus_ops: Iterable[np.ndarray],
    site_global: int,
    n_total: int,
) -> np.ndarray:
    out = np.zeros_like(rho)
    for op2 in kraus_ops:
        op = _embed_single(op2, n_total, site_global)
        out += op @ rho @ op.conj().T
    return 0.5 * (out + out.conj().T)


def _resolve_site(qubit: int, n_clock: int, n_system: int) -> int:
    if not 0 <= qubit < n_system:
        raise ValueError(f"System qubit index {qubit} out of range [0,{n_system})")
    return n_clock + qubit


def _validated_probability(p: float, *, kind: str) -> float:
    value = float(p)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{kind} expects probability in [0, 1], got {value!r}")
    return value


def _rotation_unitary(pauli: np.ndarray, theta: float) -> np.ndarray:
    angle = float(theta)
    return np.cos(angle / 2.0) * PAULI_I - 1j * np.sin(angle / 2.0) * pauli


def kraus_ops_for_step(step: dict) -> list[np.ndarray]:
    kind = str(step["kind"])
    raw = float(step["p"])
    if kind == "bitflip":
        p = _validated_probability(raw, kind=kind)
        return [np.sqrt(1.0 - p) * PAULI_I, np.sqrt(p) * PAULI_X]
    if kind == "phaseflip":
        p = _validated_probability(raw, kind=kind)
        return [np.sqrt(1.0 - p) * PAULI_I, np.sqrt(p) * PAULI_Z]
    if kind == "dephasing":
        p = _validated_probability(raw, kind=kind)
        return [np.sqrt(1.0 - p / 2.0) * PAULI_I, np.sqrt(p / 2.0) * PAULI_Z]
    if kind == "depolarizing":
        p = _validated_probability(raw, kind=kind)
        return [
            np.sqrt(1.0 - p) * PAULI_I,
            np.sqrt(p / 3.0) * PAULI_X,
            np.sqrt(p / 3.0) * PAULI_Y,
            np.sqrt(p / 3.0) * PAULI_Z,
        ]
    if kind == "amplitude_damping":
        gamma = _validated_probability(raw, kind=kind)
        return [
            np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
            np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
        ]
    if kind == "coherent_x":
        return [_rotation_unitary(PAULI_X, raw)]
    if kind == "coherent_z":
        return [_rotation_unitary(PAULI_Z, raw)]
    raise KeyError(f"Unknown channel kind: {kind!r}")


def _apply_kind(
    rho: np.ndarray,
    qubit: int,
    p: float,
    kind: str,
    n_clock: int,
    n_system: int,
) -> np.ndarray:
    site = _resolve_site(qubit, n_clock, n_system)
    return _apply_kraus_on_site(
        rho,
        kraus_ops_for_step({"kind": kind, "p": p}),
        site,
        n_clock + n_system,
    )


def apply_bitflip_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "bitflip", n_clock, n_system)


def apply_phaseflip_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "phaseflip", n_clock, n_system)


def apply_dephasing_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "dephasing", n_clock, n_system)


def apply_depolarizing_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "depolarizing", n_clock, n_system)


def apply_amplitude_damping_channel(
    rho: np.ndarray,
    qubit: int,
    p: float,
    n_clock: int,
    n_system: int,
) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "amplitude_damping", n_clock, n_system)


def apply_coherent_x_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "coherent_x", n_clock, n_system)


def apply_coherent_z_channel(rho: np.ndarray, qubit: int, p: float, n_clock: int, n_system: int) -> np.ndarray:
    return _apply_kind(rho, qubit, p, "coherent_z", n_clock, n_system)


CHANNELS: dict[str, Callable[..., np.ndarray]] = {
    "bitflip": apply_bitflip_channel,
    "phaseflip": apply_phaseflip_channel,
    "dephasing": apply_dephasing_channel,
    "depolarizing": apply_depolarizing_channel,
    "amplitude_damping": apply_amplitude_damping_channel,
    "coherent_x": apply_coherent_x_channel,
    "coherent_z": apply_coherent_z_channel,
}


def apply_noise_schedule(
    rho: np.ndarray,
    schedule: Iterable[dict],
    n_clock: int,
    n_system: int,
) -> np.ndarray:
    """Sequentially apply a list of channels.

    Each entry of `schedule` is a dict such as
    `{"kind": "bitflip", "qubit": 0, "p": 0.05}`. Stochastic channels read
    `p` as a probability. Coherent channels read `p` as a rotation angle.
    """
    out = rho
    for step in schedule:
        kind = str(step["kind"])
        if kind not in CHANNELS:
            raise KeyError(f"Unknown channel kind: {kind!r}")
        out = CHANNELS[kind](
            out,
            int(step["qubit"]),
            float(step["p"]),
            n_clock,
            n_system,
        )
    return out

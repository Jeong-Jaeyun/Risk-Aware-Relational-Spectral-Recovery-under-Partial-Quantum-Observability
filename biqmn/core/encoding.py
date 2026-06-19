"""Logical states and standard syndrome recovery for stabilizer QEC codes.

The public API keeps the original 3-qubit repetition-code helpers, but the
implementation now runs through a ``CodeSpec`` abstraction that also supports
the Phase 2 distance-3 codes:

* ``perfect5``: [[5,1,3]] perfect code
* ``steane7``: [[7,1,3]] Steane code
* ``shor9``: [[9,1,3]] Shor code

All physical qubit indices follow the repo convention: qubit 0 is the leftmost
factor in the NumPy Kronecker product.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .hamiltonian import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z


PAULI_BY_CHAR = {
    "I": PAULI_I,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}

ANTI_COMMUTING = {
    ("X", "Y"),
    ("Y", "X"),
    ("X", "Z"),
    ("Z", "X"),
    ("Y", "Z"),
    ("Z", "Y"),
}


@dataclass(frozen=True)
class CodeSpec:
    """Stabilizer-code contract used by encoded experiments."""

    name: str
    n: int
    k: int
    distance: int
    stabilizers: Tuple[str, ...]
    logical_x: str
    logical_z: str
    correctable_paulis: Tuple[str, ...] = ("X", "Y", "Z")
    aliases: Tuple[str, ...] = ()
    description: str = ""

    @property
    def syndrome_bits(self) -> int:
        return len(self.stabilizers)


CODE_SPECS: Dict[str, CodeSpec] = {
    "bitflip": CodeSpec(
        name="bitflip",
        n=3,
        k=1,
        distance=1,
        stabilizers=("ZZI", "IZZ"),
        logical_x="XXX",
        logical_z="ZII",
        correctable_paulis=("X",),
        aliases=("3q_bitflip", "repetition_bitflip"),
        description="3-qubit bit-flip repetition code",
    ),
    "phaseflip": CodeSpec(
        name="phaseflip",
        n=3,
        k=1,
        distance=1,
        stabilizers=("XXI", "IXX"),
        logical_x="ZZZ",
        logical_z="XII",
        correctable_paulis=("Z",),
        aliases=("3q_phaseflip", "repetition_phaseflip"),
        description="3-qubit phase-flip repetition code",
    ),
    "perfect5": CodeSpec(
        name="perfect5",
        n=5,
        k=1,
        distance=3,
        stabilizers=("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"),
        logical_x="XXXXX",
        logical_z="ZZZZZ",
        aliases=("5q", "5q_perfect", "perfect", "five_qubit"),
        description="[[5,1,3]] perfect stabilizer code",
    ),
    "steane7": CodeSpec(
        name="steane7",
        n=7,
        k=1,
        distance=3,
        stabilizers=(
            "IIIXXXX",
            "IXXIIXX",
            "XIXIXIX",
            "IIIZZZZ",
            "IZZIIZZ",
            "ZIZIZIZ",
        ),
        logical_x="XXXXXXX",
        logical_z="ZZZZZZZ",
        aliases=("7q", "7q_steane", "steane", "seven_qubit"),
        description="[[7,1,3]] Steane CSS code",
    ),
    "shor9": CodeSpec(
        name="shor9",
        n=9,
        k=1,
        distance=3,
        stabilizers=(
            "ZZIIIIIII",
            "IZZIIIIII",
            "IIIZZIIII",
            "IIIIZZIII",
            "IIIIIIZZI",
            "IIIIIIIZZ",
            "XXXXXXIII",
            "IIIXXXXXX",
        ),
        logical_x="ZIIZIIZII",
        logical_z="XXXIIIIII",
        aliases=("9q", "9q_shor", "shor", "nine_qubit"),
        description="[[9,1,3]] Shor code",
    ),
}

SUPPORTED_CODES: Tuple[str, ...] = tuple(CODE_SPECS.keys())
_ALIAS_TO_NAME = {
    alias: spec.name
    for spec in CODE_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}


def get_code_spec(code: str) -> CodeSpec:
    key = str(code).strip().lower()
    if key not in _ALIAS_TO_NAME:
        raise ValueError(
            f"Unsupported QEC code {code!r}; expected one of {SUPPORTED_CODES}."
        )
    return CODE_SPECS[_ALIAS_TO_NAME[key]]


def _validated_code(code: str) -> str:
    return get_code_spec(code).name


def _normalise_pauli_string(pauli: str, *, n: Optional[int] = None) -> str:
    text = "".join(str(pauli).upper().split())
    if any(ch not in PAULI_BY_CHAR for ch in text):
        raise ValueError(f"Invalid Pauli string {pauli!r}.")
    if n is not None and len(text) != int(n):
        raise ValueError(f"Expected a length-{n} Pauli string, got {len(text)}.")
    return text


def _kron_ops(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def pauli_string_operator(pauli: str) -> np.ndarray:
    """Return the dense matrix for a Pauli string in NumPy qubit ordering."""
    text = _normalise_pauli_string(pauli)
    return _kron_ops([PAULI_BY_CHAR[ch] for ch in text])


def single_qubit_pauli_operator(n: int, qubit: int, pauli: str) -> np.ndarray:
    """Embed a one-qubit Pauli on ``qubit`` in an ``n``-qubit register."""
    if not 0 <= int(qubit) < int(n):
        raise ValueError(f"Qubit index {qubit} out of range [0,{n}).")
    ch = _normalise_pauli_string(pauli, n=1)
    text = ["I"] * int(n)
    text[int(qubit)] = ch
    return pauli_string_operator("".join(text))


def _pauli_action(pauli: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``perm, phase`` such that U|j> = phase[j] |perm[j]>."""
    text = _normalise_pauli_string(pauli)
    n = len(text)
    dim = 2 ** n
    perm = np.zeros(dim, dtype=int)
    phase = np.ones(dim, dtype=complex)
    for basis in range(dim):
        target = basis
        coeff = 1.0 + 0.0j
        for q, ch in enumerate(text):
            bit_pos = n - 1 - q
            bit = (basis >> bit_pos) & 1
            if ch == "I":
                continue
            if ch == "X":
                target ^= 1 << bit_pos
            elif ch == "Y":
                coeff *= 1j if bit == 0 else -1j
                target ^= 1 << bit_pos
            elif ch == "Z":
                coeff *= 1.0 if bit == 0 else -1.0
        perm[basis] = target
        phase[basis] = coeff
    return perm, phase


def _operator_from_action(action: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    perm, phase = action
    dim = int(perm.size)
    op = np.zeros((dim, dim), dtype=complex)
    op[perm, np.arange(dim)] = phase
    return op


@lru_cache(maxsize=None)
def _pauli_action_cached(pauli: str) -> tuple[np.ndarray, np.ndarray]:
    return _pauli_action(pauli)


def _apply_pauli_left(
    rho: np.ndarray,
    action: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    perm, phase = action
    out = np.empty_like(rho, dtype=complex)
    out[perm, :] = phase[:, None] * rho
    return out


def _apply_pauli_right(
    rho: np.ndarray,
    action: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    perm, phase = action
    out = np.empty_like(rho, dtype=complex)
    out[:, perm] = rho * np.conjugate(phase)[None, :]
    return out


def _apply_pauli_both(
    rho: np.ndarray,
    action: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    perm, phase = action
    out = np.empty_like(rho, dtype=complex)
    out[np.ix_(perm, perm)] = phase[:, None] * rho * np.conjugate(phase)[None, :]
    return out


def _apply_pauli_to_kets(
    kets: np.ndarray,
    action: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    perm, phase = action
    out = np.empty_like(kets, dtype=complex)
    out[perm, :] = phase[:, None] * kets
    return out


def _commutes(pauli_a: str, pauli_b: str) -> bool:
    a = _normalise_pauli_string(pauli_a)
    b = _normalise_pauli_string(pauli_b, n=len(a))
    anti_count = sum((left, right) in ANTI_COMMUTING for left, right in zip(a, b))
    return bool(anti_count % 2 == 0)


def _multiply_pauli_strings_ignoring_phase(pauli_a: str, pauli_b: str) -> str:
    a = _normalise_pauli_string(pauli_a)
    b = _normalise_pauli_string(pauli_b, n=len(a))
    table = {
        ("I", "I"): "I",
        ("I", "X"): "X",
        ("I", "Y"): "Y",
        ("I", "Z"): "Z",
        ("X", "I"): "X",
        ("X", "X"): "I",
        ("X", "Y"): "Z",
        ("X", "Z"): "Y",
        ("Y", "I"): "Y",
        ("Y", "X"): "Z",
        ("Y", "Y"): "I",
        ("Y", "Z"): "X",
        ("Z", "I"): "Z",
        ("Z", "X"): "Y",
        ("Z", "Y"): "X",
        ("Z", "Z"): "I",
    }
    return "".join(table[(left, right)] for left, right in zip(a, b))


def _single_qubit_error_string(n: int, qubit: int, pauli: str) -> str:
    ch = _normalise_pauli_string(pauli, n=1)
    if not 0 <= int(qubit) < int(n):
        raise ValueError(f"Qubit index {qubit} out of range [0,{n}).")
    chars = ["I"] * int(n)
    chars[int(qubit)] = ch
    return "".join(chars)


def syndrome_for_pauli_error(code: str, pauli_error: str) -> tuple[int, ...]:
    """Return the stabilizer syndrome bits for a Pauli error."""
    spec = get_code_spec(code)
    error = _normalise_pauli_string(pauli_error, n=spec.n)
    return tuple(0 if _commutes(error, stab) else 1 for stab in spec.stabilizers)


@lru_cache(maxsize=None)
def _syndrome_correction_map_cached(code_name: str) -> tuple[tuple[tuple[int, ...], str], ...]:
    spec = get_code_spec(code_name)
    no_error = tuple(0 for _ in spec.stabilizers)
    corrections: dict[tuple[int, ...], str] = {no_error: "I" * spec.n}
    for qubit in range(spec.n):
        for pauli in spec.correctable_paulis:
            error = _single_qubit_error_string(spec.n, qubit, pauli)
            syndrome = syndrome_for_pauli_error(spec.name, error)
            if syndrome == no_error:
                continue
            corrections.setdefault(syndrome, error)
    return tuple(sorted(corrections.items()))


def syndrome_correction_map(code: str) -> dict[tuple[int, ...], str]:
    """Return the standard single-error decoder table for ``code``."""
    spec = get_code_spec(code)
    return dict(_syndrome_correction_map_cached(spec.name))


def warm_up_syndrome_recovery(code: str) -> None:
    """Populate code-level caches used by syndrome recovery."""
    spec = get_code_spec(code)
    _code_projector_cached(spec.name)
    _syndrome_correction_map_cached(spec.name)
    _recovery_basis_cache(spec.name)


@lru_cache(maxsize=None)
def _stabilizer_actions(code_name: str) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    spec = get_code_spec(code_name)
    return tuple(_pauli_action_cached(stab) for stab in spec.stabilizers)


@lru_cache(maxsize=None)
def _stabilizer_group_actions(code_name: str) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    spec = get_code_spec(code_name)
    group_strings: list[str] = []
    for mask in product((0, 1), repeat=spec.syndrome_bits):
        pauli = "I" * spec.n
        for include, stabilizer in zip(mask, spec.stabilizers):
            if include:
                pauli = _multiply_pauli_strings_ignoring_phase(pauli, stabilizer)
        group_strings.append(pauli)
    return tuple(_pauli_action_cached(pauli) for pauli in group_strings)


def _measurement_dephase(rho: np.ndarray, code_name: str) -> np.ndarray:
    actions = _stabilizer_group_actions(code_name)
    out = np.zeros_like(rho, dtype=complex)
    for action in actions:
        out += _apply_pauli_both(rho, action)
    out = out / float(len(actions))
    return 0.5 * (out + out.conj().T)


@lru_cache(maxsize=None)
def _syndrome_basis_cache(code_name: str) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[int, ...], np.ndarray, np.ndarray], ...],
]:
    """Return a simultaneous stabilizer eigenbasis and corrected branch bases.

    The weighted stabilizer sum has one eigenvalue per syndrome, with a two
    dimensional eigenspace for each [[n,1,d]] syndrome sector. This makes ideal
    syndrome measurement a cheap block extraction in the cached basis.
    """
    spec = get_code_spec(code_name)
    dim = 2 ** spec.n
    stabilizers = stabilizer_operators(spec.name)
    weighted = np.zeros((dim, dim), dtype=complex)
    for idx, stabilizer in enumerate(stabilizers):
        weighted += float(2 ** idx) * stabilizer
    _, basis = np.linalg.eigh(0.5 * (weighted + weighted.conj().T))

    grouped: dict[tuple[int, ...], list[int]] = {}
    for col in range(dim):
        vec = basis[:, col]
        syndrome = []
        for stabilizer in stabilizers:
            eig = float(np.real(np.vdot(vec, stabilizer @ vec)))
            syndrome.append(0 if eig >= 0.0 else 1)
        grouped.setdefault(tuple(syndrome), []).append(col)

    corrections = syndrome_correction_map(spec.name)
    identity_correction = "I" * spec.n
    branches = []
    for syndrome in product((0, 1), repeat=spec.syndrome_bits):
        syndrome_key = tuple(int(bit) for bit in syndrome)
        indices = np.asarray(grouped.get(syndrome_key, []), dtype=int)
        if indices.size == 0:
            corrected_basis = np.zeros((dim, 0), dtype=complex)
        else:
            raw_basis = basis[:, indices]
            correction = corrections.get(syndrome_key, identity_correction)
            corrected_basis = _apply_pauli_to_kets(
                raw_basis,
                _pauli_action_cached(correction),
            )
        branches.append((syndrome_key, indices, corrected_basis))
    return basis, tuple(branches)


@lru_cache(maxsize=None)
def _recovery_basis_cache(code_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matrices for fast syndrome recovery in the cached eigenbasis."""
    basis, branches = _syndrome_basis_cache(code_name)
    corrected_basis = np.zeros_like(basis, dtype=complex)
    block_mask = np.zeros((basis.shape[1], basis.shape[1]), dtype=bool)
    for _, indices, branch_corrected_basis in branches:
        if indices.size == 0:
            continue
        corrected_basis[:, indices] = branch_corrected_basis
        block_mask[np.ix_(indices, indices)] = True
    return basis, corrected_basis, block_mask


def _project_syndrome_component(
    rho: np.ndarray,
    stabilizer_actions: Sequence[tuple[np.ndarray, np.ndarray]],
    syndrome: Sequence[int],
) -> np.ndarray:
    out = np.asarray(rho, dtype=complex)
    for bit, action in zip(syndrome, stabilizer_actions):
        sign = 1.0 if int(bit) == 0 else -1.0
        left = _apply_pauli_left(out, action)
        right = _apply_pauli_right(out, action)
        both = _apply_pauli_both(out, action)
        out = 0.25 * (out + sign * left + sign * right + both)
    return 0.5 * (out + out.conj().T)


@lru_cache(maxsize=None)
def _code_projector_cached(code_name: str) -> np.ndarray:
    spec = get_code_spec(code_name)
    dim = 2 ** spec.n
    projector = np.eye(dim, dtype=complex)
    for stab in spec.stabilizers:
        op = pauli_string_operator(stab)
        projector = projector @ (0.5 * (np.eye(dim, dtype=complex) + op))
    return 0.5 * (projector + projector.conj().T)


def code_space_projector(code: str) -> np.ndarray:
    return np.array(_code_projector_cached(get_code_spec(code).name), copy=True)


def _deterministic_phase(state: np.ndarray) -> np.ndarray:
    vec = np.asarray(state, dtype=complex).reshape(-1)
    idx = int(np.argmax(np.abs(vec)))
    if abs(vec[idx]) > 0.0:
        vec = vec * np.exp(-1j * np.angle(vec[idx]))
    return vec


@lru_cache(maxsize=None)
def _logical_basis_cached(code_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_code_spec(code_name)
    dim = 2 ** spec.n
    identity = np.eye(dim, dtype=complex)
    code_projector = _code_projector_cached(spec.name)
    logical_z = pauli_string_operator(spec.logical_z)
    logical_x = pauli_string_operator(spec.logical_x)
    zero_projector = code_projector @ (0.5 * (identity + logical_z))
    zero_projector = 0.5 * (zero_projector + zero_projector.conj().T)
    vals, vecs = np.linalg.eigh(zero_projector)
    psi0 = vecs[:, int(np.argmax(vals.real))]
    psi0 = _deterministic_phase(psi0 / (np.linalg.norm(psi0) + 1e-30))
    psi1 = logical_x @ psi0
    psi1 = _deterministic_phase(psi1 / (np.linalg.norm(psi1) + 1e-30))
    return psi0, psi1


def logical_basis(code: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(|0_L>, |1_L>)`` as dense kets in NumPy kron ordering."""
    psi0, psi1 = _logical_basis_cached(get_code_spec(code).name)
    return np.array(psi0, copy=True), np.array(psi1, copy=True)


def logical_state(code: str, amplitudes: Sequence[complex]) -> np.ndarray:
    """Build ``c_0 |0_L> + c_1 |1_L>`` as a normalised dense ket."""
    amps = list(amplitudes)
    if len(amps) != 2:
        raise ValueError("One-logical-qubit codes take exactly two amplitudes.")
    psi0, psi1 = logical_basis(code)
    state = complex(amps[0]) * psi0 + complex(amps[1]) * psi1
    norm = float(np.linalg.norm(state))
    if norm < 1.0e-14:
        raise ValueError("Logical amplitudes collapse to the zero vector.")
    return state / norm


def stabilizer_operators(code: str) -> list[np.ndarray]:
    """Return stabilizer generators as dense operators."""
    spec = get_code_spec(code)
    return [pauli_string_operator(stab) for stab in spec.stabilizers]


def logical_operators(code: str) -> dict:
    """Return logical X/Z and stabilizers as dense operators."""
    spec = get_code_spec(code)
    return {
        "X_L": pauli_string_operator(spec.logical_x),
        "Z_L": pauli_string_operator(spec.logical_z),
        "stabilizers": stabilizer_operators(spec.name),
    }


def syndrome_probabilities(rho_data: np.ndarray, code: str) -> dict:
    """Return syndrome outcome probabilities and the no-error probability."""
    spec = get_code_spec(code)
    rho = np.asarray(rho_data, dtype=complex)
    dim = 2 ** spec.n
    if rho.shape != (dim, dim):
        raise ValueError(f"Expected a {dim}x{dim} density matrix, got shape {rho.shape}.")
    basis, branches = _syndrome_basis_cache(spec.name)
    rho_eigen = basis.conj().T @ rho @ basis
    probs: dict[tuple[int, ...], float] = {}
    for syndrome, indices, _ in branches:
        if indices.size == 0:
            probs[syndrome] = 0.0
        else:
            block = rho_eigen[np.ix_(indices, indices)]
            probs[syndrome] = max(0.0, float(np.trace(block).real))
    total = sum(probs.values())
    if total > 0.0:
        probs = {key: float(value / total) for key, value in probs.items()}
    no_error = tuple(0 for _ in spec.stabilizers)
    return {
        "probabilities": probs,
        "no_error_probability": float(probs.get(no_error, 0.0)),
    }


def apply_syndrome_recovery(rho_data: np.ndarray, code: str) -> np.ndarray:
    """Apply ideal stabilizer syndrome measurement and standard correction.

    The decoder table is code-specific. For distance-3 codes it corrects all
    single-qubit Pauli errors; unrecognised higher-weight syndromes are left
    uncorrected.
    """
    spec = get_code_spec(code)
    rho = np.asarray(rho_data, dtype=complex)
    dim = 2 ** spec.n
    if rho.shape != (dim, dim):
        raise ValueError(f"Expected a {dim}x{dim} density matrix, got shape {rho.shape}.")

    basis, corrected_basis, block_mask = _recovery_basis_cache(spec.name)
    rho_eigen = basis.conj().T @ rho @ basis
    rho_eigen = np.where(block_mask, rho_eigen, 0.0)
    out = corrected_basis @ rho_eigen @ corrected_basis.conj().T
    out = 0.5 * (out + out.conj().T)
    trace = complex(np.trace(out))
    if abs(trace) > 0.0:
        out = out / trace
    return out


def code_space_probability(rho_data: np.ndarray, code: str) -> float:
    """Return Tr(P_code rho), clamped to [0, 1]."""
    projector = _code_projector_cached(get_code_spec(code).name)
    rho = np.asarray(rho_data, dtype=complex)
    value = float(np.trace(projector @ rho).real)
    return float(min(1.0, max(0.0, value)))


def collapse_probability(rho_data: np.ndarray, code: str) -> float:
    """Return probability mass outside the code space."""
    return float(1.0 - code_space_probability(rho_data, code))


def logical_state_fidelity(
    rho_data: np.ndarray,
    code: str,
    amplitudes: Sequence[complex],
) -> float:
    """Return fidelity against a target pure logical state."""
    psi = logical_state(code, amplitudes)
    value = float(np.real(np.vdot(psi, np.asarray(rho_data, dtype=complex) @ psi)))
    return float(min(1.0, max(0.0, value)))

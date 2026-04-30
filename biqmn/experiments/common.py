from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.clock import default_clock_initial_state
from ..core.global_state import (
    build_encoded_entangled_state,
    build_global_null_state,
    build_manual_entangled_state,
)
from ..core.hamiltonian import (
    align_clock_system_spectra,
    build_clock_hamiltonian,
    build_system_hamiltonian,
    build_total_hamiltonian,
    check_nullspace,
)
from ..core.relative_state import build_relative_family, generate_tau_grid
from ..core.simulator import simulate_global_density
from ..core.trajectory import build_spectral_trajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"
RESULT_ROOT = REPO_ROOT / "results"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = CONFIG_ROOT / path
    if path.suffix.lower() != ".yaml":
        path = path.with_suffix(".yaml")
    return path


def _load_yaml(path_like: str | Path | None) -> dict[str, Any]:
    path = _resolve_config_path(path_like)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def load_config(
    *,
    experiment_config: str | Path | None = None,
    state_config: str | Path | None = None,
    noise_config: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_yaml("default.yaml")
    for path_like in (state_config, noise_config, experiment_config):
        fragment = _load_yaml(path_like)
        config = _deep_merge(config, fragment)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def build_parser(
    *,
    description: str,
    default_experiment: str,
    default_state: str,
    default_noise: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_experiment,
        help="Path under configs/ to an experiment YAML fragment.",
    )
    parser.add_argument(
        "--state-config",
        default=default_state,
        help="Path under configs/ to a state YAML fragment.",
    )
    if default_noise is not None:
        parser.add_argument(
            "--noise-config",
            default=default_noise,
            help="Path under configs/ to a noise YAML fragment.",
        )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Override the JSON output stem written under results/raw.",
    )
    return parser


def resolve_output_stem(config: dict[str, Any], fallback: str, override: str | None) -> str:
    if override:
        return override
    base = str(config.get("output", {}).get("stem", "")).strip()
    if not base:
        return fallback
    if base == fallback:
        return base
    return f"{base}_{fallback}"


def _parse_complex_coeff(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return complex(float(value), 0.0)
    if isinstance(value, dict):
        return complex(
            float(value.get("real", 0.0)),
            float(value.get("imag", 0.0)),
        )
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise TypeError(f"Unsupported complex coefficient payload: {value!r}")


def _parse_null_coeffs(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    values = [_parse_complex_coeff(item) for item in raw]
    return np.asarray(values, dtype=complex)


def build_pipeline(config: dict[str, Any], *, with_noise: bool = False) -> dict[str, Any]:
    clock_cfg = config.get("clock", {})
    system_cfg = config.get("system", {})
    state_cfg = config.get("state", {})
    noise_cfg = config.get("noise", {})
    time_cfg = config.get("time", {})
    mapping_cfg = config.get("mapping", {})
    sim_cfg = config.get("simulation", {})

    n_clock = int(clock_cfg.get("n_qubits", 1))
    n_system = int(system_cfg.get("n_qubits", 2))

    Hc = build_clock_hamiltonian(n_clock, clock_cfg.get("hamiltonian", {}))
    Hs = build_system_hamiltonian(n_system, system_cfg.get("hamiltonian", {}))
    aligned_shift = None
    if bool(state_cfg.get("align_spectra", False)):
        shift = align_clock_system_spectra(Hc, Hs)
        if shift is not None:
            Hs = Hs + shift * np.eye(Hs.shape[0], dtype=complex)
            aligned_shift = float(shift)

    Htot = build_total_hamiltonian(Hc, Hs)
    mode = str(state_cfg.get("mode", "manual"))
    if mode == "manual":
        psi = build_manual_entangled_state(
            n_clock,
            n_system,
            recipe=str(state_cfg.get("recipe", "uniform_bell")),
        )
    elif mode == "null":
        psi = build_global_null_state(
            Htot,
            mode=str(state_cfg.get("null_mode", "ground_null")),
            tol=float(state_cfg.get("tol", 1e-9)),
            coeffs=_parse_null_coeffs(state_cfg.get("null_coeffs")),
        )
    elif mode == "encoded":
        psi = build_encoded_entangled_state(
            n_clock,
            n_system,
            code=str(state_cfg.get("code", "bitflip")),
            amplitudes=_parse_null_coeffs(state_cfg.get("logical_amplitudes")),
        )
    else:
        raise ValueError(f"Unsupported state.mode={mode!r}")

    applied_noise = []
    if with_noise and noise_cfg.get("schedule"):
        applied_noise = deepcopy(noise_cfg["schedule"])
    rho = simulate_global_density(
        psi,
        applied_noise,
        n_clock=n_clock,
        n_system=n_system,
        backend=str(sim_cfg.get("backend", "linear_algebra")),
    )

    psi0_clock = default_clock_initial_state(
        n_clock,
        kind=str(clock_cfg.get("initial_state", "plus")),
    )
    tau_grid = generate_tau_grid(
        Hc,
        float(time_cfg.get("tau_min", -2.0)),
        float(time_cfg.get("tau_max", 2.0)),
        int(time_cfg.get("n_tau", 9)),
        scale_by_norm=bool(time_cfg.get("scale_by_norm", False)),
    )
    family = build_relative_family(
        rho,
        Hc,
        tau_grid,
        psi0_clock,
        2 ** n_clock,
        2 ** n_system,
    )
    trajectory = build_spectral_trajectory(
        family,
        n_system=n_system,
        tau_grid=tau_grid,
        mapping_cfg=mapping_cfg,
    )
    nullspace = check_nullspace(Htot, tol=float(state_cfg.get("tol", 1e-9)))
    constraint_residual = float(np.linalg.norm(Htot @ psi))
    return {
        "config": config,
        "n_clock": n_clock,
        "n_system": n_system,
        "Hc": Hc,
        "Hs": Hs,
        "Htot": Htot,
        "psi0_clock": psi0_clock,
        "global_state": psi,
        "global_rho": rho,
        "tau_grid": tau_grid,
        "relative_family": family,
        "trajectory": trajectory,
        "meta": {
            "with_noise": with_noise,
            "applied_noise": applied_noise,
            "aligned_shift": aligned_shift,
            "mapping_mode": mapping_cfg.get("mode", "coherence_abs"),
            "state_mode": mode,
            "simulation_backend": str(sim_cfg.get("backend", "linear_algebra")),
            "constraint_residual": constraint_residual,
            "nullspace_dim": int(nullspace["dim"]),
        },
    }


def build_reference_bank(config: dict[str, Any]) -> list[dict[str, Any]]:
    ref_cfg = config.get("reference_bank", {})
    entries = list(ref_cfg.get("trajectories", []))
    if not entries:
        entries = [{"label": "base"}]

    clean_base = _deep_merge(config, {"noise": {"schedule": []}})
    bundles: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        label = str(entry.get("label", f"ref_{index}"))
        override = {key: value for key, value in entry.items() if key != "label"}
        bundle_cfg = _deep_merge(clean_base, override)
        bundle = build_pipeline(bundle_cfg, with_noise=False)
        bundle["label"] = label
        bundles.append(bundle)
    return bundles


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def write_json_result(payload: dict[str, Any], stem: str, *, subdir: str = "raw") -> Path:
    target_dir = RESULT_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}.json"
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path

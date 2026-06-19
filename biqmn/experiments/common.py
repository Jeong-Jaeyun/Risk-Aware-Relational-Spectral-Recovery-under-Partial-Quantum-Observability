from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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


def progress_iter(
    iterable: Iterable,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
):
    """Wrap an iterable in tqdm when available.

    Set ``BIQMN_PROGRESS=0`` to disable progress bars for non-interactive runs.
    """
    if str(os.environ.get("BIQMN_PROGRESS", "1")).strip().lower() in {"0", "false", "no"}:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, file=sys.stdout)


def progress_bar(
    *,
    total: int,
    desc: str | None = None,
    unit: str = "it",
    initial: int = 0,
):
    """Return a tqdm progress bar when available, else a small no-op object."""
    if str(os.environ.get("BIQMN_PROGRESS", "1")).strip().lower() in {"0", "false", "no"}:
        return _NoOpProgress(total=total, initial=initial)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NoOpProgress(total=total, initial=initial)
    return tqdm(total=total, desc=desc, unit=unit, initial=initial, file=sys.stdout)


class _NoOpProgress:
    def __init__(self, *, total: int, initial: int = 0):
        self.total = int(total)
        self.n = int(initial)

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


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


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(row), ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL row in {target} at line {line_no}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _case_key(case: dict[str, Any], key_field: str) -> Any:
    if key_field not in case:
        raise KeyError(f"Case is missing key field {key_field!r}: {case!r}")
    return case[key_field]


def run_case_tasks(
    cases: Sequence[dict[str, Any]],
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_workers: int = 1,
    desc: str | None = None,
    unit: str = "case",
    key_field: str = "sample_index",
    jsonl_path: str | Path | None = None,
    resume: bool = True,
    existing_rows: Sequence[dict[str, Any]] | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Run independent case tasks, optionally in processes, with main-process JSONL writes."""
    cases = list(cases)
    max_workers = max(1, int(max_workers))

    rows_by_key: dict[Any, dict[str, Any]] = {}
    if jsonl_path is not None:
        path = Path(jsonl_path)
        if resume:
            cached = list(existing_rows) if existing_rows is not None else load_jsonl_rows(path)
            valid_keys = {_case_key(case, key_field) for case in cases}
            for row in cached:
                if key_field not in row:
                    continue
                key = row[key_field]
                if key in valid_keys:
                    rows_by_key[key] = dict(row)
        elif path.exists():
            path.unlink()

    pending_cases = [
        case for case in cases if _case_key(case, key_field) not in rows_by_key
    ]
    if not pending_cases:
        return [rows_by_key[_case_key(case, key_field)] for case in cases]

    if max_workers == 1:
        if initializer is not None:
            initializer(*initargs)
        with progress_bar(total=len(cases), desc=desc, unit=unit, initial=len(rows_by_key)) as progress:
            for case in pending_cases:
                row = worker_fn(case)
                key = _case_key(case, key_field)
                rows_by_key[key] = row
                if jsonl_path is not None:
                    append_jsonl_row(jsonl_path, row)
                progress.update(1)
    else:
        with progress_bar(total=len(cases), desc=desc, unit=unit, initial=len(rows_by_key)) as progress:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=initializer,
                initargs=initargs,
            ) as executor:
                futures = {
                    executor.submit(worker_fn, case): _case_key(case, key_field)
                    for case in pending_cases
                }
                for future in as_completed(futures):
                    key = futures[future]
                    row = future.result()
                    rows_by_key[key] = row
                    if jsonl_path is not None:
                        append_jsonl_row(jsonl_path, row)
                    progress.update(1)

    return [rows_by_key[_case_key(case, key_field)] for case in cases]

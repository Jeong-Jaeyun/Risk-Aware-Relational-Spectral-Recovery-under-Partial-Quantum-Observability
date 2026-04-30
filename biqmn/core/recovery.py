"""Reference-guided recovery on a discrete trajectory bank.

The paper-level geodesic recovery problem is approximated here by a practical
two-stage proxy:

1. compute the nearest reference anchor to the observed trajectory
2. optimize a restoration-aware objective over a discrete candidate bank
3. optionally scan an admissible local convex hull for diagnostics/refinement

The key modeling change is that restoration is driven by distance to a fixed
reference anchor, while Laplacian / smoothness / clock penalties remain
feasibility or regularization terms.
"""
from __future__ import annotations

from itertools import product
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .admissibility import admissibility_report, phi_clock, phi_lap, phi_ref, phi_smooth
from .laplacian import graph_laplacian, ordered_spectrum
from .trajectory import SpectralTrajectory, trajectory_distance, trajectory_distance_squared


DEFAULT_WEIGHTS = {
    "obs_fit": 1.0,
    "ref_anchor": 1.0,
    "lap": 1.0,
    "smooth": 0.5,
    "clock": 1.0,
    "phi_ref": 0.0,
}


WEIGHT_ALIASES = {
    "data": "obs_fit",
    "anchor": "ref_anchor",
    "ref": "phi_ref",
}


def _validated_weights(weights: Optional[dict]) -> dict:
    incoming = dict(weights or {})
    normalized: dict[str, float] = {}
    for key, value in incoming.items():
        mapped = WEIGHT_ALIASES.get(str(key), str(key))
        normalized[mapped] = value
    merged = {**DEFAULT_WEIGHTS, **normalized}
    for key, value in merged.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"Recovery weight '{key}' must be a finite non-negative value."
            )
        merged[key] = numeric
    return merged


def nearest_reference_anchor(
    traj_obs: SpectralTrajectory,
    ref_bank: Sequence[SpectralTrajectory],
) -> dict:
    """Return the nearest trajectory in the reference family to the observation."""
    if not ref_bank:
        raise ValueError("ref_bank must contain at least one trajectory.")
    scored = [
        {
            "index": int(idx),
            "distance_squared": float(trajectory_distance_squared(traj_obs, traj_ref)),
            "trajectory": traj_ref,
        }
        for idx, traj_ref in enumerate(ref_bank)
    ]
    best = min(scored, key=lambda item: item["distance_squared"])
    return {
        "index": int(best["index"]),
        "distance_squared": float(best["distance_squared"]),
        "distance": float(np.sqrt(best["distance_squared"])),
        "trajectory": best["trajectory"],
        "scored": scored,
    }


def recovery_objective(traj_candidate: SpectralTrajectory,
                       traj_obs: SpectralTrajectory,
                       traj_ref_anchor: SpectralTrajectory,
                       Hc: np.ndarray,
                       psi0_clock: np.ndarray,
                       weights: Optional[dict] = None,
                       ref_bank: Optional[Sequence[SpectralTrajectory]] = None,
                       eta: Optional[float] = None) -> Tuple[float, dict]:
    """Reference-guided recovery objective with explicit anchor restoration."""
    w = _validated_weights(weights)
    terms = {
        "obs_fit": trajectory_distance_squared(traj_candidate, traj_obs),
        "ref_anchor": trajectory_distance_squared(traj_candidate, traj_ref_anchor),
        "lap": phi_lap(traj_candidate),
        "smooth": phi_smooth(traj_candidate),
        "clock": phi_clock(traj_candidate, Hc, psi0_clock, eta),
    }
    if ref_bank is not None:
        terms["phi_ref"] = phi_ref(traj_candidate, ref_bank)
    obj = float(sum(w.get(k, 0.0) * v for k, v in terms.items()))
    return obj, terms


def recover_via_reference_projection(traj_obs: SpectralTrajectory,
                                     candidate_bank: Sequence[SpectralTrajectory],
                                     Hc: np.ndarray,
                                     psi0_clock: np.ndarray,
                                     weights: Optional[dict] = None,
                                     ref_bank: Optional[Sequence[SpectralTrajectory]] = None,
                                     eta: Optional[float] = None) -> dict:
    """Project onto a discrete candidate bank under the new restoration objective."""
    if not candidate_bank:
        raise ValueError("candidate_bank must contain at least one trajectory.")
    ref_bank = list(ref_bank) if ref_bank is not None else list(candidate_bank)
    ref_anchor = nearest_reference_anchor(traj_obs, ref_bank)
    traj_ref_anchor = ref_anchor["trajectory"]

    scored: List[dict] = []
    best_idx = -1
    best_obj = float("inf")
    best_terms: dict = {}
    for idx, cand in enumerate(candidate_bank):
        obj, terms = recovery_objective(
            cand,
            traj_obs,
            traj_ref_anchor,
            Hc,
            psi0_clock,
            weights,
            ref_bank=ref_bank,
            eta=eta,
        )
        scored.append({"index": idx, "objective": obj, "terms": terms})
        if obj < best_obj:
            best_obj = obj
            best_idx = idx
            best_terms = terms

    anchor_obj, anchor_terms = recovery_objective(
        traj_ref_anchor,
        traj_obs,
        traj_ref_anchor,
        Hc,
        psi0_clock,
        weights,
        ref_bank=ref_bank,
        eta=eta,
    )
    observed_obj, observed_terms = recovery_objective(
        traj_obs,
        traj_obs,
        traj_ref_anchor,
        Hc,
        psi0_clock,
        weights,
        ref_bank=ref_bank,
        eta=eta,
    )
    return {
        "best_index": best_idx,
        "best_objective": best_obj,
        "best_terms": best_terms,
        "reference_anchor_index": int(ref_anchor["index"]),
        "reference_anchor_distance_squared": float(ref_anchor["distance_squared"]),
        "reference_anchor_objective": float(anchor_obj),
        "reference_anchor_terms": anchor_terms,
        "objective_gain_vs_reference_anchor": float(anchor_obj - best_obj),
        "observed_objective": observed_obj,
        "observed_terms": observed_terms,
        "reference_anchor": traj_ref_anchor,
        "recovered": candidate_bank[best_idx],
        "scored": scored,
    }


def _interpolate_trajectory(traj_obs: SpectralTrajectory,
                            traj_target: SpectralTrajectory,
                            alpha: float) -> SpectralTrajectory:
    if len(traj_obs) != len(traj_target):
        raise ValueError("Interpolated recovery requires equal slice counts.")
    if not np.allclose(traj_obs.tau_grid, traj_target.tau_grid):
        raise ValueError("Interpolated recovery requires matching tau grids.")
    if traj_obs.n_nodes != traj_target.n_nodes:
        raise ValueError("Interpolated recovery requires matching graph sizes.")

    mixed_densities = []
    mixed_adjacency = []
    mixed_laplacians = []
    mixed_spectra = []
    for rho_obs, rho_target, W_obs, W_target in zip(
        traj_obs.densities,
        traj_target.densities,
        traj_obs.adjacency,
        traj_target.adjacency,
    ):
        rho = ((1.0 - alpha) * np.asarray(rho_obs) + alpha * np.asarray(rho_target))
        rho = 0.5 * (rho + rho.conj().T)
        W = ((1.0 - alpha) * np.asarray(W_obs, dtype=float)
             + alpha * np.asarray(W_target, dtype=float))
        np.fill_diagonal(W, 0.0)
        W = np.maximum(0.0, 0.5 * (W + W.T))
        L = graph_laplacian(W)
        mixed_densities.append(rho)
        mixed_adjacency.append(W)
        mixed_laplacians.append(L)
        mixed_spectra.append(ordered_spectrum(L))

    meta = dict(traj_obs.meta)
    meta.update({
        "interpolation_alpha": float(alpha),
        "interpolation_target": traj_target.meta.get("label", "recovered"),
        "interpolation_mode": "density_adjacency_convex",
    })
    return SpectralTrajectory(
        tau_grid=np.array(traj_obs.tau_grid, copy=True),
        densities=mixed_densities,
        adjacency=mixed_adjacency,
        laplacians=mixed_laplacians,
        spectra=mixed_spectra,
        meta=meta,
    )


def convex_combination_recovery(traj_obs: SpectralTrajectory,
                                candidate_bank: Sequence[SpectralTrajectory],
                                Hc: np.ndarray,
                                psi0_clock: np.ndarray,
                                weights: Optional[dict] = None,
                                eta: Optional[float] = None,
                                alphas: Optional[np.ndarray] = None,
                                ref_bank: Optional[Sequence[SpectralTrajectory]] = None,
                                thresholds: Optional[dict] = None) -> dict:
    """Convex interpolation diagnostic around the discrete recovered trajectory.

    For each τ_r we replace Λ_obs(τ_r) by α Λ_best(τ_r) + (1−α) Λ_obs(τ_r)
    and scan α ∈ alphas to find the objective minimum.  Useful for visualising
    the interpolated recovery path even without re-running Page-Wootters.
    """
    alphas = np.linspace(0.0, 1.0, 21) if alphas is None else np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("Recovery alpha grid must be a non-empty 1D array.")
    if np.any(~np.isfinite(alphas)):
        raise ValueError("Recovery alphas must be finite.")
    if np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError("Recovery alphas must lie in [0, 1].")
    scan_ref_bank = list(ref_bank) if ref_bank is not None else list(candidate_bank)
    base = recover_via_reference_projection(
        traj_obs,
        candidate_bank,
        Hc,
        psi0_clock,
        weights=weights,
        ref_bank=scan_ref_bank,
        eta=eta,
    )
    best = base["recovered"]
    traj_ref_anchor = base["reference_anchor"]
    scan = []
    best_obj = float("inf")
    best_alpha = 0.0
    best_traj = traj_obs.copy()
    best_admissible_obj = float("inf")
    best_admissible_alpha: Optional[float] = None
    best_admissible_traj: Optional[SpectralTrajectory] = None
    for a in alphas:
        mixed = _interpolate_trajectory(traj_obs, best, float(a))
        obj, terms = recovery_objective(
            mixed,
            traj_obs,
            traj_ref_anchor,
            Hc,
            psi0_clock,
            weights,
            ref_bank=scan_ref_bank,
            eta=eta,
        )
        entry = {"alpha": float(a), "objective": obj, "terms": terms}
        if thresholds is not None:
            report = admissibility_report(
                mixed,
                scan_ref_bank,
                Hc,
                psi0_clock,
                thresholds,
                eta=eta,
            )
            entry["admissibility"] = report
            if report["admissible"] and obj < best_admissible_obj:
                best_admissible_obj = obj
                best_admissible_alpha = float(a)
                best_admissible_traj = mixed.copy()
        scan.append(entry)
        if obj < best_obj:
            best_obj = obj
            best_alpha = float(a)
            best_traj = mixed.copy()
    result = {
        "base_projection": base,
        "scan": scan,
        "best_alpha": best_alpha,
        "best_objective": best_obj,
        "best_spectra": [s.copy() for s in best_traj.spectra],
    }
    if thresholds is not None:
        result.update({
            "admissibility_thresholds": {k: float(v) for k, v in thresholds.items()},
            "admissible_alpha_count": int(sum(
                1 for item in scan if item.get("admissibility", {}).get("admissible", False)
            )),
            "best_admissible_alpha": best_admissible_alpha,
            "best_admissible_objective": (
                None if best_admissible_alpha is None else float(best_admissible_obj)
            ),
            "best_admissible_spectra": (
                None if best_admissible_traj is None
                else [s.copy() for s in best_admissible_traj.spectra]
            ),
        })
    return result


def _simplex_weight_grid(n_candidates: int, grid_steps: int) -> list[np.ndarray]:
    if n_candidates <= 0:
        raise ValueError("Simplex search requires at least one candidate.")
    if grid_steps <= 0:
        raise ValueError("Simplex grid steps must be positive.")
    if n_candidates == 1:
        return [np.array([1.0], dtype=float)]

    weights: list[np.ndarray] = []
    for counts in product(range(grid_steps + 1), repeat=n_candidates - 1):
        partial = int(sum(counts))
        if partial > grid_steps:
            continue
        last = grid_steps - partial
        vec = np.asarray((*counts, last), dtype=float) / float(grid_steps)
        weights.append(vec)
    return weights


def _convex_hull_trajectory(candidates: Sequence[SpectralTrajectory],
                            weights: np.ndarray) -> SpectralTrajectory:
    if len(candidates) != int(weights.size):
        raise ValueError("Candidate/weight count mismatch in convex hull trajectory.")
    base = candidates[0]
    mixed_densities = []
    mixed_adjacency = []
    mixed_laplacians = []
    mixed_spectra = []
    for slice_idx in range(len(base)):
        rho = np.zeros_like(base.densities[slice_idx], dtype=complex)
        W = np.zeros_like(base.adjacency[slice_idx], dtype=float)
        for weight, traj in zip(weights, candidates):
            rho = rho + float(weight) * np.asarray(traj.densities[slice_idx])
            W = W + float(weight) * np.asarray(traj.adjacency[slice_idx], dtype=float)
        rho = 0.5 * (rho + rho.conj().T)
        np.fill_diagonal(W, 0.0)
        W = np.maximum(0.0, 0.5 * (W + W.T))
        L = graph_laplacian(W)
        mixed_densities.append(rho)
        mixed_adjacency.append(W)
        mixed_laplacians.append(L)
        mixed_spectra.append(ordered_spectrum(L))

    meta = dict(base.meta)
    meta.update({
        "interpolation_mode": "local_admissible_hull",
        "convex_weights": weights.tolist(),
    })
    return SpectralTrajectory(
        tau_grid=np.array(base.tau_grid, copy=True),
        densities=mixed_densities,
        adjacency=mixed_adjacency,
        laplacians=mixed_laplacians,
        spectra=mixed_spectra,
        meta=meta,
    )


def stage2_admissible_convex_refinement(
    traj_obs: SpectralTrajectory,
    candidate_bank: Sequence[SpectralTrajectory],
    Hc: np.ndarray,
    psi0_clock: np.ndarray,
    *,
    stage1_index: int,
    weights: Optional[dict] = None,
    eta: Optional[float] = None,
    ref_bank: Optional[Sequence[SpectralTrajectory]] = None,
    thresholds: Optional[dict] = None,
    local_count: int = 3,
    grid_steps: int = 10,
    require_interior: bool = True,
) -> dict:
    """Section 10.9 Stage 2: search an admissible local convex hull."""
    if not candidate_bank:
        raise ValueError("Stage-2 refinement requires a non-empty candidate bank.")
    if local_count <= 0:
        raise ValueError("Stage-2 local_count must be positive.")
    if stage1_index < 0 or stage1_index >= len(candidate_bank):
        raise ValueError("stage1_index out of range for candidate_bank.")

    ref_bank = list(ref_bank) if ref_bank is not None else list(candidate_bank)
    traj_ref_anchor = nearest_reference_anchor(traj_obs, ref_bank)["trajectory"]
    anchor = candidate_bank[stage1_index]
    neighbor_order = sorted(
        range(len(candidate_bank)),
        key=lambda idx: (
            0 if idx == stage1_index else 1,
            trajectory_distance(anchor, candidate_bank[idx]),
            idx,
        ),
    )
    local_indices = neighbor_order[: min(local_count, len(candidate_bank))]
    local_candidates = [candidate_bank[idx] for idx in local_indices]
    weight_grid = _simplex_weight_grid(len(local_candidates), grid_steps)

    scan = []
    best_obj = float("inf")
    best_weights: Optional[np.ndarray] = None
    best_traj: Optional[SpectralTrajectory] = None
    best_admissible_obj = float("inf")
    best_admissible_weights: Optional[np.ndarray] = None
    best_admissible_traj: Optional[SpectralTrajectory] = None
    best_admissible_terms: Optional[dict] = None
    best_admissible_report: Optional[dict] = None
    best_admissible_is_interior: Optional[bool] = None

    for simplex_weights in weight_grid:
        mixed = _convex_hull_trajectory(local_candidates, simplex_weights)
        obj, terms = recovery_objective(
            mixed,
            traj_obs,
            traj_ref_anchor,
            Hc,
            psi0_clock,
            weights,
            ref_bank=ref_bank,
            eta=eta,
        )
        is_interior = bool(np.count_nonzero(simplex_weights > 1e-9) > 1)
        entry = {
            "weights": simplex_weights.tolist(),
            "objective": obj,
            "terms": terms,
            "is_interior": is_interior,
        }
        if obj < best_obj:
            best_obj = obj
            best_weights = simplex_weights.copy()
            best_traj = mixed.copy()

        if thresholds is not None:
            report = admissibility_report(
                mixed,
                ref_bank,
                Hc,
                psi0_clock,
                thresholds,
                eta=eta,
            )
            entry["admissibility"] = report
            admissible = bool(report["admissible"])
            if admissible and (not require_interior or is_interior) and obj < best_admissible_obj:
                best_admissible_obj = obj
                best_admissible_weights = simplex_weights.copy()
                best_admissible_traj = mixed.copy()
                best_admissible_terms = dict(terms)
                best_admissible_report = report
                best_admissible_is_interior = is_interior
        scan.append(entry)

    feasible_count = sum(
        1
        for item in scan
        if item.get("admissibility", {}).get("admissible", False)
    )
    interior_feasible_count = sum(
        1
        for item in scan
        if item.get("admissibility", {}).get("admissible", False) and item["is_interior"]
    )
    return {
        "local_indices": [int(idx) for idx in local_indices],
        "scan": scan,
        "best_objective": float(best_obj),
        "best_weights": None if best_weights is None else best_weights.tolist(),
        "best_recovered": best_traj,
        "best_admissible_objective": (
            None if best_admissible_weights is None else float(best_admissible_obj)
        ),
        "best_admissible_weights": (
            None if best_admissible_weights is None else best_admissible_weights.tolist()
        ),
        "best_admissible_terms": best_admissible_terms,
        "best_admissible_report": best_admissible_report,
        "best_admissible_is_interior": best_admissible_is_interior,
        "best_admissible_recovered": best_admissible_traj,
        "feasible_count": int(feasible_count),
        "interior_feasible_count": int(interior_feasible_count),
        "require_interior": bool(require_interior),
        "grid_steps": int(grid_steps),
    }


def two_stage_admissible_recovery(
    traj_obs: SpectralTrajectory,
    candidate_bank: Sequence[SpectralTrajectory],
    Hc: np.ndarray,
    psi0_clock: np.ndarray,
    *,
    stage2_enabled: bool = True,
    weights: Optional[dict] = None,
    eta: Optional[float] = None,
    ref_bank: Optional[Sequence[SpectralTrajectory]] = None,
    thresholds: Optional[dict] = None,
    stage2_local_count: int = 3,
    stage2_grid_steps: int = 10,
    stage2_require_interior: bool = True,
    stage2_tol: float = 1e-9,
    stage2_apply_rule: str = "diagnostic_only",
) -> dict:
    """Section 10.9 two-stage admissible recovery."""
    apply_rule = str(stage2_apply_rule).strip().lower()
    if apply_rule not in {"diagnostic_only", "objective_only"}:
        raise ValueError(
            "stage2_apply_rule must be one of {'diagnostic_only', 'objective_only'}"
        )
    stage1 = recover_via_reference_projection(
        traj_obs,
        candidate_bank,
        Hc,
        psi0_clock,
        weights=weights,
        ref_bank=ref_bank,
        eta=eta,
    )
    if not stage2_enabled:
        return {
            "stage1": stage1,
            "stage2": {
                "local_indices": [int(stage1["best_index"])],
                "scan": [],
                "best_objective": float(stage1["best_objective"]),
                "best_weights": [1.0],
                "best_recovered": stage1["recovered"],
                "best_admissible_objective": None,
                "best_admissible_weights": None,
                "best_admissible_terms": None,
                "best_admissible_report": None,
                "best_admissible_is_interior": None,
                "best_admissible_recovered": None,
                "feasible_count": 0,
                "interior_feasible_count": 0,
                "require_interior": bool(stage2_require_interior),
                "grid_steps": int(stage2_grid_steps),
            },
            "stage2_applied": False,
            "final_recovered": stage1["recovered"],
            "final_objective": float(stage1["best_objective"]),
            "final_stage": "stage1",
            "stage2_apply_rule": apply_rule,
            "stage2_apply_reason": "stage2_disabled",
            "stage2_candidate_objective": None,
            "stage2_candidate_objective_gain": 0.0,
            "stage2_objective_gain": 0.0,
        }
    stage2 = stage2_admissible_convex_refinement(
        traj_obs,
        candidate_bank,
        Hc,
        psi0_clock,
        stage1_index=int(stage1["best_index"]),
        weights=weights,
        eta=eta,
        ref_bank=ref_bank,
        thresholds=thresholds,
        local_count=stage2_local_count,
        grid_steps=stage2_grid_steps,
        require_interior=stage2_require_interior,
    )
    stage2_candidate_objective = (
        None
        if stage2["best_admissible_objective"] is None
        else float(stage2["best_admissible_objective"])
    )
    stage2_candidate_gain = (
        0.0
        if stage2_candidate_objective is None
        else float(stage1["best_objective"]) - stage2_candidate_objective
    )
    stage2_applied = (
        apply_rule == "objective_only"
        and stage2["best_admissible_recovered"] is not None
        and stage2_candidate_objective is not None
        and stage2_candidate_objective < float(stage1["best_objective"]) - stage2_tol
    )
    final_recovered = (
        stage2["best_admissible_recovered"] if stage2_applied else stage1["recovered"]
    )
    final_objective = (
        stage2_candidate_objective
        if stage2_applied
        else float(stage1["best_objective"])
    )
    if stage2_candidate_objective is None:
        apply_reason = "no_admissible_stage2_candidate"
    elif apply_rule == "diagnostic_only":
        apply_reason = "diagnostic_only"
    elif stage2_applied:
        apply_reason = "objective_improves"
    else:
        apply_reason = "candidate_not_better_than_stage1"
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage2_applied": bool(stage2_applied),
        "final_recovered": final_recovered,
        "final_objective": final_objective,
        "final_stage": "stage2" if stage2_applied else "stage1",
        "stage2_apply_rule": apply_rule,
        "stage2_apply_reason": apply_reason,
        "stage2_candidate_objective": stage2_candidate_objective,
        "stage2_candidate_objective_gain": float(stage2_candidate_gain),
        "stage2_objective_gain": float(stage1["best_objective"] - final_objective),
    }

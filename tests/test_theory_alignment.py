from __future__ import annotations

import unittest
import warnings

import numpy as np

from biqmn.core.admissibility import phi_ref
from biqmn.core.admissibility import admissibility_report, calibrate_thresholds
from biqmn.core.clock import (
    clock_bures_distance,
    clock_geom_distance,
    clock_state,
    default_clock_initial_state,
)
from biqmn.core.clock_consistency import (
    calibrate_clock_eta,
    clock_consistency_ratio,
    clock_penalty,
)
from biqmn.core.hamiltonian import check_nullspace, clock_energy_variance
from biqmn.core.metrics import bures_distance, fidelity
from biqmn.core.relative_state import (
    adjacent_fidelity,
    relative_state_density,
    relative_state_pure,
)
from biqmn.core.recovery import (
    convex_combination_recovery,
    recover_via_reference_projection,
    recovery_objective,
    stage2_admissible_convex_refinement,
    two_stage_admissible_recovery,
)
from biqmn.core.trajectory import (
    SpectralTrajectory,
    spectral_response,
    trajectory_distance,
    trajectory_distance_squared,
    trajectory_smoothness_penalty,
)
from biqmn.experiments.common import build_pipeline, build_reference_bank, load_config


def _toy_traj(scale: float = 1.0) -> SpectralTrajectory:
    tau = np.array([0.0, 0.5, 1.0], dtype=float)
    spectra = [
        np.array([0.0, 0.0], dtype=float),
        np.array([0.0, 1.0 * scale], dtype=float),
        np.array([0.0, 3.0 * scale], dtype=float),
    ]
    laplacians = [np.diag(spec) for spec in spectra]
    densities = [np.eye(2, dtype=complex) / 2.0 for _ in spectra]
    adjacency = [np.zeros((2, 2), dtype=float) for _ in spectra]
    return SpectralTrajectory(
        tau_grid=tau,
        densities=densities,
        adjacency=adjacency,
        laplacians=laplacians,
        spectra=spectra,
    )


class TheoryAlignmentTests(unittest.TestCase):
    def test_clock_geometry_matches_paper_definition(self) -> None:
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        tau = 0.4
        overlap = np.vdot(psi0, np.array([np.cos(tau), -1j * np.sin(tau)], dtype=complex))
        expected = 1.0 - abs(overlap) ** 2
        observed = clock_geom_distance(Hc, 0.0, tau, psi0)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_clock_bures_distance_matches_pure_state_formula(self) -> None:
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        tau = 0.4
        overlap = abs(np.vdot(psi0, np.array([np.cos(tau), -1j * np.sin(tau)], dtype=complex)))
        expected = np.sqrt(2.0 * (1.0 - overlap))
        observed = clock_bures_distance(Hc, 0.0, tau, psi0)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_clock_geometry_has_quadratic_small_dt_scaling(self) -> None:
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        variance = clock_energy_variance(clock_state(Hc, 0.0, psi0), Hc)
        dt = 1e-3
        observed = clock_geom_distance(Hc, 0.0, dt, psi0) / (dt ** 2)
        self.assertAlmostEqual(observed, variance, places=4)

    def test_clock_state_rejects_dimension_mismatch(self) -> None:
        Hc = np.eye(2, dtype=complex)
        psi0 = np.ones(3, dtype=complex)
        with self.assertRaises(ValueError):
            clock_state(Hc, 0.0, psi0)

    def test_default_clock_initial_state_requires_positive_qubits(self) -> None:
        with self.assertRaises(ValueError):
            default_clock_initial_state(0)

    def test_trajectory_distance_matches_discrete_metric(self) -> None:
        traj_a = _toy_traj(scale=1.0)
        traj_b = _toy_traj(scale=2.0)
        expected_sq = (0.0 ** 2 + 1.0 ** 2 + 3.0 ** 2) / 3.0
        self.assertAlmostEqual(trajectory_distance_squared(traj_a, traj_b), expected_sq)
        self.assertAlmostEqual(trajectory_distance(traj_a, traj_b), np.sqrt(expected_sq))

    def test_spectral_response_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            spectral_response(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))

    def test_smoothness_penalty_matches_sum_of_squared_slice_slopes(self) -> None:
        traj = _toy_traj(scale=1.0)
        expected = (1.0 ** 2) / (0.5 ** 2) + (2.0 ** 2) / (0.5 ** 2)
        self.assertAlmostEqual(trajectory_smoothness_penalty(traj), expected)

    def test_trajectory_requires_strictly_increasing_tau_grid(self) -> None:
        tau = np.array([0.0, 0.5, 0.5], dtype=float)
        spectra = [np.array([0.0, 0.0], dtype=float) for _ in tau]
        laplacians = [np.diag(spec) for spec in spectra]
        densities = [np.eye(2, dtype=complex) / 2.0 for _ in tau]
        adjacency = [np.zeros((2, 2), dtype=float) for _ in tau]
        with self.assertRaises(ValueError):
            SpectralTrajectory(
                tau_grid=tau,
                densities=densities,
                adjacency=adjacency,
                laplacians=laplacians,
                spectra=spectra,
            )

    def test_phi_ref_uses_squared_distance(self) -> None:
        traj = _toy_traj(scale=1.0)
        ref = _toy_traj(scale=2.0)
        self.assertAlmostEqual(phi_ref(traj, [ref]), trajectory_distance(traj, ref) ** 2)

    def test_admissibility_threshold_calibration_returns_background_samples(self) -> None:
        traj_a = _toy_traj(scale=1.0)
        traj_b = _toy_traj(scale=1.5)
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        report = calibrate_thresholds([traj_a, traj_b], Hc, psi0, kappa=2.0)
        for key in ("phi_lap", "phi_smooth", "phi_ref", "phi_clock"):
            self.assertIn("phis", report[key])
            self.assertEqual(len(report[key]["phis"]), 2)

    def test_admissibility_report_requires_all_thresholds(self) -> None:
        traj = _toy_traj(scale=1.0)
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        with self.assertRaises(ValueError):
            admissibility_report(
                traj,
                [traj],
                Hc,
                psi0,
                thresholds={"phi_lap": 0.0},
            )

    def test_clock_penalty_is_sum_of_slice_ratios(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
        )
        bundle = build_pipeline(cfg, with_noise=False)
        penalty = clock_penalty(bundle["trajectory"], bundle["Hc"], bundle["psi0_clock"])
        expected = sum(
            clock_consistency_ratio(resp, geom)
            for resp, geom in zip(penalty["resp"], penalty["geom"])
        )
        self.assertAlmostEqual(penalty["phi"], expected)

    def test_recovery_objective_rejects_negative_weight(self) -> None:
        traj = _toy_traj(scale=1.0)
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        with self.assertRaises(ValueError):
            recovery_objective(
                traj,
                traj,
                traj,
                Hc,
                psi0,
                weights={"clock": -1.0},
            )

    def test_convex_recovery_requires_alphas_in_unit_interval(self) -> None:
        traj = _toy_traj(scale=1.0)
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        with self.assertRaises(ValueError):
            convex_combination_recovery(
                traj,
                [traj],
                Hc,
                psi0,
                alphas=np.array([-0.1, 0.5, 1.1]),
            )

    def test_reference_projection_reports_gain_vs_reference_anchor(self) -> None:
        traj_obs = _toy_traj(scale=1.2)
        traj_a = _toy_traj(scale=1.0)
        traj_b = _toy_traj(scale=2.0)
        Hc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        psi0 = np.array([1.0, 0.0], dtype=complex)
        report = recover_via_reference_projection(
            traj_obs,
            [traj_a, traj_b],
            Hc,
            psi0,
            weights={"obs_fit": 1.0, "ref_anchor": 1.0, "lap": 0.0, "smooth": 0.0, "clock": 0.0},
            ref_bank=[traj_a, traj_b],
        )
        self.assertIn("reference_anchor_index", report)
        self.assertIn("objective_gain_vs_reference_anchor", report)
        self.assertGreaterEqual(report["objective_gain_vs_reference_anchor"], 0.0)
        self.assertAlmostEqual(
            report["objective_gain_vs_reference_anchor"],
            report["reference_anchor_objective"] - report["best_objective"],
        )

    def test_stage2_refinement_can_report_admissible_hull_points(self) -> None:
        cfg = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        reference_bundles = build_reference_bank(cfg)
        reference_trajs = [bundle["trajectory"] for bundle in reference_bundles]
        observed = build_pipeline(cfg, with_noise=True)
        clean = build_pipeline(cfg, with_noise=False)
        threshold_report = calibrate_thresholds(
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            kappa=float(cfg.get("admissibility", {}).get("kappa", 3.0)),
        )
        thresholds = {
            key: value["threshold"]
            for key, value in threshold_report.items()
            if key.startswith("phi_")
        }
        stage1 = recover_via_reference_projection(
            observed["trajectory"],
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            ref_bank=reference_trajs,
        )
        report = stage2_admissible_convex_refinement(
            observed["trajectory"],
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            stage1_index=stage1["best_index"],
            ref_bank=reference_trajs,
            thresholds=thresholds,
        )
        self.assertGreaterEqual(report["feasible_count"], 1)
        self.assertIsNotNone(report["best_admissible_weights"])

    def test_two_stage_recovery_prefers_stage2_when_it_improves(self) -> None:
        cfg = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        reference_bundles = build_reference_bank(cfg)
        reference_trajs = [bundle["trajectory"] for bundle in reference_bundles]
        observed = build_pipeline(cfg, with_noise=True)
        clean = build_pipeline(cfg, with_noise=False)
        threshold_report = calibrate_thresholds(
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            kappa=float(cfg.get("admissibility", {}).get("kappa", 3.0)),
        )
        thresholds = {
            key: value["threshold"]
            for key, value in threshold_report.items()
            if key.startswith("phi_")
        }
        report = two_stage_admissible_recovery(
            observed["trajectory"],
            reference_trajs,
            clean["Hc"],
            clean["psi0_clock"],
            ref_bank=reference_trajs,
            thresholds=thresholds,
            stage2_apply_rule="objective_only",
        )
        self.assertIn(report["final_stage"], {"stage1", "stage2"})
        if report["stage2_applied"]:
            self.assertGreater(report["stage2_objective_gain"], 0.0)

    def test_clock_consistency_ratio_uses_additive_epsilon(self) -> None:
        resp = 2.0
        geom = 1.0e-12
        eps = 1.0e-3
        observed = clock_consistency_ratio(resp, geom, eps)
        expected = resp / (geom + eps)
        self.assertAlmostEqual(observed, expected)

    def test_calibrate_clock_eta_matches_background_distribution(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
        )
        ref_a = build_pipeline(cfg, with_noise=False)["trajectory"]
        cfg_shifted = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
            overrides={
                "state": {
                    "null_mode": "custom",
                    "null_coeffs": [
                        {"real": 1.0, "imag": 0.0},
                        {"real": 0.7071067811865476, "imag": 0.7071067811865476},
                    ],
                }
            },
        )
        bundle_b = build_pipeline(cfg_shifted, with_noise=False)
        ref_b = bundle_b["trajectory"]
        calibration = calibrate_clock_eta(
            [ref_a, ref_b],
            bundle_b["Hc"],
            bundle_b["psi0_clock"],
            kappa=2.0,
        )
        phis = np.asarray(calibration["phis"], dtype=float)
        self.assertEqual(phis.size, 2)
        self.assertAlmostEqual(calibration["phi_mean"], float(phis.mean()))
        self.assertAlmostEqual(calibration["phi_std"], float(phis.std()))
        self.assertAlmostEqual(
            calibration["eta"],
            float(phis.mean() + 2.0 * phis.std()),
        )

    def test_adjacent_fidelity_uses_uhlmann_fidelity(self) -> None:
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)
        sigma = np.array([[0.4, 0.0], [0.0, 0.6]], dtype=complex)
        observed = adjacent_fidelity([rho, sigma])[0]
        expected = fidelity(rho, sigma)
        self.assertAlmostEqual(observed, expected)

    def test_pure_and_density_relative_states_agree(self) -> None:
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
        clock = np.array([1.0, 0.0], dtype=complex)
        density_branch = relative_state_density(
            np.outer(psi, psi.conj()),
            clock,
            dim_clock=2,
            dim_system=2,
        )
        pure_branch = relative_state_pure(
            psi,
            clock,
            dim_clock=2,
            dim_system=2,
        )
        self.assertTrue(
            np.allclose(density_branch, np.outer(pure_branch, pure_branch.conj()))
        )

    def test_relative_state_density_rejects_zero_probability_slice(self) -> None:
        rho = np.diag([0.0, 0.0, 1.0, 0.0]).astype(complex)
        clock = np.array([1.0, 0.0], dtype=complex)
        with self.assertRaises(ValueError):
            relative_state_density(
                rho,
                clock,
                dim_clock=2,
                dim_system=2,
            )

    def test_rank_deficient_fidelity_is_stable(self) -> None:
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        sigma = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            observed_fidelity = fidelity(rho, sigma)
            observed_bures = bures_distance(rho, sigma)
        self.assertEqual(caught, [])
        self.assertAlmostEqual(observed_fidelity, 0.5, places=10)
        expected_bures = np.sqrt(2.0 * (1.0 - np.sqrt(0.5)))
        self.assertAlmostEqual(observed_bures, expected_bures, places=10)

    def test_null_state_pipeline_respects_constraint(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
        )
        bundle = build_pipeline(cfg, with_noise=False)
        null_info = check_nullspace(bundle["Htot"])
        self.assertGreaterEqual(null_info["dim"], 1)
        self.assertLess(bundle["meta"]["constraint_residual"], 1e-8)

    def test_encoded_bitflip_mapping_sees_bitflip_noise(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/repetition_bitflip.yaml",
            noise_config="noise/bitflip.yaml",
        )
        clean = build_pipeline(cfg, with_noise=False)
        noisy = build_pipeline(cfg, with_noise=True)
        self.assertGreater(
            trajectory_distance(clean["trajectory"], noisy["trajectory"]),
            0.0,
        )

    def test_encoded_phaseflip_mapping_sees_phaseflip_noise(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/repetition_phaseflip.yaml",
            noise_config="noise/phaseflip.yaml",
        )
        clean = build_pipeline(cfg, with_noise=False)
        noisy = build_pipeline(cfg, with_noise=True)
        self.assertGreater(
            trajectory_distance(clean["trajectory"], noisy["trajectory"]),
            0.0,
        )

    def test_qiskit_aer_backend_matches_linear_algebra_without_noise(self) -> None:
        base_kwargs = dict(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
        )
        aer = build_pipeline(
            load_config(
                **base_kwargs,
                overrides={"simulation": {"backend": "qiskit_aer"}},
            ),
            with_noise=False,
        )
        direct = build_pipeline(
            load_config(
                **base_kwargs,
                overrides={"simulation": {"backend": "linear_algebra"}},
            ),
            with_noise=False,
        )
        self.assertEqual(aer["meta"]["simulation_backend"], "qiskit_aer")
        self.assertLess(
            float(np.max(np.abs(aer["global_rho"] - direct["global_rho"]))),
            1e-9,
        )
        for spec_a, spec_b in zip(aer["trajectory"].spectra, direct["trajectory"].spectra):
            self.assertTrue(np.allclose(spec_a, spec_b, atol=1e-9))

    def test_qiskit_aer_backend_matches_linear_algebra_with_noise(self) -> None:
        base_kwargs = dict(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        aer = build_pipeline(
            load_config(
                **base_kwargs,
                overrides={"simulation": {"backend": "qiskit_aer"}},
            ),
            with_noise=True,
        )
        direct = build_pipeline(
            load_config(
                **base_kwargs,
                overrides={"simulation": {"backend": "linear_algebra"}},
            ),
            with_noise=True,
        )
        self.assertLess(
            float(np.max(np.abs(aer["global_rho"] - direct["global_rho"]))),
            1e-9,
        )
        for rho_a, rho_b in zip(aer["relative_family"], direct["relative_family"]):
            self.assertTrue(np.allclose(rho_a, rho_b, atol=1e-9))
        for spec_a, spec_b in zip(aer["trajectory"].spectra, direct["trajectory"].spectra):
            self.assertTrue(np.allclose(spec_a, spec_b, atol=1e-9))

    def test_qiskit_aer_backend_matches_linear_algebra_for_new_noise_families(self) -> None:
        families = [
            ("amplitude_damping", 0.08),
            ("coherent_x", 0.08),
            ("coherent_z", 0.11),
        ]
        for kind, strength in families:
            with self.subTest(kind=kind):
                base_kwargs = dict(
                    experiment_config="experiment/recovery_eval.yaml",
                    state_config="states/null_dynamic.yaml",
                )
                aer = build_pipeline(
                    load_config(
                        **base_kwargs,
                        overrides={
                            "noise": {"schedule": [{"kind": kind, "qubit": 0, "p": strength}]},
                            "simulation": {"backend": "qiskit_aer"},
                        },
                    ),
                    with_noise=True,
                )
                direct = build_pipeline(
                    load_config(
                        **base_kwargs,
                        overrides={
                            "noise": {"schedule": [{"kind": kind, "qubit": 0, "p": strength}]},
                            "simulation": {"backend": "linear_algebra"},
                        },
                    ),
                    with_noise=True,
                )
                self.assertLess(
                    float(np.max(np.abs(aer["global_rho"] - direct["global_rho"]))),
                    1e-9,
                )
                for rho_a, rho_b in zip(aer["relative_family"], direct["relative_family"]):
                    self.assertTrue(np.allclose(rho_a, rho_b, atol=1e-9))
                for spec_a, spec_b in zip(aer["trajectory"].spectra, direct["trajectory"].spectra):
                    self.assertTrue(np.allclose(spec_a, spec_b, atol=1e-9))

    def test_hadamard_basis_mapping_detects_phase_coherent_error(self) -> None:
        cfg = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/repetition_phaseflip_coherent.yaml",
            overrides={"noise": {"schedule": [{"kind": "coherent_z", "qubit": 0, "p": 0.1}]}},
        )
        clean = build_pipeline(cfg, with_noise=False)
        noisy = build_pipeline(cfg, with_noise=True)
        self.assertGreater(
            trajectory_distance(clean["trajectory"], noisy["trajectory"]),
            1e-3,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from biqmn.experiments.common import load_config
from biqmn.experiments.run_coherent_casebook import run as run_coherent_casebook
from biqmn.experiments.run_coherent_v1_baseline import run as run_coherent_v1_baseline
from biqmn.experiments.run_hybrid_c123_baseline import (
    C3RPolicyConfig,
    PolicyScoreConfig,
    _aggregate_rows,
    _choose_c3r_policy,
    _decision_disagreement_ab,
    enrich_c3r_row,
    run as run_hybrid_c123_baseline,
)
from biqmn.experiments.run_hybrid_c123_casebook import run as run_hybrid_c123_casebook
from biqmn.experiments.run_c2_preferred_analysis import run as run_c2_preferred_analysis
from biqmn.experiments.run_hybrid_c123_regime_map import (
    _preferred_policy,
    run as run_hybrid_c123_regime_map,
)
from biqmn.experiments.run_partial_syndrome_baseline import run as run_partial_syndrome_baseline
from biqmn.experiments.run_noisy_syndrome_baseline import run as run_noisy_syndrome_baseline
from biqmn.experiments.run_partial_noisy_syndrome_regime_map import run as run_partial_noisy_syndrome_regime_map
from biqmn.experiments.run_ambiguity_measurement_syndrome_regime_map import run as run_ambiguity_measurement_syndrome_regime_map
from biqmn.experiments.syndrome_observation import (
    SyndromeObservationConfig,
    observe_syndrome_statistics,
)
from biqmn.experiments.run_coherent_veto_analysis import run as run_coherent_veto_analysis
from biqmn.experiments.run_coherent_veto_threshold_sweep import run as run_coherent_veto_threshold_sweep
from biqmn.experiments.run_admissible_projection import run as run_admissible_projection
from biqmn.experiments.run_recovery_ablation import run as run_recovery_ablation
from biqmn.experiments.run_ref_anchor_validation import run as run_ref_anchor_validation
from biqmn.experiments.run_detection_metrics import run as run_detection_metrics
from biqmn.experiments.run_encoded_casebook import run as run_encoded_casebook
from biqmn.experiments.run_encoded_qec_baseline import run as run_encoded_qec_baseline
from biqmn.experiments.run_encoded_regime_map import run as run_encoded_regime_map
from biqmn.experiments.run_noise_trajectory import run as run_noise_trajectory
from biqmn.experiments.run_random_noise_baseline import run as run_random_noise_baseline
from biqmn.experiments.run_recovery_objective import run as run_recovery_objective
from biqmn.experiments.run_recovery_sweep import run as run_recovery_sweep
from biqmn.experiments.run_relative_slices import run as run_relative_slices


def _toy_coherent_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "bitflip-1",
            "code_type": "bitflip",
            "noise_family": "coherent_x",
            "noise_strength": 0.05,
            "noise_depth": 1,
            "seed": 11,
            "syndrome": "00",
            "reason_C": "keep_syndrome",
            "fidelity_before": 0.98,
            "fidelity_after_A": 1.00,
            "fidelity_after_B": 1.00,
            "fidelity_after_C": 1.00,
            "gain_A": 0.02,
            "gain_B": 0.02,
            "gain_C": 0.02,
            "admissible_A": True,
            "admissible_B": True,
            "recovered_C_admissible": True,
            "syndrome_consistent": True,
            "trajectory_inconsistent": True,
            "syndrome_consistent_but_trajectory_inconsistent": True,
            "hybrid_use_relational": False,
            "A_recovery_nonworsen": True,
            "B_recovery_nonworsen": True,
            "C_recovery_nonworsen": True,
            "clean_observed_distance": 0.30,
            "clean_observed_distance": 0.30,
            "objective_A": 1.0,
            "objective_B": 1.0,
            "objective_C": 1.0,
            "traj_dist_A": 0.0,
            "traj_dist_B": 0.0,
            "hybrid_objective_gain_B_vs_A": 0.0,
            "syndrome_mean_no_error": 0.95,
        },
        {
            "experiment_id": "bitflip-2",
            "code_type": "bitflip",
            "noise_family": "coherent_x",
            "noise_strength": 0.10,
            "noise_depth": 2,
            "seed": 12,
            "syndrome": "00",
            "reason_C": "keep_syndrome",
            "fidelity_before": 0.95,
            "fidelity_after_A": 1.00,
            "fidelity_after_B": 1.00,
            "fidelity_after_C": 1.00,
            "gain_A": 0.05,
            "gain_B": 0.05,
            "gain_C": 0.05,
            "admissible_A": True,
            "admissible_B": True,
            "recovered_C_admissible": True,
            "syndrome_consistent": True,
            "trajectory_inconsistent": True,
            "syndrome_consistent_but_trajectory_inconsistent": True,
            "hybrid_use_relational": False,
            "A_recovery_nonworsen": True,
            "B_recovery_nonworsen": True,
            "C_recovery_nonworsen": True,
            "clean_observed_distance": 0.60,
            "objective_A": 1.0,
            "objective_B": 1.0,
            "objective_C": 1.0,
            "traj_dist_A": 0.0,
            "traj_dist_B": 0.0,
            "hybrid_objective_gain_B_vs_A": 0.0,
            "syndrome_mean_no_error": 0.92,
        },
        {
            "experiment_id": "phaseflip-1",
            "code_type": "phaseflip",
            "noise_family": "coherent_z",
            "noise_strength": 0.05,
            "noise_depth": 2,
            "seed": 13,
            "syndrome": "00",
            "reason_C": "veto_nonadmissible_A",
            "fidelity_before": 0.96,
            "fidelity_after_A": 1.00,
            "fidelity_after_B": 0.97,
            "fidelity_after_C": 0.97,
            "gain_A": 0.04,
            "gain_B": 0.01,
            "gain_C": 0.01,
            "admissible_A": False,
            "admissible_B": True,
            "recovered_C_admissible": True,
            "syndrome_consistent": True,
            "trajectory_inconsistent": True,
            "syndrome_consistent_but_trajectory_inconsistent": True,
            "hybrid_use_relational": True,
            "A_recovery_nonworsen": True,
            "B_recovery_nonworsen": True,
            "C_recovery_nonworsen": True,
            "clean_observed_distance": 0.70,
            "objective_A": 1.4,
            "objective_B": 1.2,
            "objective_C": 1.2,
            "traj_dist_A": 0.0,
            "traj_dist_B": 0.03,
            "hybrid_objective_gain_B_vs_A": 0.2,
            "syndrome_mean_no_error": 0.94,
        },
        {
            "experiment_id": "phaseflip-2",
            "code_type": "phaseflip",
            "noise_family": "coherent_z",
            "noise_strength": 0.10,
            "noise_depth": 3,
            "seed": 14,
            "syndrome": "00",
            "reason_C": "tie_break_objective",
            "fidelity_before": 0.95,
            "fidelity_after_A": 1.00,
            "fidelity_after_B": 0.94,
            "fidelity_after_C": 0.94,
            "gain_A": 0.05,
            "gain_B": -0.01,
            "gain_C": -0.01,
            "admissible_A": True,
            "admissible_B": True,
            "recovered_C_admissible": True,
            "syndrome_consistent": True,
            "trajectory_inconsistent": True,
            "syndrome_consistent_but_trajectory_inconsistent": True,
            "hybrid_use_relational": True,
            "A_recovery_nonworsen": True,
            "B_recovery_nonworsen": True,
            "C_recovery_nonworsen": True,
            "clean_observed_distance": 1.10,
            "objective_A": 1.5,
            "objective_B": 1.2,
            "objective_C": 1.2,
            "traj_dist_A": 0.0,
            "traj_dist_B": 0.06,
            "hybrid_objective_gain_B_vs_A": 0.3,
            "syndrome_mean_no_error": 0.91,
        },
    ]


class ExperimentSmokeTests(unittest.TestCase):
    def test_relative_slices_are_valid(self) -> None:
        config = load_config(
            experiment_config="experiment/trajectory_probe.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_relative_slices(config)
        self.assertTrue(result["summary"]["all_valid"])
        self.assertGreater(result["summary"]["n_slices"], 4)
        self.assertGreater(result["summary"]["adjacent_fidelity_mean"], 0.0)
        self.assertGreaterEqual(result["summary"]["nullspace_dim"], 2)
        self.assertLess(result["summary"]["constraint_residual"], 1e-8)
        self.assertGreater(result["summary"]["trajectory_smoothness"], 1.0)

    def test_detection_scores_see_noise(self) -> None:
        config = load_config(
            experiment_config="experiment/detection_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        result = run_detection_metrics(config)
        self.assertGreater(result["summary"]["trajectory_distance"], 0.0)
        self.assertGreater(result["summary"]["score_mean"], 0.0)

    def test_recovery_default_operating_point_is_stable(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        result = run_recovery_objective(config)
        self.assertGreaterEqual(result["summary"]["candidate_count"], 1)
        self.assertLessEqual(
            result["summary"]["clean_to_recovered_distance"],
            result["summary"]["clean_to_observed_distance"],
        )
        self.assertTrue(result["summary"]["observed_admissible"])
        self.assertTrue(result["summary"]["recovered_admissible"])
        self.assertGreaterEqual(
            result["summary"]["objective_gain_vs_reference_anchor"],
            0.0,
        )
        self.assertEqual(result["summary"]["reference_anchor_label"], "phase_pi_over_2")
        self.assertEqual(result["summary"]["clean_to_recovered_distance"], 0.0)
        self.assertIn("reference_anchor_label", result["summary"])
        self.assertIn("stage1_projection", result)
        self.assertIn("stage2_refinement", result)
        self.assertIn("diagnostics", result)
        self.assertIn("stage1_objective", result["summary"])
        self.assertIn("stage2_applied", result["summary"])
        self.assertEqual(result["summary"]["stage2_apply_rule"], "diagnostic_only")
        self.assertEqual(result["summary"]["final_stage"], "stage1")
        self.assertGreaterEqual(
            result["stage2_refinement"]["feasible_count"],
            1,
        )
        self.assertIsNotNone(
            result["stage2_refinement"]["best_admissible_weights"]
        )

    def test_admissible_projection_reports_stable_default_thresholds(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
            noise_config="noise/dephasing.yaml",
        )
        result = run_admissible_projection(config)
        self.assertTrue(result["summary"]["clean_admissible"])
        self.assertIsInstance(result["summary"]["noisy_admissible"], bool)
        self.assertEqual(result["summary"]["reference_count"], 3)
        self.assertGreater(result["summary"]["phi_ref_threshold"], 0.0)
        self.assertGreater(result["summary"]["phi_clock_threshold"], 0.0)

    def test_dynamic_null_separates_noise_channels(self) -> None:
        bitflip = run_noise_trajectory(
            load_config(
                experiment_config="experiment/trajectory_probe.yaml",
                state_config="states/null_dynamic.yaml",
                noise_config="noise/bitflip.yaml",
            )
        )
        phaseflip = run_noise_trajectory(
            load_config(
                experiment_config="experiment/trajectory_probe.yaml",
                state_config="states/null_dynamic.yaml",
                noise_config="noise/phaseflip.yaml",
            )
        )
        dephasing = run_noise_trajectory(
            load_config(
                experiment_config="experiment/trajectory_probe.yaml",
                state_config="states/null_dynamic.yaml",
                noise_config="noise/dephasing.yaml",
            )
        )
        self.assertGreater(
            phaseflip["summary"]["trajectory_distance"],
            bitflip["summary"]["trajectory_distance"],
        )
        self.assertGreater(
            bitflip["summary"]["trajectory_distance"],
            dephasing["summary"]["trajectory_distance"],
        )
        self.assertGreater(dephasing["summary"]["trajectory_distance"], 0.0)

    def test_recovery_sweep_emits_rows_and_summary(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_recovery_sweep(
            config,
            noise_kinds=["dephasing"],
            strengths=[0.12],
            kappas=[1.5],
            bank_widths_deg=[10.0],
        )
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["summary"]["overall"]["cases"], 1)
        self.assertEqual(result["rows"][0]["noise_kind"], "dephasing")
        self.assertGreaterEqual(result["rows"][0]["stage2_feasible_count"], 1)
        self.assertIn("stage2_candidate_location", result["rows"][0])
        self.assertIn("location_rate", result["summary"]["overall"])

    def test_recovery_ablation_exports_stage_comparison_tables(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_recovery_ablation(
            config,
            noise_kinds=["dephasing"],
            strengths=[0.12],
            kappas=[1.5],
            bank_widths_deg=[10.0],
        )
        self.assertEqual(result["overall"]["cases"], 1)
        self.assertEqual(len(result["tables"]["by_noise_kind"]), 1)
        self.assertEqual(len(result["tables"]["by_stage2_candidate_location"]), 1)
        self.assertIn("stage2_helpful_tradeoff_rate", result["overall"])
        self.assertIn("corr_gain_vs_clean_delta", result["overall"])
        self.assertIn("stage2_candidate_improvement_rate", result["overall"])
        self.assertIn("## By Noise Kind", result["markdown"])
        self.assertIn("## By Stage-2 Location", result["markdown"])

    def test_recovery_ablation_can_compare_objective_variants(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_recovery_ablation(
            config,
            noise_kinds=["dephasing"],
            strengths=[0.12],
            kappas=[1.5],
            bank_widths_deg=[10.0],
            objective_variants=["old_base", "old_plus_phi_ref", "new_ref_anchor"],
        )
        self.assertEqual(len(result["tables"]["by_objective_variant"]), 3)
        self.assertIn("## By Objective Variant", result["markdown"])

    def test_ref_anchor_validation_emits_weight_width_table(self) -> None:
        config = load_config(
            experiment_config="experiment/recovery_eval.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_ref_anchor_validation(
            config,
            ref_anchor_weights=[32.0],
            bank_widths_deg=[10.0],
            noise_kinds=["dephasing"],
            strengths=[0.12],
            kappas=[1.5],
        )
        self.assertEqual(result["overall"]["cases"], 1)
        self.assertEqual(len(result["tables"]["by_ref_anchor_weight_and_bank_width"]), 1)
        self.assertIn("final_recovery_stability", result["overall"])
        self.assertIn("## By Ref-Anchor Weight And Bank Width", result["markdown"])

    def test_random_noise_baseline_emits_rows_and_uses_qiskit_aer(self) -> None:
        config = load_config(
            experiment_config="experiment/random_noise_baseline.yaml",
            state_config="states/null_dynamic.yaml",
        )
        result = run_random_noise_baseline(
            config,
            n_samples=2,
            seed=3,
            min_steps=1,
            max_steps=2,
            kinds=["bitflip", "dephasing"],
            p_min=0.03,
            p_max=0.05,
        )
        self.assertEqual(result["overall"]["cases"], 2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["overall"]["simulation_backend"], "qiskit_aer")
        self.assertIn("by_schedule_signature", result["tables"])
        self.assertIn("## By Schedule Signature", result["markdown"])

    def test_encoded_qec_baseline_emits_code_tables(self) -> None:
        result = run_encoded_qec_baseline(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing"],
                "phaseflip": ["phaseflip", "dephasing"],
            },
            n_samples=1,
            seed=5,
            min_steps=1,
            max_steps=1,
            kinds=["bitflip", "phaseflip"],
            p_min=0.03,
            p_max=0.05,
            experiment_config="experiment/encoded_qec_baseline.yaml",
        )
        self.assertEqual(result["overall"]["cases"], 2)
        self.assertEqual(result["overall"]["simulation_backend"], "qiskit_aer")
        self.assertEqual(result["overall"]["code_count"], 2)
        self.assertEqual(len(result["tables"]["by_code"]), 2)
        self.assertIn("by_noise_family", result["tables"])
        self.assertIn("by_noise_depth", result["tables"])
        self.assertIn("hybrid_uses_relational_rate", result["overall"])
        self.assertIn("fid_recovered_C_mean", result["overall"])
        self.assertIn("experiment_id", result["rows"][0])
        self.assertIn("noise_family", result["rows"][0])
        self.assertIn("candidate_A", result["rows"][0])
        self.assertIn("chosen_C", result["rows"][0])
        self.assertIn("## By Code", result["markdown"])
        self.assertIn("## By Noise Family", result["markdown"])

    def test_encoded_casebook_extracts_representative_groups(self) -> None:
        result = run_encoded_casebook(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing"],
                "phaseflip": ["phaseflip", "dephasing"],
            },
            n_samples=2,
            seed=7,
            min_steps=1,
            max_steps=2,
            kinds=["bitflip", "phaseflip"],
            p_min=0.03,
            p_max=0.06,
            experiment_config="experiment/encoded_qec_baseline.yaml",
        )
        self.assertIn("group_summary", result["tables"])
        self.assertIn("representative_cases", result)
        self.assertGreaterEqual(len(result["groups"]), 4)
        self.assertIn("syndrome_consistent_trajectory_inconsistent", result["groups"])
        self.assertIn("hybrid_veto_triggered", result["groups"])
        self.assertIn("## Group Summary", result["markdown"])

    def test_encoded_regime_map_emits_boundary_and_mode_tables(self) -> None:
        result = run_encoded_regime_map(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing"],
                "phaseflip": ["phaseflip", "dephasing"],
            },
            noise_families=["bitflip", "mixed"],
            strengths=[0.05],
            depths=[1],
            seeds=[3],
            fidelity_margin=0.01,
            trajectory_inconsistency_threshold=0.05,
            syndrome_consistent_threshold=0.9,
            hybrid_objective_tol=1.0e-9,
            tie_break_requires_syndrome_consistent=True,
            experiment_config="experiment/encoded_regime_map.yaml",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertIn("by_noise_family", result["tables"])
        self.assertIn("failure_boundary_summary", result["tables"])
        self.assertIn("by_recovery_mode", result["tables"])
        self.assertIn("top_failure_boundary_cases", result["tables"])
        self.assertIn("## Failure Boundary Summary", result["markdown"])

    def test_encoded_regime_map_supports_amplitude_and_coherent_families(self) -> None:
        result = run_encoded_regime_map(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing", "amplitude_damping", "coherent_x"],
                "phaseflip": ["phaseflip", "dephasing", "amplitude_damping", "coherent_z"],
            },
            noise_families=["amplitude_damping", "coherent_x", "coherent_z"],
            families_by_code={
                "bitflip": ["amplitude_damping", "coherent_x"],
                "phaseflip": ["amplitude_damping", "coherent_z"],
            },
            strengths=[0.05],
            depths=[1],
            seeds=[5],
            fidelity_margin=0.01,
            trajectory_inconsistency_threshold=0.05,
            syndrome_consistent_threshold=0.9,
            hybrid_objective_tol=1.0e-9,
            tie_break_requires_syndrome_consistent=True,
            experiment_config="experiment/encoded_regime_map.yaml",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        families = {row["noise_family"] for row in result["rows"]}
        self.assertEqual(families, {"amplitude_damping", "coherent_x", "coherent_z"})
        self.assertIn("by_noise_family", result["tables"])

    def test_encoded_regime_map_can_pair_code_specific_families(self) -> None:
        result = run_encoded_regime_map(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["coherent_x"],
                "phaseflip": ["coherent_z"],
            },
            noise_families=["coherent_x", "coherent_z"],
            families_by_code={
                "bitflip": ["coherent_x"],
                "phaseflip": ["coherent_z"],
            },
            strengths=[0.05],
            depths=[1],
            seeds=[5],
            fidelity_margin=0.01,
            trajectory_inconsistency_threshold=0.05,
            syndrome_consistent_threshold=0.9,
            hybrid_objective_tol=1.0e-9,
            tie_break_requires_syndrome_consistent=True,
            experiment_config="experiment/encoded_coherent_validation.yaml",
        )
        self.assertEqual(result["overall"]["cases"], 2)
        seen = {(row["code_type"], row["noise_family"]) for row in result["rows"]}
        self.assertEqual(seen, {("bitflip", "coherent_x"), ("phaseflip", "coherent_z")})

    def test_coherent_veto_analysis_emits_summary_and_figures(self) -> None:
        result = run_coherent_veto_analysis(
            rows=_toy_coherent_rows(),
            source_stem="toy_coherent",
            case_limit=3,
            plot_stem="test_coherent_veto_analysis",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertEqual(len(result["tables"]["by_pair"]), 2)
        self.assertGreaterEqual(len(result["negative_gain_cases"]), 1)
        self.assertIn("## By Pair", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_coherent_veto_threshold_sweep_emits_operating_points(self) -> None:
        result = run_coherent_veto_threshold_sweep(
            rows=_toy_coherent_rows(),
            quantiles=[0.5, 0.75, 0.9],
            plot_stem="test_coherent_veto_threshold",
        )
        self.assertEqual(len(result["tables"]["operating_points"]), 2)
        self.assertGreaterEqual(len(result["tables"]["overall_threshold_sweep"]), 6)
        self.assertEqual(len(result["tables"]["false_safe_comparison"]), 3)
        self.assertIn("## Operating Points", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_coherent_v1_baseline_emits_v1_schema(self) -> None:
        result = run_coherent_v1_baseline(
            rows=_toy_coherent_rows(),
            flag_threshold_quantile=0.75,
            plot_stem="test_coherent_v1",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertIn("flag_threshold", result["overall"])
        self.assertEqual(len(result["tables"]["by_pair"]), 2)
        self.assertIn("flag_structural_risk", result["rows"][0])
        self.assertIn("## By Pair", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_coherent_casebook_extracts_negative_and_agreement_cases(self) -> None:
        result = run_coherent_casebook(
            rows=_toy_coherent_rows(),
            negative_case_limit=4,
            agreement_case_limit=4,
        )
        self.assertGreaterEqual(len(result["negative_gain_cases"]), 1)
        self.assertGreaterEqual(len(result["agreement_cases"]), 1)
        self.assertIn("# Coherent Casebook Negative Gain", result["negative_markdown"])
        self.assertIn("# Coherent Casebook Agreement", result["agreement_markdown"])

    def test_hybrid_c123_baseline_emits_policy_rows_and_markdown(self) -> None:
        result = run_hybrid_c123_baseline(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing"],
                "phaseflip": ["phaseflip", "dephasing"],
            },
            noise_families=["bitflip", "dephasing"],
            strengths=[0.05],
            depths=[1],
            seeds=[11],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            experiment_config="experiment/hybrid_c123_baseline.yaml",
            output_stem="test_hybrid_c123",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertIn("fid_gain_C2_mean", result["overall"])
        self.assertIn("false_safe_rate_C3", result["overall"])
        self.assertIn("false_safe_fidelity_rate_C3", result["overall"])
        self.assertIn("false_safe_fidelity_rate_C3R", result["overall"])
        self.assertIn("c3r_blocks_c2_switch_rate", result["overall"])
        self.assertIn("decision_disagreement_rate_C3R_vs_C2", result["overall"])
        self.assertIn("c3r_gate_uncertainty_rate_given_c2_B", result["overall"])
        self.assertIn("c3r_raw_syndrome_uncertainty_mean", result["overall"])
        self.assertIn("c3r_harmful_block_precision", result["overall"])
        self.assertIn("c3r_harmful_switch_recall", result["overall"])
        self.assertIn("c3r_net_intervention_gain", result["overall"])
        self.assertIn("fid_gain_q05_C3R", result["overall"])
        self.assertIn("fid_gain_cvar05_C3R", result["overall"])
        self.assertIn("oracle_regret_mean_C3R", result["overall"])
        self.assertGreaterEqual(len(result["tables"]["reason_summary"]), 3)
        self.assertIn("by_syndrome_obs_ratio", result["tables"])
        self.assertIn("by_syndrome_noise_prob", result["tables"])
        self.assertIn("c3r_by_uncertainty_bin", result["tables"])
        row = result["rows"][0]
        self.assertIn("candidate_C1", row)
        self.assertIn("candidate_C2", row)
        self.assertIn("candidate_C3", row)
        self.assertIn("candidate_C3R", row)
        self.assertIn("decision_reason_C2", row)
        self.assertIn("decision_reason_C3R", row)
        self.assertIn("logical_success_C3", row)
        self.assertIn("logical_success_C3R", row)
        self.assertIn("true_syndrome_mean_no_error", row)
        self.assertIn("observed_syndrome_mean_no_error", row)
        self.assertIn("false_safe_fidelity_flag_C3", row)
        self.assertIn("false_safe_fidelity_flag_C3R", row)
        self.assertIn("c3r_score_margin", row)
        self.assertIn("c3r_raw_syndrome_uncertainty", row)
        self.assertIn("decision_disagreement_C3R_vs_C2_flag", row)
        self.assertIn("c3r_harmful_c2_switch_flag", row)
        self.assertIn("c3r_beneficial_c2_switch_flag", row)
        self.assertIn("c3r_intervention_gain", row)
        self.assertIn("oracle_regret_C3R", row)
        self.assertIn("true_failure_boundary_flag_C3R", row)
        self.assertTrue(row["candidate_C3R"] == "A" or row["candidate_C2"] == "B")
        self.assertAlmostEqual(row["true_syndrome_mean_no_error"], row["observed_syndrome_mean_no_error"])
        self.assertIn("# Hybrid C1/C2/C3/C3R Baseline Summary", result["markdown"])
        self.assertIn("## C3R Switch Intervention Quality", result["markdown"])
        self.assertIn("## Tail Risk and Oracle Diagnostics", result["markdown"])

    def test_hybrid_ab_disagreement_uses_admissibility_or_fidelity_gap(self) -> None:
        self.assertFalse(
            _decision_disagreement_ab(
                candidate_A={"admissible": True, "fidelity_after": 0.995},
                candidate_B={"admissible": True, "fidelity_after": 0.999},
                fidelity_margin=0.01,
            )
        )
        self.assertTrue(
            _decision_disagreement_ab(
                candidate_A={"admissible": True, "fidelity_after": 1.000},
                candidate_B={"admissible": True, "fidelity_after": 0.970},
                fidelity_margin=0.01,
            )
        )
        self.assertTrue(
            _decision_disagreement_ab(
                candidate_A={"admissible": False, "fidelity_after": 1.000},
                candidate_B={"admissible": True, "fidelity_after": 1.000},
                fidelity_margin=0.01,
            )
        )

    def test_c3r_intervention_metrics_are_computed_from_rows(self) -> None:
        def row(fid_a: float, fid_b: float, c3r_candidate: str) -> dict[str, object]:
            item: dict[str, object] = {
                "backend": "test",
                "code_family": "phaseflip",
                "noise_family": "dephasing",
                "true_syndrome_mean_no_error": 1.0,
                "observed_syndrome_mean_no_error": 1.0,
                "syndrome_corruption_rate": 0.0,
                "syndrome_information_loss": 0.0,
                "observed_syndrome_consistent": True,
                "true_syndrome_consistent": True,
                "is_syndrome_partial": False,
                "is_syndrome_ambiguous": False,
                "decision_disagreement_rate_AB_flag": True,
                "decision_disagreement_rate_C1A_flag": False,
                "decision_disagreement_rate_C2A_flag": True,
                "decision_disagreement_rate_C3A_flag": True,
                "decision_disagreement_rate_C3RA_flag": c3r_candidate == "B",
                "decision_disagreement_C3R_vs_C2_flag": c3r_candidate != "B",
                "safe_case_flag": False,
                "risky_case_flag": True,
                "decision_reason_C1": "score_prefers_A",
                "decision_reason_C2": "score_prefers_B",
                "decision_reason_C3": "safety_prefers_B",
                "decision_reason_C3R": "c3r_blocks_high_syndrome_uncertainty"
                if c3r_candidate == "A"
                else "c3r_all_gates_pass_switch_to_B",
                "c3r_score_margin": 1.0,
                "c3r_structural_margin": 1.0,
                "c3r_violation_A": 0.2,
                "c3r_violation_B": 0.0,
                "c3r_syndrome_uncertainty": 1.0,
                "c3r_raw_syndrome_uncertainty": 1.0,
                "c3r_gate_c2_switch": True,
                "c3r_gate_score_margin": True,
                "c3r_gate_leave_A": True,
                "c3r_gate_B_safe": True,
                "c3r_gate_uncertainty": c3r_candidate == "B",
                "c3r_allow_B": c3r_candidate == "B",
            }
            gains = {
                "A": fid_a,
                "B": fid_b,
                "C1": fid_a,
                "C2": fid_b,
                "C3": fid_b,
                "C3R": fid_b if c3r_candidate == "B" else fid_a,
            }
            candidates = {"C1": "A", "C2": "B", "C3": "B", "C3R": c3r_candidate}
            for mode, gain in gains.items():
                item[f"fid_gain_{mode}"] = float(gain)
                item[f"nonworsen_{mode}"] = gain >= 0.0
                item[f"admissible_{mode}"] = True
                item[f"logical_success_{mode}"] = gain >= 0.0
                item[f"false_safe_flag_{mode}"] = False
                item[f"false_safe_fidelity_flag_{mode}"] = gain < -0.01
                item[f"failure_boundary_flag_{mode}"] = False
            item.update({f"candidate_{mode}": candidate for mode, candidate in candidates.items()})
            return enrich_c3r_row(item, fidelity_margin=0.01)

        rows = [
            row(0.10, 0.00, "A"),
            row(0.00, 0.05, "A"),
            row(0.10, 0.00, "B"),
        ]
        overall = _aggregate_rows(rows)
        self.assertEqual(overall["c2_B_count"], 3)
        self.assertEqual(overall["c3r_block_count"], 2)
        self.assertAlmostEqual(overall["c3r_block_rate_given_c2_B"], 2.0 / 3.0)
        self.assertAlmostEqual(overall["c3r_harmful_block_precision"], 0.5)
        self.assertAlmostEqual(overall["c3r_harmful_switch_recall"], 0.5)
        self.assertAlmostEqual(overall["c3r_beneficial_switch_block_rate"], 1.0)
        self.assertAlmostEqual(overall["c3r_beneficial_switch_retention"], 0.0)
        self.assertAlmostEqual(overall["c3r_prevented_loss_sum"], 0.10)
        self.assertAlmostEqual(overall["c3r_missed_gain_sum"], 0.05)
        self.assertAlmostEqual(overall["c3r_net_intervention_gain"], 0.05)
        self.assertAlmostEqual(overall["c3r_intervention_gain_sum"], 0.05)

    def test_c3r_gate_only_allows_c2_switches_when_all_gates_pass(self) -> None:
        candidate_A = {
            "candidate_label": "A",
            "admissible": False,
            "objective": 2.0,
            "traj_distance_to_observed": 0.20,
            "traj_distance_to_clean": 0.20,
            "fidelity_after": 0.96,
            "fid_gain": 0.00,
            "logical_success": False,
        }
        candidate_B = {
            "candidate_label": "B",
            "admissible": True,
            "objective": 1.0,
            "traj_distance_to_observed": 0.10,
            "traj_distance_to_clean": 0.05,
            "fidelity_after": 0.99,
            "fid_gain": 0.03,
            "logical_success": True,
        }
        report_A = {
            "penalties": {"phi_lap": 1.20},
            "thresholds": {"phi_lap": 1.00},
        }
        report_B = {
            "penalties": {"phi_lap": 0.90},
            "thresholds": {"phi_lap": 1.00},
        }
        cfg = C3RPolicyConfig(
            score_margin_min=0.05,
            admissibility_gap_min=0.10,
            b_violation_tolerance=0.0,
            uncertainty_max=0.50,
        )
        passed = _choose_c3r_policy(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            c2_policy={"candidate": "B"},
            score_A=1.00,
            score_B=1.10,
            report_A=report_A,
            report_B=report_B,
            syndrome_observation_ratio=1.0,
            syndrome_noise_prob=0.0,
            syndrome_ambiguity_level=0.0,
            measurement_error_prob=0.0,
            reset_error_prob=0.0,
            syndrome_information_loss=0.0,
            cfg=cfg,
        )
        self.assertEqual(passed["candidate"], "B")
        self.assertTrue(passed["allow_B"])
        self.assertTrue(passed["gate_score_margin"])
        self.assertTrue(passed["gate_leave_A"])
        self.assertTrue(passed["gate_B_safe"])

        c2_preserves_a = _choose_c3r_policy(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            c2_policy={"candidate": "A"},
            score_A=1.00,
            score_B=1.10,
            report_A=report_A,
            report_B=report_B,
            syndrome_observation_ratio=1.0,
            syndrome_noise_prob=0.0,
            syndrome_ambiguity_level=0.0,
            measurement_error_prob=0.0,
            reset_error_prob=0.0,
            syndrome_information_loss=0.0,
            cfg=cfg,
        )
        self.assertEqual(c2_preserves_a["candidate"], "A")
        self.assertFalse(c2_preserves_a["allow_B"])
        self.assertEqual(c2_preserves_a["decision_reason"], "c3r_c2_preserves_A")

        high_uncertainty = _choose_c3r_policy(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            c2_policy={"candidate": "B"},
            score_A=1.00,
            score_B=1.10,
            report_A=report_A,
            report_B=report_B,
            syndrome_observation_ratio=0.25,
            syndrome_noise_prob=0.0,
            syndrome_ambiguity_level=0.0,
            measurement_error_prob=0.0,
            reset_error_prob=0.0,
            syndrome_information_loss=0.0,
            cfg=cfg,
        )
        self.assertEqual(high_uncertainty["candidate"], "A")
        self.assertFalse(high_uncertainty["gate_uncertainty"])
        self.assertAlmostEqual(high_uncertainty["raw_syndrome_uncertainty"], 0.75)
        self.assertAlmostEqual(high_uncertainty["syndrome_uncertainty"], 0.75)
        self.assertEqual(high_uncertainty["decision_reason"], "c3r_blocks_high_syndrome_uncertainty")

        clipped_uncertainty = _choose_c3r_policy(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            c2_policy={"candidate": "B"},
            score_A=1.00,
            score_B=1.10,
            report_A=report_A,
            report_B=report_B,
            syndrome_observation_ratio=0.25,
            syndrome_noise_prob=0.50,
            syndrome_ambiguity_level=0.50,
            measurement_error_prob=0.25,
            reset_error_prob=0.25,
            syndrome_information_loss=0.50,
            cfg=C3RPolicyConfig(
                score_margin_min=0.05,
                admissibility_gap_min=0.10,
                b_violation_tolerance=0.0,
                uncertainty_max=1.0,
            ),
        )
        self.assertAlmostEqual(clipped_uncertainty["raw_syndrome_uncertainty"], 2.75)
        self.assertAlmostEqual(clipped_uncertainty["syndrome_uncertainty"], 1.0)
        self.assertTrue(clipped_uncertainty["gate_uncertainty"])

        default_threshold_blocks_clipped_uncertainty = _choose_c3r_policy(
            candidate_A=candidate_A,
            candidate_B=candidate_B,
            c2_policy={"candidate": "B"},
            score_A=1.00,
            score_B=1.10,
            report_A=report_A,
            report_B=report_B,
            syndrome_observation_ratio=0.25,
            syndrome_noise_prob=0.50,
            syndrome_ambiguity_level=0.50,
            measurement_error_prob=0.25,
            reset_error_prob=0.25,
            syndrome_information_loss=0.50,
            cfg=C3RPolicyConfig(
                score_margin_min=0.05,
                admissibility_gap_min=0.10,
                b_violation_tolerance=0.0,
            ),
        )
        self.assertFalse(default_threshold_blocks_clipped_uncertainty["gate_uncertainty"])
        self.assertEqual(
            default_threshold_blocks_clipped_uncertainty["decision_reason"],
            "c3r_blocks_high_syndrome_uncertainty",
        )

    def test_preferred_policy_uses_fidelity_false_safe_before_gain(self) -> None:
        row = {}
        for policy in ("C1", "C2", "C3", "C3R"):
            row[f"false_safe_rate_{policy}"] = 0.0
            row[f"false_safe_fidelity_rate_{policy}"] = 0.0
            row[f"fid_gain_{policy}_mean"] = 0.01
            row[f"logical_success_rate_{policy}"] = 0.5
            row[f"nonworsen_rate_{policy}"] = 0.5
            row[f"chosen_B_rate_{policy}"] = 0.5
        row["false_safe_fidelity_rate_C2"] = 0.20
        row["fid_gain_C2_mean"] = 0.08
        row["fid_gain_C3R_mean"] = 0.02
        row["logical_success_rate_C3R"] = 0.9
        row["nonworsen_rate_C3R"] = 0.9
        row["chosen_B_rate_C3R"] = 0.0

        preferred = _preferred_policy(row, safety_tolerance=0.0, gain_tolerance=0.0)

        self.assertEqual(preferred["preferred_policy"], "C3R")
        self.assertEqual(preferred["preferred_false_safe_fidelity_rate"], 0.0)

    def test_observed_syndrome_matches_true_when_uncorrupted(self) -> None:
        stats = {
            "mean_no_error_probability": 0.85,
            "min_no_error_probability": 0.85,
            "per_slice": [
                {
                    "tau_index": 0,
                    "probabilities": {
                        "(0, 0)": 0.85,
                        "(0, 1)": 0.10,
                        "(1, 0)": 0.05,
                        "(1, 1)": 0.00,
                    },
                    "no_error_probability": 0.85,
                }
            ],
        }
        observed = observe_syndrome_statistics(
            stats,
            cfg=SyndromeObservationConfig(),
            rng=np.random.default_rng(123),
        )
        self.assertAlmostEqual(observed["mean_no_error_probability"], 0.85)
        self.assertFalse(observed["is_syndrome_partial"])
        self.assertFalse(observed["is_syndrome_corrupted"])

    def test_observed_syndrome_changes_under_partial_or_noisy_observation(self) -> None:
        stats = {
            "mean_no_error_probability": 0.70,
            "min_no_error_probability": 0.70,
            "per_slice": [
                {
                    "tau_index": 0,
                    "probabilities": {
                        "(0, 0)": 0.70,
                        "(0, 1)": 0.20,
                        "(1, 0)": 0.10,
                        "(1, 1)": 0.0,
                    },
                    "no_error_probability": 0.70,
                }
            ],
        }
        partial = observe_syndrome_statistics(
            stats,
            cfg=SyndromeObservationConfig(observation_ratio=0.0),
            rng=np.random.default_rng(7),
        )
        noisy = observe_syndrome_statistics(
            stats,
            cfg=SyndromeObservationConfig(observation_ratio=1.0, noise_prob=1.0),
            rng=np.random.default_rng(7),
        )
        self.assertNotAlmostEqual(partial["mean_no_error_probability"], 0.70)
        self.assertTrue(partial["is_syndrome_partial"])
        self.assertTrue(partial["is_syndrome_corrupted"])
        self.assertEqual(noisy["dominant"], "11")
        self.assertAlmostEqual(noisy["mean_no_error_probability"], 0.0)
        self.assertTrue(noisy["is_syndrome_corrupted"])

    def test_hybrid_c123_regime_map_emits_figures_and_preferred_policy(self) -> None:
        result = run_hybrid_c123_regime_map(
            codes=["bitflip", "phaseflip"],
            state_configs={
                "bitflip": "states/repetition_bitflip.yaml",
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "bitflip": ["bitflip", "depolarizing", "phaseflip"],
                "phaseflip": ["phaseflip", "dephasing", "bitflip"],
            },
            noise_families=["bitflip", "dephasing"],
            strengths=[0.05],
            depths=[1],
            seeds=[11],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            syndrome_obs_cfg=SyndromeObservationConfig(
                observation_ratio=0.5,
                noise_prob=0.25,
                ambiguity_level=0.0,
                measurement_error_prob=0.0,
                reset_error_prob=0.0,
                consistency_threshold=0.9,
            ),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            regime_safety_tolerance=0.02,
            regime_gain_tolerance=0.005,
            experiment_config="experiment/hybrid_c123_regime_map.yaml",
            output_stem="test_hybrid_c123_regime_map",
            plot_prefix="test_hybrid_c123",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertIn("by_regime_cell", result["tables"])
        self.assertIn("preferred_policy_summary", result["tables"])
        self.assertIn("by_syndrome_obs_ratio", result["tables"])
        self.assertIn("by_syndrome_noise_prob", result["tables"])
        self.assertEqual(result["syndrome_observation"]["observation_ratio"], 0.5)
        self.assertEqual(result["syndrome_observation"]["noise_prob"], 0.25)
        self.assertEqual(result["tables"]["by_syndrome_obs_ratio"][0]["syndrome_observation_ratio"], 0.5)
        self.assertEqual(result["tables"]["by_syndrome_noise_prob"][0]["syndrome_noise_prob"], 0.25)
        self.assertGreaterEqual(len(result["figures"]), 5)
        self.assertIn("# Hybrid C1/C2/C3/C3R Regime Map", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_hybrid_c123_casebook_extracts_type_groups(self) -> None:
        rows = [
            {
                "experiment_id": "type1",
                "code_family": "bitflip",
                "noise_family": "bitflip",
                "noise_strength": 0.05,
                "noise_depth": 1,
                "seed": 11,
                "syndrome_label": "00",
                "candidate_C1": "A",
                "candidate_C2": "A",
                "candidate_C3": "A",
                "decision_reason_C1": "default_A",
                "decision_reason_C2": "score_prefers_A",
                "decision_reason_C3": "safety_prefers_A",
                "admissible_C1": True,
                "admissible_C2": True,
                "admissible_C3": True,
                "false_safe_flag_A": False,
                "false_safe_flag_C1": False,
                "false_safe_flag_C2": False,
                "false_safe_flag_C3": False,
                "fid_gain_C1": 0.020,
                "fid_gain_C2": 0.021,
                "fid_gain_C3": 0.019,
                "fidelity_after_C1": 0.99,
                "fidelity_after_C2": 0.991,
                "fidelity_after_C3": 0.989,
                "logical_success_C1": True,
                "logical_success_C2": True,
                "logical_success_C3": True,
                "score_C2_A": 0.8,
                "score_C2_B": 0.7,
                "score_C3_A": 0.9,
                "score_C3_B": 0.6,
            },
            {
                "experiment_id": "type2",
                "code_family": "phaseflip",
                "noise_family": "dephasing",
                "noise_strength": 0.10,
                "noise_depth": 2,
                "seed": 12,
                "syndrome_label": "00",
                "candidate_C1": "A",
                "candidate_C2": "B",
                "candidate_C3": "A",
                "decision_reason_C1": "default_A",
                "decision_reason_C2": "score_prefers_B",
                "decision_reason_C3": "safety_prefers_A",
                "admissible_C1": True,
                "admissible_C2": True,
                "admissible_C3": True,
                "false_safe_flag_A": False,
                "false_safe_flag_C1": False,
                "false_safe_flag_C2": False,
                "false_safe_flag_C3": False,
                "fid_gain_C1": 0.010,
                "fid_gain_C2": 0.035,
                "fid_gain_C3": 0.015,
                "fidelity_after_C1": 0.97,
                "fidelity_after_C2": 0.995,
                "fidelity_after_C3": 0.975,
                "logical_success_C1": False,
                "logical_success_C2": True,
                "logical_success_C3": False,
                "score_C2_A": 0.7,
                "score_C2_B": 0.9,
                "score_C3_A": 0.8,
                "score_C3_B": 0.75,
            },
            {
                "experiment_id": "type3",
                "code_family": "phaseflip",
                "noise_family": "phaseflip",
                "noise_strength": 0.15,
                "noise_depth": 3,
                "seed": 13,
                "syndrome_label": "01",
                "candidate_C1": "A",
                "candidate_C2": "A",
                "candidate_C3": "B",
                "decision_reason_C1": "default_A",
                "decision_reason_C2": "score_prefers_A",
                "decision_reason_C3": "hard_inadmissibility_block",
                "admissible_C1": False,
                "admissible_C2": False,
                "admissible_C3": True,
                "false_safe_flag_A": True,
                "false_safe_flag_C1": True,
                "false_safe_flag_C2": True,
                "false_safe_flag_C3": False,
                "fid_gain_C1": -0.010,
                "fid_gain_C2": 0.000,
                "fid_gain_C3": 0.018,
                "fidelity_after_C1": 0.94,
                "fidelity_after_C2": 0.95,
                "fidelity_after_C3": 0.968,
                "logical_success_C1": False,
                "logical_success_C2": False,
                "logical_success_C3": False,
                "score_C2_A": 0.4,
                "score_C2_B": 0.3,
                "score_C3_A": None,
                "score_C3_B": None,
            },
            {
                "experiment_id": "type4",
                "code_family": "bitflip",
                "noise_family": "depolarizing",
                "noise_strength": 0.20,
                "noise_depth": 4,
                "seed": 14,
                "syndrome_label": "00",
                "candidate_C1": "A",
                "candidate_C2": "B",
                "candidate_C3": "A",
                "decision_reason_C1": "default_A",
                "decision_reason_C2": "score_prefers_B",
                "decision_reason_C3": "safety_prefers_A",
                "admissible_C1": True,
                "admissible_C2": True,
                "admissible_C3": True,
                "false_safe_flag_A": False,
                "false_safe_flag_C1": False,
                "false_safe_flag_C2": True,
                "false_safe_flag_C3": False,
                "fid_gain_C1": 0.005,
                "fid_gain_C2": 0.030,
                "fid_gain_C3": 0.006,
                "fidelity_after_C1": 0.955,
                "fidelity_after_C2": 0.98,
                "fidelity_after_C3": 0.956,
                "logical_success_C1": False,
                "logical_success_C2": False,
                "logical_success_C3": False,
                "score_C2_A": 0.5,
                "score_C2_B": 0.8,
                "score_C3_A": 0.7,
                "score_C3_B": 0.6,
            },
        ]
        result = run_hybrid_c123_casebook(rows=rows, per_group=2, gain_margin=0.01, similarity_tol=0.005)
        self.assertIn("type1_c1_sufficient", result["groups"])
        self.assertIn("type2_c2_wins", result["groups"])
        self.assertIn("type3_c3_necessary", result["groups"])
        self.assertIn("type4_c2_overaggressive", result["groups"])
        self.assertIn("typeR1_c2_sufficient_c3r_allows", result["groups"])
        self.assertIn("typeR2_c3r_prevents_harmful_switch", result["groups"])
        self.assertIn("typeR3_c3r_overconservative", result["groups"])
        self.assertIn("typeR4_c3r_inactive", result["groups"])
        self.assertEqual(len(result["tables"]["group_summary"]), 8)
        self.assertIn("# Hybrid C1/C2/C3/C3R Casebook", result["markdown"])

    def test_partial_syndrome_baseline_emits_ratio_tables_and_figures(self) -> None:
        result = run_partial_syndrome_baseline(
            codes=["phaseflip"],
            state_configs={
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "phaseflip": ["phaseflip", "dephasing", "bitflip"],
            },
            noise_families=["phaseflip"],
            strengths=[0.05],
            depths=[1],
            seeds=[11],
            observation_ratios=[1.0, 0.5],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            syndrome_obs_cfg=SyndromeObservationConfig(
                observation_ratio=1.0,
                noise_prob=0.0,
                ambiguity_level=0.0,
                measurement_error_prob=0.0,
                reset_error_prob=0.0,
                consistency_threshold=0.9,
            ),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            regime_safety_tolerance=0.02,
            regime_gain_tolerance=0.005,
            experiment_config="experiment/partial_syndrome_baseline.yaml",
            output_stem="test_partial_syndrome",
            plot_prefix="test_partial_syndrome",
        )
        self.assertEqual(result["overall"]["cases"], 2)
        self.assertIn("by_syndrome_obs_ratio", result["tables"])
        self.assertIn("preferred_policy_counts_by_ratio", result["tables"])
        self.assertEqual(len(result["tables"]["by_syndrome_obs_ratio"]), 2)
        self.assertEqual(len(result["tables"]["preferred_policy_counts_by_ratio"]), 2)
        self.assertIn("# Partial Syndrome Baseline", result["markdown"])
        self.assertIn("## C3R Gate Summary", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_noisy_syndrome_baseline_emits_noise_tables_and_figures(self) -> None:
        result = run_noisy_syndrome_baseline(
            codes=["phaseflip"],
            state_configs={
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "phaseflip": ["phaseflip", "dephasing", "bitflip"],
            },
            noise_families=["phaseflip"],
            strengths=[0.05],
            depths=[1],
            seeds=[11],
            syndrome_noise_probs=[0.0, 0.05],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            syndrome_obs_cfg=SyndromeObservationConfig(
                observation_ratio=1.0,
                noise_prob=0.0,
                ambiguity_level=0.0,
                measurement_error_prob=0.0,
                reset_error_prob=0.0,
                consistency_threshold=0.9,
            ),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            regime_safety_tolerance=0.02,
            regime_gain_tolerance=0.005,
            experiment_config="experiment/noisy_syndrome_baseline.yaml",
            output_stem="test_noisy_syndrome",
            plot_prefix="test_noisy_syndrome",
        )
        self.assertEqual(result["overall"]["cases"], 2)
        self.assertIn("by_syndrome_noise_prob", result["tables"])
        self.assertIn("preferred_policy_counts_by_noise", result["tables"])
        self.assertEqual(len(result["tables"]["by_syndrome_noise_prob"]), 2)
        self.assertEqual(len(result["tables"]["preferred_policy_counts_by_noise"]), 2)
        self.assertIn("# Noisy Syndrome Baseline", result["markdown"])
        self.assertIn("## C3R Gate Summary", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_partial_noisy_syndrome_regime_map_emits_combo_tables_and_figures(self) -> None:
        result = run_partial_noisy_syndrome_regime_map(
            codes=["phaseflip"],
            state_configs={
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "phaseflip": ["phaseflip", "dephasing", "bitflip"],
            },
            noise_families=["phaseflip"],
            strengths=[0.05],
            depths=[1],
            seeds=[11],
            observation_ratios=[0.5],
            syndrome_noise_probs=[0.03, 0.05],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            syndrome_obs_cfg=SyndromeObservationConfig(
                observation_ratio=1.0,
                noise_prob=0.0,
                ambiguity_level=0.0,
                measurement_error_prob=0.0,
                reset_error_prob=0.0,
                consistency_threshold=0.9,
            ),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            regime_safety_tolerance=0.02,
            regime_gain_tolerance=0.005,
            experiment_config="experiment/partial_noisy_syndrome_regime_map.yaml",
            output_stem="test_partial_noisy_syndrome",
            plot_prefix="test_partial_noisy_syndrome",
        )
        self.assertEqual(result["overall"]["cases"], 2)
        self.assertIn("by_obs_ratio_and_noise_prob", result["tables"])
        self.assertIn("preferred_policy_counts_by_combo", result["tables"])
        self.assertEqual(len(result["tables"]["by_obs_ratio_and_noise_prob"]), 2)
        self.assertEqual(len(result["tables"]["preferred_policy_counts_by_combo"]), 2)
        self.assertIn("# Partial+Noisy Syndrome Regime Map", result["markdown"])
        self.assertIn("## C3R Gate Summary", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_ambiguity_measurement_syndrome_regime_map_emits_combo_tables_and_figures(self) -> None:
        result = run_ambiguity_measurement_syndrome_regime_map(
            codes=["phaseflip"],
            state_configs={
                "phaseflip": "states/repetition_phaseflip.yaml",
            },
            kinds_by_code={
                "phaseflip": ["phaseflip", "dephasing", "bitflip"],
            },
            noise_families=["phaseflip"],
            strengths=[0.05],
            depths=[2],
            seeds=[11],
            ambiguity_levels=[0.0, 0.5],
            measurement_reset_probs=[0.0, 0.05],
            fidelity_margin=0.01,
            logical_success_threshold=0.99,
            c2_cfg=PolicyScoreConfig(lambda_s=1.0, lambda_t=1.0, lambda_i=2.0, lambda_o=0.5),
            c3_cfg=PolicyScoreConfig(lambda_s=0.8, lambda_t=1.2, lambda_i=5.0, lambda_o=0.3),
            syndrome_obs_cfg=SyndromeObservationConfig(
                observation_ratio=1.0,
                noise_prob=0.0,
                ambiguity_level=0.0,
                measurement_error_prob=0.0,
                reset_error_prob=0.0,
                consistency_threshold=0.9,
            ),
            c1_objective_tol=1.0e-9,
            c1_tie_break_requires_syndrome_consistent=True,
            regime_safety_tolerance=0.02,
            regime_gain_tolerance=0.005,
            experiment_config="experiment/ambiguity_measurement_syndrome_regime_map.yaml",
            output_stem="test_ambiguity_measurement",
            plot_prefix="test_ambiguity_measurement",
        )
        self.assertEqual(result["overall"]["cases"], 4)
        self.assertIn("by_ambiguity_and_measurement_reset", result["tables"])
        self.assertIn("preferred_policy_counts_by_combo", result["tables"])
        self.assertEqual(len(result["tables"]["by_ambiguity_and_measurement_reset"]), 4)
        self.assertEqual(len(result["tables"]["preferred_policy_counts_by_combo"]), 4)
        self.assertIn("# Ambiguity + Measurement/Reset Syndrome Regime Map", result["markdown"])
        self.assertIn("## C3R Gate Summary", result["markdown"])
        for path in result["figures"].values():
            self.assertTrue(Path(path).exists())

    def test_c2_preferred_analysis_emits_tables_and_heatmap(self) -> None:
        payload = {
            "grid": {
                "codes": ["phaseflip"],
                "noise_families": ["phaseflip", "dephasing"],
                "strengths": [0.05, 0.10],
                "depths": [1, 2],
            },
            "policies": {
                "c2": {"lambda_s": 1.0, "lambda_t": 1.0, "lambda_i": 2.0, "lambda_o": 0.5},
            },
            "rows": [
                {
                    "experiment_id": "c2-cell-1",
                    "code_family": "phaseflip",
                    "noise_family": "phaseflip",
                    "noise_strength": 0.05,
                    "noise_depth": 1,
                    "admissible_A": True,
                    "admissible_B": True,
                    "admissible_C1": True,
                    "admissible_C2": True,
                    "admissible_C3": True,
                    "false_safe_flag_A": False,
                    "false_safe_flag_C1": False,
                    "false_safe_flag_C2": False,
                    "false_safe_flag_C3": False,
                    "fid_gain_A": 0.02,
                    "fid_gain_B": 0.05,
                    "fid_gain_C1": 0.02,
                    "fid_gain_C2": 0.05,
                    "fid_gain_C3": 0.05,
                    "fidelity_after_A": 0.97,
                    "fidelity_after_B": 1.0,
                    "fidelity_after_C1": 0.97,
                    "fidelity_after_C2": 1.0,
                    "fidelity_after_C3": 1.0,
                    "decision_reason_C1": "tie_break_objective",
                    "decision_reason_C2": "score_prefers_A",
                    "decision_reason_C3": "safety_prefers_A",
                    "logical_success_C1": False,
                    "logical_success_C2": True,
                    "logical_success_C3": True,
                    "score_C2_A": 1.00,
                    "score_C2_B": 0.80,
                    "score_C3_A": 0.90,
                    "score_C3_B": 0.70,
                    "objective_A": 2.0,
                    "objective_B": 1.5,
                    "traj_distance_A": 0.20,
                    "traj_distance_B": 0.10,
                },
                {
                    "experiment_id": "c1-cell-1",
                    "code_family": "phaseflip",
                    "noise_family": "dephasing",
                    "noise_strength": 0.10,
                    "noise_depth": 2,
                    "admissible_A": True,
                    "admissible_B": True,
                    "admissible_C1": True,
                    "admissible_C2": True,
                    "admissible_C3": True,
                    "false_safe_flag_A": False,
                    "false_safe_flag_C1": False,
                    "false_safe_flag_C2": False,
                    "false_safe_flag_C3": False,
                    "fid_gain_A": 0.10,
                    "fid_gain_B": 0.08,
                    "fid_gain_C1": 0.10,
                    "fid_gain_C2": 0.08,
                    "fid_gain_C3": 0.08,
                    "fidelity_after_A": 1.0,
                    "fidelity_after_B": 0.98,
                    "fidelity_after_C1": 1.0,
                    "fidelity_after_C2": 0.98,
                    "fidelity_after_C3": 0.98,
                    "decision_reason_C1": "keep_syndrome",
                    "decision_reason_C2": "score_prefers_A",
                    "decision_reason_C3": "safety_prefers_A",
                    "logical_success_C1": True,
                    "logical_success_C2": False,
                    "logical_success_C3": False,
                    "score_C2_A": 1.20,
                    "score_C2_B": 0.70,
                    "score_C3_A": 1.00,
                    "score_C3_B": 0.50,
                    "objective_A": 1.0,
                    "objective_B": 1.4,
                    "traj_distance_A": 0.10,
                    "traj_distance_B": 0.20,
                },
            ],
            "tables": {
                "by_regime_cell": [
                    {
                        "code_family": "phaseflip",
                        "noise_family": "phaseflip",
                        "noise_strength": 0.05,
                        "noise_depth": 1,
                        "preferred_policy": "C2",
                        "fid_gain_C1_mean": 0.02,
                        "fid_gain_C2_mean": 0.05,
                        "fid_gain_C3_mean": 0.05,
                        "false_safe_rate_C1": 0.0,
                        "false_safe_rate_C2": 0.0,
                        "false_safe_rate_C3": 0.0,
                        "chosen_B_rate_C1": 1.0,
                        "chosen_B_rate_C2": 0.0,
                        "chosen_B_rate_C3": 0.0,
                        "admissible_rate_A": 1.0,
                        "admissible_rate_B": 1.0,
                        "decision_disagreement_rate_AB": 1.0,
                    },
                    {
                        "code_family": "phaseflip",
                        "noise_family": "dephasing",
                        "noise_strength": 0.10,
                        "noise_depth": 2,
                        "preferred_policy": "C1",
                        "fid_gain_C1_mean": 0.10,
                        "fid_gain_C2_mean": 0.08,
                        "fid_gain_C3_mean": 0.08,
                        "false_safe_rate_C1": 0.0,
                        "false_safe_rate_C2": 0.0,
                        "false_safe_rate_C3": 0.0,
                        "chosen_B_rate_C1": 0.0,
                        "chosen_B_rate_C2": 0.0,
                        "chosen_B_rate_C3": 0.0,
                        "admissible_rate_A": 1.0,
                        "admissible_rate_B": 1.0,
                        "decision_disagreement_rate_AB": 0.0,
                    },
                ],
            },
        }
        result = run_c2_preferred_analysis(payload=payload, type2_gain_margin=0.01, type2_limit=5)
        self.assertEqual(result["summary"]["c2_preferred_cell_count"], 1)
        self.assertIn("c2_preferred_distribution", result["tables"])
        self.assertIn("c2_score_component_decomposition", result["tables"])
        self.assertIn("c2_type2_casebook_extended", result["tables"])
        self.assertTrue(Path(result["figure"]["c2_preferred_heatmap"]).exists())
        self.assertIn("recommended_interpretation", result["interpretation"])


if __name__ == "__main__":
    unittest.main()

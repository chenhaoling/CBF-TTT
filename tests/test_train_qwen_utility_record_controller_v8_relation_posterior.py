import unittest

from tasks.train_qwen_utility_record_controller_v8_relation_posterior import (
    extended_gate,
)


class TestV8RelationPosteriorTraining(unittest.TestCase):
    def test_extended_gate_adds_registered_improvements(self):
        mechanisms = {
            name: 0.0 for name in (
                "different_subject_same_relation", "explicit_reset_revoke",
                "mixed_lifecycle", "same_subject_different_relation",
                "semantic_duplicate", "semantic_overwrite", "stale_repetition",
            )
        }
        domains = {
            name: {"oracle_headroom": .02, "physics": {
                "per_mechanism_accuracy": mechanisms,
            }} for name in ("templama", "lama")
        }
        seeds = []
        for seed in (26101651, 26101652, 26101653):
            metric = {
                "gain_over_lifecycle_physics": .016,
                "oracle_gap_closure": .8,
                "added_unrelated_errors": 0,
                "per_mechanism_gain": mechanisms,
            }
            seeds.append({
                "selected_metric": {"domains": {
                    "templama": metric, "lama": metric,
                }},
                "head_metrics": {"lama": {"benefit": {
                    "average_precision": .40,
                }}},
            })
        diagnostic = {
            "diagnosis_evidence": {"joint_lama_domain_specific_ceiling": {
                "mean_gain": .008,
            }},
            "ranking_and_calibration": {
                str(seed): {"lama": {"benefit": {"overall": {
                    "average_precision": .25,
                }}}} for seed in (26101651, 26101652, 26101653)
            },
        }
        gate = extended_gate(seeds, domains, diagnostic)
        self.assertTrue(gate["passed_pending_independent_audit"])
        self.assertAlmostEqual(gate["lama_gain_over_v7"], .008)
        self.assertAlmostEqual(gate["lama_benefit_average_precision_gain"], .15)


if __name__ == "__main__":
    unittest.main()

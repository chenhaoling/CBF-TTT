import unittest

from scripts.audit_pair_localized_relation_memory_s4_data import candidate_schema_safe


class PairLocalizedRelationMemoryS4DataAuditTests(unittest.TestCase):
    def test_candidate_schema(self):
        candidate = {
            "candidate_id": "x",
            "center_turn": 0,
            "turn_indices": [0],
            "model_text": "Arguments: [ARG1] ; [ARG2]",
            "features": {
                "arg1_present": True,
                "arg2_present": True,
                "distance_to_arg1": 0.0,
                "distance_to_arg2": 0.0,
                "normalized_position": 0.0,
            },
        }
        self.assertTrue(candidate_schema_safe(candidate))
        candidate["gold_trigger"] = 1
        self.assertFalse(candidate_schema_safe(candidate))


if __name__ == "__main__":
    unittest.main()

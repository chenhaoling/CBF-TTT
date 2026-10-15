import unittest

from tasks.extract_pair_localized_relation_memory_s4_embeddings import (
    candidate_rows,
    control_rows,
    partition,
)


def mention():
    return {
        "source": "dialogre", "source_row_sha256": "m", "relation": "r",
        "dialogue_sha256": "d",
        "candidates": [{
            "candidate_id": "c", "center_turn": 0, "model_text": "x",
            "features": {"arg1_present": True, "arg2_present": False,
                         "distance_to_arg1": 0.0, "distance_to_arg2": 1.0,
                         "normalized_position": 0.0},
        }],
        "deterministic_windows": {
            str(i): {"model_text": str(i)} for i in range(3)
        },
        "oracle_window_audit": {"available": True, "model_text": "oracle"},
    }


class PairLocalizedRelationMemoryS4EmbeddingTests(unittest.TestCase):
    def test_flatten_rows(self):
        self.assertEqual(len(candidate_rows([mention()])), 1)
        controls = control_rows([mention()])
        self.assertEqual([row["view"] for row in controls],
                         ["deterministic_r0", "deterministic_r1",
                          "deterministic_r2", "oracle"])

    def test_partition(self):
        rows = list(range(9))
        left, right = partition(rows, 0, 2), partition(rows, 1, 2)
        self.assertEqual(sorted(index for index, _ in left + right), list(range(9)))
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})


if __name__ == "__main__":
    unittest.main()

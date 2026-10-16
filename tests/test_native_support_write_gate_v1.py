import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_native_support_write_gate_v1 import (
    MECHANISM_SOURCE, convert_episode, pack_chunk, stable_hash,
)
from tasks.native_support_write_gate_v1 import support_features


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 1 for char in text]


def record(index, role="filler"):
    return {"record_id": f"r{index}", "input": {"statement": f"S {index}",
            "subject": "A", "value": f"V{index}"},
            "audit": {"role": role, "semantic_key": "k", "relation_id": "P"}}


def query(role, answer="V0"):
    candidates = ["V0", "V1", "V2", "V3", "V4", "unknown"]
    return {"role": role, "input": {"query": "A maps to ____", "subject": "A"},
            "audit": {"semantic_key": "k"}, "answer": answer, "candidates": candidates}


def episode(mechanism="semantic_duplicate"):
    events = [{"event_index": i, "record": record(i), "reset": False, "revoke_keys": []}
              for i in range(8)]
    boundaries = []
    for i, event_index in enumerate((5, 6, 7)):
        boundaries.append({"boundary_index": i, "event_index": event_index,
                           "queries": [query("current_exact"), query("current_rephrase"),
                                       query("retained_1"), query("retained_2"), query("anchor")]})
    return {"episode_id": "e", "source_group": "g", "twin": 0,
            "mechanism": mechanism, "events": events, "boundaries": boundaries}


class DataTests(unittest.TestCase):
    def test_chunk_and_six_mechanisms(self):
        self.assertEqual(len(MECHANISM_SOURCE), 6)
        self.assertEqual(len(pack_chunk(TinyTokenizer(), ["abc"])), 4096)

    def test_controller_input_has_no_semantic_plaintext(self):
        model, audit = convert_episode(TinyTokenizer(), episode(), "useful_repeat")
        self.assertEqual(len(model["boundaries"]), 3)
        self.assertNotIn("mechanism", model)
        self.assertIn("mechanism", audit)
        self.assertTrue(all(len(row["chunk_ids"]) == 4096 for row in model["boundaries"]))
        self.assertTrue(all(len(row["queries"]) == 5 for row in model["boundaries"]))

    def test_exact_duplicate_uses_exact_initial_surface(self):
        model, audit = convert_episode(TinyTokenizer(), episode("explicit_reset_revoke"), "exact_duplicate")
        initial = audit["setup_statements"][0]
        self.assertTrue(all(row["incoming_statement"] == initial for row in audit["boundaries"]))


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch required")
class FeatureTests(unittest.TestCase):
    def test_support_feature_dimensions_and_numeric_state(self):
        import torch
        layers = [0, 6]
        current = {layer: {"centroid": torch.ones(128) * (layer + 1),
                           "variance": torch.ones(128)} for layer in layers}
        values, state = support_features(current, None, layers)
        self.assertEqual(len(values), 259 * len(layers))
        values2, _ = support_features(current, state, layers)
        self.assertAlmostEqual(values2[256], 1.0, places=6)
        self.assertEqual(values2[258], 1.0)

    def test_native_reject_admit_branch_uses_actual_cache(self):
        from tasks.native_support_write_gate_v1 import branch_boundary, fast_weight_digest
        from tasks.cbf_checkpoint_native import NativeSession
        from tests.test_ttt_train_infer import TrainInferTests
        model = TrainInferTests().pair()[1].eval().requires_grad_(False)
        chunk = [1, 2, 3, 4][:model.config.ttt_chunk]
        parent = NativeSession(model)
        before = fast_weight_digest(parent)
        reject, admit, _, support, audit = branch_boundary(parent, chunk)
        self.assertEqual(before, fast_weight_digest(parent))
        self.assertEqual(before, fast_weight_digest(reject))
        self.assertNotEqual(before, fast_weight_digest(admit))
        self.assertEqual(set(support), set(model.config.ttt_layers))
        self.assertEqual(audit["current_hidden_max_abs_error"], 0.0)


if __name__ == "__main__": unittest.main()

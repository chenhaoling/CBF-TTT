import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_semantic_controller_source_inventory import (
    assign_relation_folds,
    construct,
    mask_surface,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value))


def cf_row(case, subject, relation):
    return {"case_id": case, "requested_rewrite": {
        "subject": subject, "relation_id": relation, "prompt": "{} has relation",
        "target_true": {"str": f"old-{case}"}, "target_new": {"str": f"new-{case}"}},
        "paraphrase_prompts": [f"What relation applies to {subject}?"]}


def mq_row(case, subject, relation):
    return {"case_id": case, "requested_rewrite": [{
        "subject": subject, "relation_id": relation, "prompt": "{} has temporal relation",
        "target_true": {"str": f"before-{case}"}, "target_new": {"str": f"after-{case}"}}],
        "orig": {"edit_triples_idx": [0]},
        "single_hops": [{"question": f"Question for {subject}", "answer": f"before-{case}"}],
        "new_single_hops": [{"question": f"Question for {subject}", "answer": f"after-{case}"}]}


class SemanticControllerSourceInventoryTest(unittest.TestCase):
    def test_mask_surface_masks_subject_and_values(self):
        self.assertEqual(mask_surface("Ada moved from Rome to Paris", "Ada", ["Rome", "Paris"]),
                         "[SUBJECT] moved from [VALUE] to [VALUE]")

    def test_relation_assignment_is_deterministic_and_nonempty(self):
        rows = [{"relation_id": f"P{i}"} for i in range(10)]
        self.assertEqual(assign_relation_folds(rows, 5),
                         assign_relation_folds(list(reversed(rows)), 5))
        self.assertEqual(set(assign_relation_folds(rows, 5).values()), set(range(5)))

    def test_construct_is_relation_and_subject_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            # Repeated subjects deliberately span relations.  The deterministic
            # filter must remove conflicting rows rather than leak across folds.
            counterfact = [cf_row(i, f"subject-{i // 2}", f"P{i}") for i in range(10)]
            mquake = [mq_row(i, f"mq-{i}", f"Q{i % 10}") for i in range(30)]
            calibration = tmp_path / "calibration.jsonl"
            facts = [{"subject": "zsre-a", "prompt": "p", "rephrase_prompt": "r",
                      "original": "o", "updated": "u", "source_row_sha256": "a" * 64},
                     {"subject": "subject-0", "prompt": "p2", "rephrase_prompt": "r2",
                      "original": "o2", "updated": "u2", "source_row_sha256": "b" * 64}]
            calibration.write_text(json.dumps({"split": "calibration", "facts": facts}) + "\n")
            cf_path, mq_path = tmp_path / "cf.json", tmp_path / "mq.json"
            dump(cf_path, counterfact); dump(mq_path, mquake)
            rows, zsre, excluded, manifest = construct(
                cf_path, mq_path, calibration, folds=5, enforce_known_hashes=False)
            for field in ("relation_id", "subject_normalized"):
                seen = {}
                for row in rows:
                    value = row[field]
                    self.assertTrue(value not in seen or seen[value] == row["outer_fold"])
                    seen[value] = row["outer_fold"]
            self.assertEqual(manifest["checks"], {
                "relation_disjoint": True, "subject_disjoint": True,
                "all_folds_nonempty": True})
            self.assertFalse(manifest["mquake_dev_used"])
            self.assertTrue(any(row["reason"] == "mquake_dev_reserved" for row in excluded))
            self.assertTrue(any(row["reason"] == "cross_fold_subject_conflict"
                                for row in excluded))
            self.assertEqual([row["subject"] for row in zsre], ["zsre-a"])
            self.assertTrue(any(row["reason"] == "semantic_subject_overlap"
                                for row in excluded))

    def test_zsre_input_rejects_noncalibration(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            cf_path = tmp_path / "cf.json"; mq_path = tmp_path / "mq.json"
            zsre_path = tmp_path / "z.jsonl"
            dump(cf_path, [cf_row(i, f"c{i}", f"P{i}") for i in range(5)])
            dump(mq_path, [mq_row(i, f"m{i}", f"Q{i}") for i in range(20)])
            zsre_path.write_text(json.dumps({"split": "test", "facts": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "non-calibration"):
                construct(cf_path, mq_path, zsre_path, enforce_known_hashes=False)


if __name__ == "__main__":
    unittest.main()

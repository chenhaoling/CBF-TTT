import unittest
from pathlib import Path

from tasks.build_docred_relation_space import (
    construct_pairs, masked_evidence, raw_candidates, selected_sentences,
)


def document():
    return {
        "title": "example",
        "sents": [["Alice", "founded", "Acme", "."],
                  ["She", "later", "led", "Acme", "."]],
        "vertexSet": [
            [{"name": "Alice", "sent_id": 0, "pos": [0, 1]}],
            [{"name": "Acme", "sent_id": 0, "pos": [2, 3]},
             {"name": "Acme", "sent_id": 1, "pos": [3, 4]}],
        ],
        "labels": [{"r": "P1", "h": 0, "t": 1, "evidence": [0]}],
    }


def instance(relation, doc, index):
    return {"partition": "representation_train", "relation_id": relation,
            "document_id": doc, "instance_id": f"{relation}-{doc}-{index}",
            "source_label_sha256": f"{relation}-{doc}-{index}",
            "surface": f"[SUBJECT] relation {relation} evidence {index} [VALUE]"}


class DocredRelationSpaceTest(unittest.TestCase):
    def test_masked_evidence_replaces_relation_entities(self):
        doc = document(); label = doc["labels"][0]
        self.assertEqual(selected_sentences(doc, label), [0])
        text, sentences = masked_evidence(doc, label)
        self.assertEqual(sentences, [0])
        self.assertEqual(text, "[SUBJECT] founded [VALUE] .")

    def test_missing_entity_sentence_is_added(self):
        doc = document(); label = {"r": "P1", "h": 0, "t": 1, "evidence": [1]}
        self.assertEqual(selected_sentences(doc, label), [0, 1])

    def test_pairs_are_balanced_and_semantically_correct(self):
        rows = []
        for relation in ("P1", "P2", "P3"):
            for index in range(4):
                rows.append(instance(relation, f"doc-{relation}-{index}", index))
        pairs = construct_pairs(rows, "representation_train")
        self.assertEqual(sum(row["same_relation"] for row in pairs),
                         sum(not row["same_relation"] for row in pairs))
        by_id = {row["instance_id"]: row for row in rows}
        for pair in pairs:
            left, right = by_id[pair["left_instance_id"]], by_id[pair["right_instance_id"]]
            self.assertEqual(pair["same_relation"],
                             int(left["relation_id"] == right["relation_id"]))

    def test_builder_does_not_embed_relation_descriptions_in_instances(self):
        source = Path(__file__).parents[1] / "tasks" / "build_docred_relation_space.py"
        builder = source.read_text()
        self.assertNotIn('"relation_description": relation_info[relation]', builder)

    def test_overlapping_mentions_are_excluded_without_aborting_build(self):
        doc = document()
        doc["vertexSet"][1][0]["pos"] = [0, 1]
        rows, exclusions = raw_candidates(
            [doc], [], {"representation_train": ["P1"],
                        "relation_development": [], "relation_source_test": []},
            {"P1": "example"})
        self.assertEqual(rows, [])
        self.assertEqual(exclusions[0]["reason"], "invalid_masked_evidence")


if __name__ == "__main__":
    unittest.main()

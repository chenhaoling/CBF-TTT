import unittest

from tasks.extract_docred_relation_space_embeddings import (
    description_prompt, partition_items, prepare_items, statement_prompt,
)


class DocredRelationEmbeddingTest(unittest.TestCase):
    def manifest(self):
        return {"partitions": {
            "representation_train": ["P1", "P2"],
            "relation_development": ["P3"],
            "relation_source_test": ["P4"],
        }}

    def test_prompts_are_frozen(self):
        self.assertEqual(
            statement_prompt("[SUBJECT] founded [VALUE]."),
            "Identify the semantic relation between [SUBJECT] and [VALUE].\n"
            "Evidence: [SUBJECT] founded [VALUE].\nRelation representation:")
        self.assertEqual(description_prompt("creator"),
                         "Relation description: creator\nRelation representation:")

    def test_only_train_and_development_statements_are_prepared(self):
        rows = [
            {"instance_id": "a", "partition": "representation_train",
             "relation_id": "P1", "surface": "[SUBJECT] a [VALUE]"},
            {"instance_id": "b", "partition": "relation_development",
             "relation_id": "P3", "surface": "[SUBJECT] b [VALUE]"},
            {"instance_id": "c", "partition": "relation_source_test",
             "relation_id": "P4", "surface": "[SUBJECT] c [VALUE]"},
        ]
        descriptions = [{"relation_id": "P1", "description": "one"},
                        {"relation_id": "P2", "description": "two"}]
        items = prepare_items(rows, descriptions, self.manifest())
        self.assertEqual([row["item_id"] for row in items],
                         ["a", "b", "description:P1", "description:P2"])
        self.assertNotIn("relation_source_test", {row["partition"] for row in items})

    def test_shards_are_disjoint_and_complete(self):
        items = [{"item_id": str(index)} for index in range(7)]
        left = partition_items(items, 0, 2)
        right = partition_items(items, 1, 2)
        self.assertEqual([index for index, _ in left], [0, 2, 4, 6])
        self.assertEqual([index for index, _ in right], [1, 3, 5])


if __name__ == "__main__":
    unittest.main()

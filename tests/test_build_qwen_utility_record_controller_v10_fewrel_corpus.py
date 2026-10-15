import unittest

from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    choose_materials, filter_rows, source_pattern,
)


def synthetic_row(relation, index):
    return {
        "relation_id": relation, "source_index": index,
        "source_row_sha256": f"{relation}-{index}",
        "source_fact_hash": f"{relation}-{index}",
        "head_entity_id": f"h-{relation}-{index}",
        "tail_entity_id": f"t-{relation}-{index}",
        "entity_pair": [f"h-{relation}-{index}", f"t-{relation}-{index}"],
        "subject": f"subject {relation} {index}",
        "subject_norm": f"subject {relation} {index}",
        "value": f"value {relation} {index}",
        "value_norm": f"value {relation} {index}",
        "pattern": f"[X] relation {relation} template {index % 9} [Y]",
        "pattern_norm": f"[x] relation {relation} template {index % 9} [y]",
        "statement": f"subject {relation} {index} relation {relation} value {relation} {index}",
    }


class V10FewRelCorpusTest(unittest.TestCase):
    def test_source_pattern_requires_single_entity_mentions(self):
        row = {
            "tokens": ["Alice", "met", "Bob"],
            "h": ["Alice", "Q1", [[0]]],
            "t": ["Bob", "Q2", [[2]]],
        }
        self.assertEqual(source_pattern(row), "[X] met [Y]")
        row["tokens"].append("Alice")
        row["h"][2].append([3])
        with self.assertRaises(ValueError):
            source_pattern(row)

    def test_development_overlap_precedence_and_deduplication(self):
        rows = [synthetic_row("P1", index) for index in range(4)]
        rows[1]["source_row_sha256"] = rows[0]["source_row_sha256"]
        rows[2]["statement"] = rows[0]["statement"]
        rows[3]["pattern_norm"] = "sealed template"
        kept, removed = filter_rows(rows, {
            "source_rows": set(), "statements": set(),
            "templates": {"sealed template"},
        })
        self.assertEqual(len(kept), 1)
        self.assertEqual([row["reason"] for row in removed], [
            "duplicate_source_row", "duplicate_relation_statement",
            "development_masked_template_overlap",
        ])

    def test_material_allocator_never_reuses_rows_or_entity_pairs(self):
        relations = [f"P{index}" for index in range(6)]
        rows = [synthetic_row(relation, index)
                for relation in relations for index in range(60)]
        materials = choose_materials(
            rows, relations, mechanisms=("semantic_overwrite",), repetitions=1,
        )
        self.assertEqual(len(materials), len(relations))
        hashes = [value for row in materials
                  for value in row["material_source_row_sha256"]]
        pairs = [tuple(value) for row in materials
                 for value in row["material_entity_pairs"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(len(patterns) == 4 for row in materials
                            for patterns in row["patterns"].values()))


if __name__ == "__main__":
    unittest.main()

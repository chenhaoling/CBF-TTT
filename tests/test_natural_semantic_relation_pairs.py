import unittest

from tasks.build_natural_semantic_relation_pairs import (
    construct,
    normalize,
    trigram_jaccard,
)


def row(fold, relation, index, surfaces=2):
    subject = f"subject-{fold}-{relation}-{index}"
    values = [f"[SUBJECT] relation {relation} phrasing {value}"
              for value in range(surfaces)]
    return {
        "outer_fold": fold,
        "relation_id": f"r{fold}-{relation}",
        "row_id": f"row-{fold}-{relation}-{index}",
        "source": "mquake" if index % 7 == 0 else "counterfact",
        "source_row_sha256": f"{fold:01x}{relation:02x}{index:061x}"[-64:],
        "subject_normalized": subject,
        "masked_surfaces": values,
    }


class NaturalSemanticRelationPairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [row(fold, relation, index) for fold in range(5)
                    for relation in range(4) for index in range(8)]

    def test_construct_is_deterministic_and_relation_disjoint(self):
        first = construct(self.rows, cap=8)
        second = construct(list(reversed(self.rows)), cap=8)
        self.assertEqual(first, second)
        pairs, surfaces, _ = first
        self.assertTrue(pairs)
        self.assertEqual(len({item["surface_id"] for item in surfaces}), len(surfaces))
        for item in pairs:
            self.assertEqual(item["same_relation"],
                             int(item["anchor_relation_id"] == item["other_relation_id"]))
            self.assertEqual(item["locality_risk"], 1 - item["same_relation"])

    def test_hard_negative_is_at_least_as_lexical_as_random(self):
        pairs, _, _ = construct(self.rows, cap=8)
        grouped = {}
        for item in pairs:
            grouped.setdefault(item["anchor_source_row_sha256"], {})[item["kind"]] = item
        for values in grouped.values():
            hard, random = values["lexical_hard_negative"], values["random_negative"]
            self.assertGreaterEqual(hard["character_trigram_jaccard"],
                                    random["character_trigram_jaccard"])

    def test_relation_cap_is_published(self):
        pairs, _, excluded = construct(self.rows, cap=4)
        self.assertTrue(excluded)
        self.assertTrue(all(item["reason"] == "relation_cap" for item in excluded))
        self.assertTrue(pairs)

    def test_normalization_and_similarity(self):
        self.assertEqual(normalize("  Foo   BAR "), "foo bar")
        self.assertEqual(trigram_jaccard("same", "same"), 1.0)
        self.assertLess(trigram_jaccard("alpha", "omega"), 1.0)


if __name__ == "__main__":
    unittest.main()

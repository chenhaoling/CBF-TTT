import unittest

from scripts.inspect_templama_utility_source import (
    choose_owner,
    extract_subject,
    singleton_answer,
    template_regex,
)


class TestInspectTempLamaUtilitySource(unittest.TestCase):
    def test_extracts_subject_from_subject_first_template(self):
        template = "<subject> plays for <object>."
        query = "Valentino Rossi plays for _X_."
        self.assertEqual(extract_subject(query, template), "Valentino Rossi")

    def test_extracts_subject_from_object_first_template(self):
        template = "<object> is the head coach of <subject>."
        query = "_X_ is the head coach of FC Example."
        self.assertEqual(extract_subject(query, template), "FC Example")
        self.assertIsNotNone(template_regex(template).fullmatch(query))

    def test_singleton_answer_rejects_ambiguity(self):
        row = {"answer": [{"wikidata_id": "Q1", "name": "Alpha"}]}
        self.assertEqual(singleton_answer(row), ("Q1", "alpha"))
        row["answer"].append({"wikidata_id": "Q2", "name": "Beta"})
        self.assertIsNone(singleton_answer(row))

    def test_overlap_owner_is_deterministic_and_observed(self):
        owners = {"train", "test"}
        selected = choose_owner("Q123", owners)
        self.assertIn(selected, owners)
        self.assertEqual(selected, choose_owner("Q123", set(reversed(sorted(owners)))))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from tasks.build_qwen_discrete_cohort_ood import (
    BOUNDARIES,
    PROTOCOL,
    SUITES,
    generate_group,
    load_corpus,
    materialize,
    parse_visible_block,
    replay,
    validate,
)


class QwenDiscreteCohortOodCorpusTest(unittest.TestCase):
    def test_all_suites_validate_and_are_causal(self):
        pairs = [
            pair
            for suite in SUITES
            for group in range(2)
            for pair in generate_group(202610154, suite, group)
        ]
        audit = validate(pairs)
        self.assertEqual(audit["groups"], 8)
        self.assertEqual(audit["episodes"], 16)
        self.assertTrue(audit["causal_answers_verified"])
        for inp, target in pairs:
            self.assertEqual(inp["protocol"], PROTOCOL)
            self.assertEqual(len(target["answers"]), 3 * len(BOUNDARIES[inp["split"]]))

    def test_duplicate_suite_contains_semantic_noops(self):
        inp, target = generate_group(202610154, "duplicate_noop", 0)[0]
        events = []
        states = {}
        for block, current in zip(inp["blocks"], target["semantic_events"]):
            before = replay(events)
            events.extend(current)
            after = replay(events)
            states[block["index"]] = (before, after, current)
        # At block 6 and 12, removing record 2 does not change visible state
        # before the new block assignment is applied.
        for block in (6, 12):
            before, _, current = states[block]
            prefix = []
            for prior in target["semantic_events"][: block - 1]:
                prefix.extend(prior)
            prelude_only = replay(prefix + current[:-1])
            self.assertEqual(before, prelude_only)

    def test_visible_parser_recovers_only_rendered_state(self):
        inp, target = generate_group(202610154, "mixed_lifecycle", 0)[0]
        for block, events in zip(inp["blocks"], target["semantic_events"]):
            parsed = parse_visible_block(block["text"], [{"private": "ignored"}])
            self.assertEqual(parsed["batch"], events[-1]["batch"])
            self.assertEqual(parsed["facts"], events[-1]["facts"])
            self.assertEqual(
                parsed["old_refs"],
                [event["batch"] for event in events[:-1] if event["op"] == "revoke"],
            )

    def test_materialize_is_deterministic_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            manifest = materialize(202610154, 2, root)
            self.assertEqual(manifest["groups_per_suite"], 2)
            self.assertEqual(set(manifest["suites"]), set(SUITES))
            self.assertEqual(len(manifest["files_sha256"]), 2 * len(SUITES))
            for suite in SUITES:
                self.assertEqual(len((root / f"{suite}.inputs.jsonl").read_text().splitlines()), 4)
            loaded_manifest, suites = load_corpus(root)
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual({key: len(value) for key, value in suites.items()},
                             {suite: 4 for suite in SUITES})


if __name__ == "__main__":
    unittest.main()

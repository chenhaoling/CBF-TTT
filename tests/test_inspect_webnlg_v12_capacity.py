"""Focused tests for WebNLG V12 source parsing utilities."""

import unittest
import xml.etree.ElementTree as ET

from scripts.inspect_webnlg_v12_capacity import normalize, slot_template, split_triple


class WebNLGCapacityTest(unittest.TestCase):
    def test_triple_split(self):
        self.assertEqual(split_triple("A | relation | B"), ["A", "relation", "B"])

    def test_aligned_template(self):
        entry = ET.fromstring("""<entry><entitymap>
          <entity>AGENT-1 | Alpha</entity><entity>PATIENT-1 | Beta</entity>
          </entitymap><lex><template>AGENT-1 was born in PATIENT-1.</template></lex></entry>""")
        self.assertEqual(slot_template(entry, entry.find("lex"), "Alpha", "Beta"),
                         "[X] was born in [Y].")

    def test_normalization(self):
        self.assertEqual(normalize("New_York"), "new york")


if __name__ == "__main__":
    unittest.main()

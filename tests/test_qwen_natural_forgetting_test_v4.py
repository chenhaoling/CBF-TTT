"""Release-boundary tests for the unopened V4 test corpus."""

import unittest

from tasks.build_qwen_natural_forgetting_test_v4 import release_allowed
from tasks.evaluate_qwen_natural_forgetting_consensus_v4 import PROTOCOL


class TestReleaseV4Tests(unittest.TestCase):
    def released(self):
        return {"protocol": PROTOCOL, "gate": {"test_authorized": True},
                "selected": {"candidate": "n_unanimity"},
                "development_loaded": False, "test_loaded": False,
                "external_benchmarks_loaded": False}

    def test_complete_release_is_required(self):
        status = "completed_qwen_natural_forgetting_consensus_v4_test_release"
        self.assertTrue(release_allowed(self.released(), {"passed": True}, status, "0"))
        failed = self.released(); failed["gate"]["test_authorized"] = False
        self.assertFalse(release_allowed(failed, {"passed": True}, status, "0"))
        self.assertFalse(release_allowed(self.released(), {"passed": False}, status, "0"))
        self.assertFalse(release_allowed(self.released(), {"passed": True}, status, "1"))


if __name__ == "__main__":
    unittest.main()

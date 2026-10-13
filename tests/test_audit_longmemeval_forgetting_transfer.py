import unittest

from scripts.audit_longmemeval_forgetting_transfer import bootstrap


class AuditTransferTest(unittest.TestCase):
    def test_bootstrap_direction(self):
        rows = [{"policy_metrics": {"a": {"m": 1.}, "b": {"m": 0.}}}
                for _ in range(4)]
        self.assertEqual(bootstrap(rows, "a", "b", "m", 1, draws=100), [1., 1.])
        self.assertEqual(bootstrap(rows, "a", "b", "m", 1, -1., draws=100), [-1., -1.])


if __name__ == "__main__": unittest.main()

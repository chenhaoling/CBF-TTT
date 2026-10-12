"""Pure tests for V8 gate and tensor construction."""

import unittest

from tasks.train_text_only_scope_calibrator_v8 import development_gate


class TextOnlyScopeCalibratorV8Tests(unittest.TestCase):
    def test_gate_requires_gain_over_v7(self):
        result={"top1_accuracy":.96,"mrr":.98}
        v4={"top1_gain":.13,"bootstrap_95ci":[.10,.16]}
        v7={"top1_gain":.02,"bootstrap_95ci":[.005,.035]}
        self.assertTrue(development_gate(result,v4,v7)["passed"])
        v7["bootstrap_95ci"][0]=-.001
        self.assertFalse(development_gate(result,v4,v7)["passed"])


if __name__ == "__main__": unittest.main()


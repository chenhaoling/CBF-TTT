"""Pure gate tests for final CounterFact entity fusion."""
import unittest
from tasks.train_entity_fusion_scope_v10 import development_gate

class EntityFusionScopeV10Tests(unittest.TestCase):
    def test_gate(self):
        result={"top1_accuracy":.96,"mrr":.98}
        v8={"top1_gain":.02,"bootstrap_95ci":[.005,.04]}
        v4={"top1_gain":.14,"bootstrap_95ci":[.11,.17]}
        self.assertTrue(development_gate(result,v8,v4)["passed"])
        v8["top1_gain"]=.009
        self.assertFalse(development_gate(result,v8,v4)["passed"])

if __name__=="__main__":unittest.main()


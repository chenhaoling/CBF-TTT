import copy
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_configs import (
    CONFIG_NAMES,
    CONFIG_ROOT,
    audit,
    validate_config,
)
from scripts.audit_in_place_ttt_level_b_configs import read_config


class TwoRTX5090ConfigAuditTest(unittest.TestCase):
    def test_frozen_repository_configs_pass(self):
        result = audit()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(result["total_tokens"], 98_566_144)

    def test_disabling_activation_offload_fails(self):
        config = read_config(CONFIG_ROOT / CONFIG_NAMES[(1, False)])
        changed = copy.deepcopy(config)
        changed["train"]["enable_activation_offload"] = False
        errors = validate_config(changed, 1, False)
        self.assertTrue(any("activation offload" in error for error in errors))

    def test_unmatched_arm_change_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, name in CONFIG_NAMES.items():
                text = (CONFIG_ROOT / name).read_text(encoding="utf-8")
                if key == (2, True):
                    text = text.replace("  weight_decay: 0.1", "  weight_decay: 0.2")
                (root / name).write_text(text, encoding="utf-8")
            result = audit(root)
            self.assertFalse(result["passed"])
            self.assertTrue(any("stage 2 arms differ" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()

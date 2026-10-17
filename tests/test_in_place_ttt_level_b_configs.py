import copy
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_level_b_configs import (
    CONFIG_NAMES,
    CONFIG_ROOT,
    audit,
    read_config,
    validate_config,
)


class LevelBConfigAuditTest(unittest.TestCase):
    def test_frozen_repository_configs_pass(self):
        result = audit()
        self.assertTrue(result["passed"], result["errors"])

    def test_unregistered_optimizer_change_fails(self):
        config = read_config(CONFIG_ROOT / CONFIG_NAMES[(1, True)])
        changed = copy.deepcopy(config)
        changed["train"]["lr"] = 1e-5
        self.assertTrue(validate_config(changed, 1, True))

    def test_pair_mismatch_is_detected(self):
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

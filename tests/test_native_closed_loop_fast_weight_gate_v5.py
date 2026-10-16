import json
from pathlib import Path
import tempfile
import unittest

from tasks.native_closed_loop_fast_weight_gate_v5 import (
    LAYERS,
    PROTOCOL,
    canonical,
    policy_specs,
    summarize,
)


def outcome(J, accuracy=.8, nll=.5, eos=.1):
    return {
        "J": J, "protected_accuracy": accuracy, "protected_nll": nll,
        "eos_rate": eos, "accuracy": accuracy,
    }


class NativeClosedLoopGateV5Test(unittest.TestCase):
    def test_registered_policy_matrix(self):
        specs = policy_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual([row["policy"] for row in specs[:2]], ["always_write", "never_write"])
        self.assertEqual(sum(row["kind"] == "controller" for row in specs), 6)

    def test_smoke_gate_uses_sequential_policy_states(self):
        values = {
            "always_write": 2.0, "never_write": 1.0,
            "full_seed2026101801": .90, "no_support_seed2026101801": .95,
            "full_seed2026101802": .91, "no_support_seed2026101802": .93,
            "full_seed2026101803": .92, "no_support_seed2026101803": .91,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); labels = root / "labels.jsonl"; decisions = root / "decisions.jsonl"
            with labels.open("w") as label_sink, decisions.open("w") as decision_sink:
                for group_index in range(12):
                    group = f"g{group_index}"
                    for spec in policy_specs():
                        policy = spec["policy"]; parent = f"initial-{group}"
                        for boundary in range(3):
                            action = ("admit_all" if policy == "always_write" else
                                      "reject_all" if policy == "never_write" else "admit_layer_0")
                            decision = {
                                "protocol": PROTOCOL, "stage": "smoke", "group_id": group,
                                "boundary": boundary, "policy": policy, "kind": spec["kind"],
                                "parent_fast_sha256": parent, "online_memory": {}, "action": action,
                            }
                            reject = {str(layer): f"r-{group}-{policy}-{boundary}-{layer}" for layer in LAYERS}
                            admit = {str(layer): f"a-{group}-{policy}-{boundary}-{layer}" for layer in LAYERS}
                            if action == "admit_all":
                                sources = {str(layer): "admit" for layer in LAYERS}
                            elif action == "reject_all":
                                sources = {str(layer): "reject" for layer in LAYERS}
                            else:
                                sources = {str(layer): ("admit" if layer == 0 else "reject") for layer in LAYERS}
                            selected = {str(layer): {"reject": reject, "admit": admit}[sources[str(layer)]][str(layer)]
                                        for layer in LAYERS}
                            next_parent = f"next-{group}-{policy}-{boundary}"
                            label = {
                                "protocol": PROTOCOL, "stage": "smoke", "group_id": group,
                                "boundary": boundary, "policy": policy,
                                "decision_sha256": canonical(decision), "action": action,
                                "branch_audit": {
                                    "parent_fast_sha256": parent, "reject_fast_sha256": parent,
                                },
                                "base_layer_sha256": {"reject": reject, "admit": admit},
                                "composition": {"sources": sources, "layer_sha256": selected},
                                "next_fast_sha256": next_parent, "outcome": outcome(values[policy]),
                                "resource": {"seconds": 1.0, "peak_allocated_gib": 1.0,
                                             "peak_reserved_gib": 1.0},
                            }
                            decision_sink.write(json.dumps(decision) + "\n")
                            label_sink.write(json.dumps(label) + "\n")
                            parent = next_parent
            result = summarize([labels], [decisions], root / "summary.json")
            self.assertTrue(result["checks"]["trajectory_continuity"])
            self.assertTrue(result["checks"]["at_least_two_full_better_than_no_support"])
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()

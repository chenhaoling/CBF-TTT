from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_stage1_ladder import (
    EIGHT_K_DATA_SHA256,
    audit_final,
    audit_prerequisite,
    sha256,
)


def make_child(root: Path, length: int, arm: str, data_sha: str,
               manifest_sha: str | None = None, input_audit_sha: str | None = None):
    root.mkdir(parents=True)
    (root / "train.log").write_text("loss: 2.0, grad_norm: 3.0\n")
    (root / "gpu.csv").write_text(
        "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n"
        "1,0,30000,32607,90\n1,1,30000,32607,90\n"
    )
    (root / "host.csv").write_text(
        "timestamp,memory_total_mib,memory_used_mib,memory_available_mib,"
        "swap_total_mib,swap_used_mib\n1,62000,46000,16000,4096,512\n"
    )
    (root / "exit_code.txt").write_text("0\n")
    checkpoint = root / "checkpoint" / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("metadata")
    with (checkpoint / "rank.distcp").open("wb") as handle:
        handle.seek(40_000_000_000 - 1)
        handle.write(b"0")
    checkpoint_bytes = sum(
        path.stat().st_size for path in (root / "checkpoint").rglob("*")
        if path.is_file()
    )
    metadata = {
        "protocol": "in_place_ttt_baseline_resource_v2",
        "hostname": "user-Super-Server",
        "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
        "upstream_clean": True,
        "resource_profile": "activation_offload",
        "activation_offload": True,
        "sequence_length": length,
        "global_batch_size": 2,
        "arm": arm,
        "data_sha256": data_sha,
    }
    if manifest_sha is not None:
        metadata.update({
            "data_manifest_sha256": manifest_sha,
            "data_audit_sha256": input_audit_sha,
        })
    (root / "run_metadata.json").write_text(json.dumps(metadata))
    dependencies = {
        "train_log_sha256": sha256(root / "train.log"),
        "gpu_monitor_sha256": sha256(root / "gpu.csv"),
        "host_monitor_sha256": sha256(root / "host.csv"),
    }
    if manifest_sha is not None:
        dependencies.update({
            "data_manifest_sha256": manifest_sha,
            "data_audit_sha256": input_audit_sha,
        })
    summary = {
        "completed": True, "activation_offload_gate_passed": True,
        "exit_code": 0, "sequence_length": length, "arm": arm,
        "last_loss": 2.0, "last_gradient_norm": 3.0,
        "optimizer_step_seconds": 4.0,
        "steady_tokens_per_second": length * 2 / 4,
        "minimum_gpu_free_fraction": {"0": 0.08, "1": 0.08},
        "peak_gpu_memory_mib": {"0": 30000, "1": 30000},
        "host_resources": {"minimum_memory_available_mib": 16000,
                           "peak_swap_used_mib": 512},
        "checkpoint_bytes": checkpoint_bytes,
        "dependencies": dependencies,
    }
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "independent_audit.json").write_text(json.dumps({
        "protocol": "in_place_ttt_2x5090_resource_audit_v1",
        "passed": True, "sequence_length": length, "arm": arm,
    }))


class TwoRTX5090Stage1LadderAuditTest(unittest.TestCase):
    def test_incomplete_8k_prerequisite_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_prerequisite(Path(temporary))
            self.assertFalse(result["passed"])
            self.assertTrue(any("not completed" in item for item in result["errors"]))

    def test_complete_ladder_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            eight = base / "eight"
            eight.mkdir()
            (eight / "state.txt").write_text("completed_8k_both_arms\n")
            for arm in ("no_ttt", "ttt"):
                make_child(eight / f"resource_8k_{arm}", 8192, arm,
                           EIGHT_K_DATA_SHA256)
            prerequisite = audit_prerequisite(eight)
            self.assertTrue(prerequisite["passed"], prerequisite["errors"])

            inputs = base / "inputs"
            inputs.mkdir()
            outputs = {}
            for length in (16384, 32768):
                data = inputs / f"resource_{length}.jsonl"
                data.write_text(f"data-{length}\n")
                outputs[str(length)] = {
                    "records": 2, "training_tokens": 2 * length,
                    "artifact": {"path": data.name,
                                 "sha256": sha256(data),
                                 "size_bytes": data.stat().st_size},
                }
            manifest = inputs / "manifest.json"
            manifest.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_resource_inputs_v1",
                "outputs": outputs,
            }))
            manifest_sha = sha256(manifest)
            input_audit = inputs / "independent_audit.json"
            input_audit.write_text(json.dumps({
                "passed": True, "bindings": {"manifest_sha256": manifest_sha},
                "outputs": {
                    length: {"artifact_sha256": record["artifact"]["sha256"]}
                    for length, record in outputs.items()
                },
            }))
            input_audit_sha = sha256(input_audit)

            root = base / "ladder"
            root.mkdir()
            (root / "state.txt").write_text("all_children_completed\n")
            (root / "config_audit.json").write_text(json.dumps({"passed": True}))
            (root / "eight_k_prerequisite_audit.json").write_text(
                json.dumps(prerequisite, indent=2, sort_keys=True) + "\n"
            )
            for length in (16384, 32768):
                for arm in ("no_ttt", "ttt"):
                    make_child(
                        root / f"resource_{length}_{arm}", length, arm,
                        outputs[str(length)]["artifact"]["sha256"],
                        manifest_sha, input_audit_sha,
                    )
            execution_addendum = base / "execution.md"
            input_addendum = base / "input.md"
            runner = base / "runner.sh"
            execution_addendum.write_text("execution\n")
            input_addendum.write_text("input\n")
            runner.write_text("runner\n")
            metadata = {
                "protocol": "in_place_ttt_2x5090_stage1_ladder_v1",
                "hostname": "user-Super-Server",
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True,
                "tool_commit": "a" * 40,
                "tool_clean": True,
                "eight_k_root": str(eight.resolve()),
                "execution_addendum_sha256": sha256(execution_addendum),
                "input_addendum_sha256": sha256(input_addendum),
                "runner_sha256": sha256(runner),
                "auditor_sha256": sha256(Path(
                    "scripts/audit_in_place_ttt_2x5090_stage1_ladder.py"
                ).resolve()),
                "data_manifest_sha256": manifest_sha,
                "data_audit_sha256": input_audit_sha,
                "config_audit_sha256": sha256(root / "config_audit.json"),
                "eight_k_prerequisite_audit_sha256":
                    sha256(root / "eight_k_prerequisite_audit.json"),
            }
            (root / "ladder_metadata.json").write_text(json.dumps(metadata))
            result = audit_final(
                root, eight, manifest, input_audit,
                execution_addendum, input_addendum, runner,
            )
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(len(result["boundaries"]), 4)
            (root / "state.txt").write_text("completed_stage1_ladder\n")
            published = audit_final(
                root, eight, manifest, input_audit,
                execution_addendum, input_addendum, runner,
            )
            self.assertTrue(published["passed"], published["errors"])


if __name__ == "__main__":
    unittest.main()

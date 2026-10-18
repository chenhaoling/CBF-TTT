from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_stage2_ladder import (
    audit,
    sha256,
    validate_start_model,
)
from tests.test_in_place_ttt_2x5090_stage1_ladder import make_child


class TwoRTX5090Stage2LadderTest(unittest.TestCase):
    def test_stale_stage1_hf_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hf = root / "no_ttt/model/hf_ckpt"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text(json.dumps({
                "ttt_mode": False, "max_position_embeddings": 32768,
            }))
            audit_data = {"arms": {"no_ttt": {
                "final_hf_relative_path": "model/hf_ckpt",
                "hf_manifest": {},
            }}}
            errors, _ = validate_start_model(root, audit_data, "no_ttt")
            self.assertTrue(errors)

    def test_complete_arm_specific_ladder_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage1 = base / "stage1"
            stage1.mkdir()
            (stage1 / "state.txt").write_text("completed_100m_stage1\n")
            stage1_arms = {}
            for arm in ("no_ttt", "ttt"):
                hf = stage1 / arm / "training/final/hf_ckpt"
                hf.mkdir(parents=True)
                (hf / "config.json").write_text(json.dumps({
                    "ttt_mode": arm == "ttt",
                    "max_position_embeddings": 32768,
                }))
                (hf / "model.safetensors").write_bytes(arm.encode())
                stage1_arms[arm] = {
                    "final_hf_relative_path": "training/final/hf_ckpt",
                    "hf_manifest": {
                        str(path.relative_to(hf)): {
                            "sha256": sha256(path), "size_bytes": path.stat().st_size,
                        }
                        for path in hf.rglob("*") if path.is_file()
                    },
                }
            stage1_audit = stage1 / "independent_audit.json"
            stage1_audit.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_100m_stage1_audit_v1",
                "passed": True, "arms": stage1_arms,
            }))

            inputs = base / "inputs"
            inputs.mkdir()
            outputs = {}
            audit_outputs = {}
            for length in (65536, 131072):
                data = inputs / f"resource_{length}.jsonl"
                data.write_text(f"fixture-{length}\n")
                digest = sha256(data)
                outputs[str(length)] = {
                    "records": 2, "training_tokens": 2 * length,
                    "artifact": {"path": data.name, "sha256": digest,
                                 "size_bytes": data.stat().st_size},
                }
                audit_outputs[str(length)] = {"artifact_sha256": digest}
            manifest = inputs / "manifest.json"
            manifest.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_resource_inputs_v1",
                "outputs": outputs,
            }))
            manifest_sha = sha256(manifest)
            input_audit = inputs / "independent_audit.json"
            input_audit.write_text(json.dumps({
                "passed": True,
                "bindings": {"manifest_sha256": manifest_sha},
                "outputs": audit_outputs,
            }))
            input_audit_sha = sha256(input_audit)

            root = base / "ladder"
            root.mkdir()
            (root / "state.txt").write_text("all_children_completed\n")
            (root / "config_audit.json").write_text(json.dumps({"passed": True}))
            configs = {arm: base / f"stage2_{arm}.yaml" for arm in ("no_ttt", "ttt")}
            for arm, path in configs.items():
                path.write_text(arm + "\n")
            start_models = {}
            for arm in ("no_ttt", "ttt"):
                errors, record = validate_start_model(
                    stage1, json.loads(stage1_audit.read_text()), arm
                )
                self.assertEqual(errors, [])
                start_models[arm] = record
            (root / "start_models.json").write_text(
                json.dumps(start_models, indent=2, sort_keys=True) + "\n"
            )
            for length in (65536, 131072):
                for arm in ("no_ttt", "ttt"):
                    child = root / f"resource_{length}_{arm}"
                    make_child(
                        child, length, arm,
                        outputs[str(length)]["artifact"]["sha256"],
                        manifest_sha, input_audit_sha,
                    )
                    metadata_path = child / "run_metadata.json"
                    metadata = json.loads(metadata_path.read_text())
                    metadata.update({
                        "model": start_models[arm]["root"],
                        "model_config_sha256": start_models[arm]["config_sha256"],
                        "config_sha256": sha256(configs[arm]),
                    })
                    metadata_path.write_text(json.dumps(metadata))

            resource_plan, addendum = base / "resource.md", base / "execution.md"
            input_addendum, runner = base / "input.md", base / "runner.sh"
            for path, value in (
                (resource_plan, "resource\n"), (addendum, "execution\n"),
                (input_addendum, "input\n"), (runner, "runner\n"),
            ):
                path.write_text(value)
            metadata = {
                "protocol": "in_place_ttt_2x5090_stage2_ladder_v1",
                "hostname": "user-Super-Server",
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True, "tool_clean": True,
                "tool_commit": "a" * 40,
                "stage1_root": str(stage1.resolve()),
                "stage1_audit_sha256": sha256(stage1_audit),
                "resource_plan_sha256": sha256(resource_plan),
                "execution_addendum_sha256": sha256(addendum),
                "input_addendum_sha256": sha256(input_addendum),
                "data_manifest_sha256": manifest_sha,
                "data_audit_sha256": input_audit_sha,
                "no_ttt_config_sha256": sha256(configs["no_ttt"]),
                "ttt_config_sha256": sha256(configs["ttt"]),
                "config_audit_sha256": sha256(root / "config_audit.json"),
                "start_models_sha256": sha256(root / "start_models.json"),
                "runner_sha256": sha256(runner),
                "auditor_sha256": sha256(Path(
                    "scripts/audit_in_place_ttt_2x5090_stage2_ladder.py"
                ).resolve()),
            }
            (root / "ladder_metadata.json").write_text(json.dumps(metadata))
            result = audit(
                root, stage1_root=stage1, resource_plan=resource_plan,
                execution_addendum=addendum, input_addendum=input_addendum,
                data_manifest_path=manifest, data_audit_path=input_audit,
                configs=configs, runner=runner,
            )
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(len(result["boundaries"]), 4)


if __name__ == "__main__":
    unittest.main()

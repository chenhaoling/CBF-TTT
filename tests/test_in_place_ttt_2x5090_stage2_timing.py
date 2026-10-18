from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_stage2_ladder import validate_start_model
from scripts.audit_in_place_ttt_2x5090_stage2_timing import (
    GLOBAL_BATCH,
    SEQ_LEN,
    STEPS,
    TOKENS_PER_STEP,
    audit,
    audit_arm,
    sha256,
)
from tests.test_in_place_ttt_2x5090_stage1_timing import make_arm


class TwoRTX5090Stage2TimingTest(unittest.TestCase):
    def test_checkpoint_in_timing_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "no_ttt"
            make_arm(root)
            checkpoint = root / "training_output/global_step_3/.metadata"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("unexpected\n")
            errors, record = audit_arm(root, "no_ttt")
            self.assertTrue(any("checkpoint" in item for item in errors))
            self.assertEqual(record["checkpoint_files"], [
                "training_output/global_step_3/.metadata"
            ])

    def test_complete_arm_specific_timing_fixture_passes(self):
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
                            "sha256": sha256(path),
                            "size_bytes": path.stat().st_size,
                        }
                        for path in hf.rglob("*") if path.is_file()
                    },
                }
            stage1_audit = stage1 / "independent_audit.json"
            stage1_audit.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_100m_stage1_audit_v1",
                "passed": True,
                "arms": stage1_arms,
            }))

            stage1_data = json.loads(stage1_audit.read_text())
            start_models = {}
            for arm in ("no_ttt", "ttt"):
                errors, record = validate_start_model(stage1, stage1_data, arm)
                self.assertEqual(errors, [])
                start_models[arm] = record

            ladder = base / "stage2_ladder"
            ladder.mkdir()
            (ladder / "state.txt").write_text("completed_stage2_ladder\n")
            ladder_audit = ladder / "independent_ladder_audit.json"
            ladder_audit.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_stage2_ladder_audit_v1",
                "passed": True,
                "start_models": start_models,
            }))

            root = base / "timing"
            root.mkdir()
            (root / "config_audit.json").write_text(json.dumps({"passed": True}))
            (root / "start_models.json").write_text(
                json.dumps(start_models, indent=2, sort_keys=True) + "\n"
            )
            for arm in ("no_ttt", "ttt"):
                make_arm(root / arm)

            data = base / "stage2_128k.jsonl"
            data.write_text('{"content_split":"fixture"}\n')
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({
                "protocol": "in_place_ttt_public_corpus_v1",
                "profile": "full",
                "stages": [{
                    "name": "stage2_128k",
                    "artifacts": {data.name: {
                        "sha256": sha256(data),
                        "size_bytes": data.stat().st_size,
                    }},
                }],
            }))
            corpus_audit = base / "corpus_audit.json"
            corpus_audit.write_text(json.dumps({
                "passed": True,
                "profile": "full",
                "bindings": {"manifest_sha256": sha256(manifest)},
            }))
            timing_plan = base / "timing_plan.md"
            addendum = base / "execution.md"
            runner = base / "runner.sh"
            configs = {arm: base / f"{arm}.yaml" for arm in ("no_ttt", "ttt")}
            for path, content in (
                (timing_plan, "timing\n"),
                (addendum, "execution\n"),
                (runner, "runner\n"),
            ):
                path.write_text(content)
            for arm, path in configs.items():
                path.write_text(arm + "\n")

            metadata = {
                "protocol": "in_place_ttt_2x5090_stage2_timing_v1",
                "hostname": "user-Super-Server",
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True,
                "tool_commit": "a" * 40,
                "tool_clean": True,
                "stage1_root": str(stage1.resolve()),
                "stage2_ladder_root": str(ladder.resolve()),
                "steps": STEPS,
                "sequence_length": SEQ_LEN,
                "global_batch_size": GLOBAL_BATCH,
                "tokens_per_arm": STEPS * TOKENS_PER_STEP,
                "timing_plan_sha256": sha256(timing_plan),
                "execution_addendum_sha256": sha256(addendum),
                "stage1_audit_sha256": sha256(stage1_audit),
                "stage2_ladder_audit_sha256": sha256(ladder_audit),
                "corpus_manifest_sha256": sha256(manifest),
                "corpus_audit_sha256": sha256(corpus_audit),
                "data_sha256": sha256(data),
                "no_ttt_config_sha256": sha256(configs["no_ttt"]),
                "ttt_config_sha256": sha256(configs["ttt"]),
                "config_audit_sha256": sha256(root / "config_audit.json"),
                "start_models_sha256": sha256(root / "start_models.json"),
                "runner_sha256": sha256(runner),
                "auditor_sha256": sha256(Path(
                    "scripts/audit_in_place_ttt_2x5090_stage2_timing.py"
                ).resolve()),
            }
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            result = audit(
                root,
                stage1_root=stage1,
                stage2_ladder_root=ladder,
                timing_plan=timing_plan,
                execution_addendum=addendum,
                corpus_manifest=manifest,
                corpus_audit=corpus_audit,
                data=data,
                configs=configs,
                runner=runner,
            )
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["projected_sequential_stage2_seconds"], 75.0)


if __name__ == "__main__":
    unittest.main()

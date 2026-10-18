from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.audit_in_place_ttt_2x5090_100m_stage1 as formal


def training_log() -> str:
    settings = (
        "data_parallel_mode: fsdp2\n"
        "enable_activation_offload: true\n"
        "activation_gpu_limit: 0.0\n"
        "enable_fsdp_offload: false\n"
    )
    rows = []
    for step in range(1, formal.STEPS + 1):
        elapsed = step * 10
        rows.append(
            f"\rEpoch 1/1: 100%|####| {step}/27 [00:{elapsed:02d}<00:00, "
            f"10.00s/it, loss: {3.0 - step / 100:.4f}, "
            f"grad_norm: {4.0 + step / 100:.4f}, lr: 1e-6]"
        )
    rows.append(
        "\rEpoch 1/1: 100%|####| 27/27 [05:00<00:00, 11.11s/it, "
        "loss: 2.7300, grad_norm: 4.2700, lr: 1e-6]\n"
    )
    rows.append("Elapsed (wall clock) time (h:mm:ss or m:ss): 5:20.00\n")
    return settings + "".join(rows)


def make_arm(root: Path, arm: str) -> None:
    root.mkdir(parents=True)
    (root / "train.log").write_text(training_log())
    (root / "exit_code.txt").write_text("0\n")
    (root / "disk_before.txt").write_text("200000000000\n")
    (root / "gpu.csv").write_text(
        "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n"
        "1,0,30000,32607,90\n1,1,30000,32607,90\n"
    )
    (root / "host.csv").write_text(
        "timestamp,memory_total_mib,memory_used_mib,memory_available_mib,"
        "swap_total_mib,swap_used_mib\n1,62000,46000,16000,4096,512\n"
    )
    (root / "preflight_gpu.csv").write_text(
        "0, NVIDIA GeForce RTX 5090, 32607, 46\n"
        "1, NVIDIA GeForce RTX 5090, 32607, 18\n"
    )
    (root / "preflight_processes.csv").write_text("")
    (root / "preflight_conflicts.txt").write_text("")
    (root / "preflight_host.txt").write_text(
        "MemTotal:       65011712 kB\nMemAvailable:   55000000 kB\n"
    )
    checkpoint = root / "training_output/checkpoint/checkpoints/global_step_27"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("metadata")
    (checkpoint / "rank.distcp").write_bytes(b"dcp")
    hf = checkpoint / "hf_ckpt"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({
        "ttt_mode": arm == "ttt", "max_position_embeddings": formal.SEQ_LEN,
    }))
    (hf / "model.safetensors").write_bytes(b"weights")


class TwoRTX5090FormalStage1Test(unittest.TestCase):
    def test_progress_keeps_first_record_for_each_of_27_steps(self):
        rows = formal.parse_progress(training_log())
        self.assertEqual([row["step"] for row in rows], list(range(1, 28)))
        self.assertEqual(rows[-1]["cumulative_seconds"], 270.0)
        self.assertEqual(rows[-1]["individual_seconds"], 10.0)

    def test_complete_paired_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "formal"
            root.mkdir()
            for arm in formal.ARMS:
                make_arm(root / arm, arm)

            data = base / "stage1_32k.jsonl"
            data.write_text('{"content_split":"fixture"}\n')
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({
                "protocol": "in_place_ttt_public_corpus_v1", "profile": "full",
                "stages": [{"name": "stage1_32k", "artifacts": {
                    data.name: {"sha256": formal.sha256(data),
                                "size_bytes": data.stat().st_size},
                }}],
            }))
            corpus_audit = base / "corpus_audit.json"
            corpus_audit.write_text(json.dumps({
                "passed": True, "profile": "full",
                "bindings": {"manifest_sha256": formal.sha256(manifest)},
            }))
            timing = base / "timing.json"
            timing.write_text(json.dumps({
                "protocol": "in_place_ttt_2x5090_stage1_timing_audit_v1",
                "passed": True, "projected_sequential_stage1_seconds": 1000.0,
            }))
            release = base / "release.json"
            release.write_text(json.dumps({
                "protocol": formal.RELEASE_PROTOCOL, "accepted": True,
                "timing_audit_sha256": formal.sha256(timing),
            }))
            formal_plan, addendum = base / "formal.md", base / "addendum.md"
            model_config, runner = base / "config.json", base / "runner.sh"
            configs = {arm: base / f"{arm}.yaml" for arm in formal.ARMS}
            for path, content in (
                (formal_plan, "formal\n"), (addendum, "addendum\n"),
                (model_config, "{}\n"), (runner, "runner\n"),
            ):
                path.write_text(content)
            for arm, path in configs.items():
                path.write_text(arm + "\n")
            config_audit = root / "config_audit.json"
            config_audit.write_text(json.dumps({"passed": True}))
            metadata = {
                "protocol": formal.RUN_PROTOCOL, "hostname": formal.HOSTNAME,
                "upstream_commit": formal.UPSTREAM_COMMIT,
                "upstream_clean": True, "tool_clean": True,
                "tool_commit": "a" * 40, "steps_per_arm": formal.STEPS,
                "sequence_length": formal.SEQ_LEN,
                "global_batch_size": formal.GLOBAL_BATCH,
                "tokens_per_arm": formal.TRAIN_SIZE,
                "formal_plan_sha256": formal.sha256(formal_plan),
                "execution_addendum_sha256": formal.sha256(addendum),
                "timing_audit_sha256": formal.sha256(timing),
                "release_sha256": formal.sha256(release),
                "corpus_manifest_sha256": formal.sha256(manifest),
                "corpus_audit_sha256": formal.sha256(corpus_audit),
                "data_sha256": formal.sha256(data),
                "model_config_sha256": formal.sha256(model_config),
                "no_ttt_config_sha256": formal.sha256(configs["no_ttt"]),
                "ttt_config_sha256": formal.sha256(configs["ttt"]),
                "config_audit_sha256": formal.sha256(config_audit),
                "runner_sha256": formal.sha256(runner),
                "auditor_sha256": formal.sha256(Path(
                    "scripts/audit_in_place_ttt_2x5090_100m_stage1.py"
                ).resolve()),
            }
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            with mock.patch.object(formal, "MIN_DCP_BYTES", 1), \
                    mock.patch.object(formal, "MIN_HF_BYTES", 1):
                result = formal.audit(
                    root, formal_plan=formal_plan, execution_addendum=addendum,
                    timing_audit=timing, release=release,
                    corpus_manifest=manifest, corpus_audit=corpus_audit,
                    data=data, model_config=model_config,
                    configs=configs, runner=runner,
                )
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(set(result["arms"]), set(formal.ARMS))


if __name__ == "__main__":
    unittest.main()

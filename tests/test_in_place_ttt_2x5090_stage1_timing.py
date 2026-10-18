from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_2x5090_stage1_timing import (
    GLOBAL_BATCH,
    SEQ_LEN,
    STEPS,
    TOKENS_PER_STEP,
    audit,
    audit_arm,
    parse_progress,
    sha256,
)


def progress_log() -> str:
    settings = (
        "data_parallel_mode: fsdp2\n"
        "enable_activation_offload: true\n"
        "activation_gpu_limit: 0.0\n"
        "enable_fsdp_offload: false\n"
    )
    rows = [
        (1, "00:10", "10.00", "2.5", "3.5", "4e-6"),
        (2, "00:18", "9.00", "2.4", "3.4", "3e-6"),
        (3, "00:25", "8.33", "2.3", "3.3", "2e-6"),
    ]
    return settings + "".join(
        f"\rEpoch 1/1: 100%|####| {step}/3 [{elapsed}<00:00, {rate}s/it, "
        f"loss: {loss}, grad_norm: {grad}, lr: {lr}]"
        for step, elapsed, rate, loss, grad, lr in rows
    ) + (
        "\rEpoch 1/1: 100%|####| 3/3 [00:30<00:00, 10.00s/it, "
        "loss: 2.3, grad_norm: 3.3, lr: 2e-6]\n"
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:40.00\n"
    )


def make_arm(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "train.log").write_text(progress_log())
    (root / "exit_code.txt").write_text("0\n")
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


class TwoRTX5090Stage1TimingTest(unittest.TestCase):
    def test_progress_uses_first_completed_trace_per_step(self):
        rows = parse_progress(progress_log())
        self.assertEqual([row["step"] for row in rows], [1, 2, 3])
        self.assertEqual(
            [row["individual_seconds"] for row in rows], [10.0, 8.0, 7.0]
        )

    def test_complete_paired_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "timing"
            root.mkdir()
            for arm in ("no_ttt", "ttt"):
                make_arm(root / arm)
                errors, record = audit_arm(root / arm, arm)
                self.assertEqual(errors, [])
                self.assertEqual(record["warm_excluded_median_seconds"], 7.5)

            data = base / "stage1_32k.jsonl"
            data.write_text('{"content_split":"fixture"}\n')
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({
                "protocol": "in_place_ttt_public_corpus_v1",
                "profile": "full",
                "stages": [{
                    "name": "stage1_32k",
                    "artifacts": {data.name: {
                        "sha256": sha256(data), "size_bytes": data.stat().st_size,
                    }},
                }],
            }))
            corpus_audit = base / "corpus_audit.json"
            corpus_audit.write_text(json.dumps({
                "passed": True, "profile": "full",
                "bindings": {"manifest_sha256": sha256(manifest)},
            }))
            ladder = base / "ladder.json"
            ladder.write_text(json.dumps({
                "passed": True,
                "boundaries": {
                    f"{length}_{arm}": {} for length in (16384, 32768)
                    for arm in ("no_ttt", "ttt")
                },
            }))
            timing_plan = base / "timing_plan.md"
            addendum = base / "addendum.md"
            model_config = base / "config.json"
            runner = base / "runner.sh"
            configs = {arm: base / f"{arm}.yaml" for arm in ("no_ttt", "ttt")}
            for path, content in (
                (timing_plan, "timing\n"), (addendum, "addendum\n"),
                (model_config, "{}\n"), (runner, "runner\n"),
            ):
                path.write_text(content)
            for arm, path in configs.items():
                path.write_text(arm + "\n")
            (root / "config_audit.json").write_text(json.dumps({"passed": True}))
            metadata = {
                "protocol": "in_place_ttt_2x5090_stage1_timing_v1",
                "hostname": "user-Super-Server",
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True,
                "tool_commit": "a" * 40,
                "tool_clean": True,
                "steps": STEPS,
                "sequence_length": SEQ_LEN,
                "global_batch_size": GLOBAL_BATCH,
                "tokens_per_arm": STEPS * TOKENS_PER_STEP,
                "timing_plan_sha256": sha256(timing_plan),
                "execution_addendum_sha256": sha256(addendum),
                "ladder_audit_sha256": sha256(ladder),
                "corpus_manifest_sha256": sha256(manifest),
                "corpus_audit_sha256": sha256(corpus_audit),
                "data_sha256": sha256(data),
                "model_config_sha256": sha256(model_config),
                "runner_sha256": sha256(runner),
                "auditor_sha256": sha256(Path(
                    "scripts/audit_in_place_ttt_2x5090_stage1_timing.py"
                ).resolve()),
                "no_ttt_config_sha256": sha256(configs["no_ttt"]),
                "ttt_config_sha256": sha256(configs["ttt"]),
                "config_audit_sha256": sha256(root / "config_audit.json"),
            }
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            result = audit(
                root, timing_plan=timing_plan, execution_addendum=addendum,
                ladder_audit=ladder, corpus_manifest=manifest,
                corpus_audit=corpus_audit, data=data,
                model_config=model_config, configs=configs, runner=runner,
            )
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["projected_sequential_stage1_seconds"], 405.0)


if __name__ == "__main__":
    unittest.main()

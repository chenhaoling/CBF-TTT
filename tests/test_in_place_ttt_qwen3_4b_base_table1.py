from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_table1 import audit
from scripts.summarize_in_place_ttt_level_b_ruler import TASK_PREFIXES
from scripts.summarize_in_place_ttt_qwen3_4b_base_table1 import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CONTEXTS,
    MODEL_REVISION,
    render_markdown,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_TABLE1_REPORT_PLAN.md"
REPORTER = ROOT / "scripts/summarize_in_place_ttt_qwen3_4b_base_table1.py"
WATCHER = ROOT / "scripts/watch_and_prepare_in_place_ttt_qwen3_4b_base_table1_report.sh"


def write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def fixtures(base: Path) -> dict[str, Path]:
    matrix_root, formal_root = base / "matrix", base / "formal_256k"
    context_scores = {}
    for context_index, (context, suffix) in enumerate(CONTEXTS):
        arm_macros = {}
        for arm in ("no_ttt", "ttt"):
            tasks = {}
            inputs = {}
            counts = {}
            for task_index, task in enumerate(TASK_PREFIXES):
                no_ttt = 40.0 + context_index + task_index * 0.25
                delta = (task_index - 6) * 0.1 + context_index * 0.02
                tasks[f"{task}_{suffix}"] = no_ttt + (delta if arm == "ttt" else 0)
                inputs[f"{task}_{suffix}"] = f"input-{context_index}-{task_index}"
                counts[f"{task}_{suffix}"] = 100
            macro = sum(tasks.values()) / len(tasks)
            arm_macros[arm] = macro
            root = ((formal_root / arm) if suffix == "256k"
                    else (matrix_root / arm / suffix))
            write(root / "summary.json", {
                "protocol": "in_place_ttt_2x5090_ruler_summary_v1",
                "completed": True, "arm": arm, "context_length": context,
                "num_samples_per_task": 100, "task_count": 13,
                "exit_code": 0, "tokenizer_regex_warning_count": 0,
                "prediction_sample_counts": counts,
                "evaluation_input_sha256": inputs,
                "task_scores": tasks, "macro_accuracy": macro,
                "missing_result_tasks": [], "unexpected_result_tasks": [],
                "missing_prediction_tasks": [], "unexpected_prediction_tasks": [],
            })
        context_scores[suffix] = {
            "no_ttt": arm_macros["no_ttt"],
            "ttt": arm_macros["ttt"],
            "ttt_minus_no_ttt": arm_macros["ttt"] - arm_macros["no_ttt"],
        }
    corrected_matrix = write(base / "corrected_matrix.json", {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "context_scores": {
            suffix: context_scores[suffix] for _, suffix in CONTEXTS[:-1]
        },
    })
    corrected_formal = write(base / "corrected_formal.json", {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_256k_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "scores": context_scores["256k"],
    })
    primary_resource = write(base / "primary_resource.json", {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_resource_audit_v1",
        "passed": True, "errors": [],
        "boundaries": {
            "4k_no_ttt": {"gpu": {"peak_memory_used_mib": 1000,
                                      "memory_total_mib": 32607}},
            "4k_ttt": {"gpu": {"peak_memory_used_mib": 2000,
                                   "memory_total_mib": 32607}},
        },
    })
    resource_256k = write(base / "resource_256k.json", {
        "protocol": "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1",
        "passed": True, "errors": [],
        "boundaries": {
            "no_ttt": {"gpu": {"peak_memory_used_mib": 28000,
                                   "memory_total_mib": 32607}},
            "ttt": {"gpu": {"peak_memory_used_mib": 29000,
                                "memory_total_mib": 32607}},
        },
    })
    return {
        "matrix_root": matrix_root,
        "corrected_matrix_audit": corrected_matrix,
        "formal_256k_root": formal_root,
        "corrected_formal_256k_audit": corrected_formal,
        "corrected_primary_resource_audit": primary_resource,
        "corrected_256k_resource_audit": resource_256k,
        "plan": PLAN,
    }


class CorrectedBaseTable1ReportTest(unittest.TestCase):
    def test_plan_and_watcher_freeze_complete_cpu_only_report(self):
        text = PLAN.read_text()
        self.assertIn("10,000 resamples", text)
        self.assertIn("seed 42", text)
        self.assertIn("13 official RULER tasks", text)
        self.assertIn("never launches", text)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_table1_inputs", watcher)
        self.assertIn("completed_corrected_table1_report", watcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=", watcher)

    def test_summary_markdown_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            kwargs = fixtures(base)
            summary = summarize(**kwargs)
            self.assertTrue(summary["baseline_evidence_complete"])
            self.assertFalse(summary["gate_experiment_started"])
            self.assertEqual(summary["paired_task_bootstrap"]["seed"],
                             BOOTSTRAP_SEED)
            self.assertEqual(summary["paired_task_bootstrap"]["resamples"],
                             BOOTSTRAP_REPS)
            markdown = render_markdown(summary)
            self.assertIn("| 256K |", markdown)
            self.assertIn("not an exact reproduction", markdown)
            summary_path = write(base / "table1_summary.json", summary)
            report_path = base / "TABLE1_RESULT.md"
            report_path.write_text(markdown)
            args = argparse.Namespace(
                **kwargs, reporter=REPORTER, summary_json=summary_path,
                report_md=report_path, output=base / "audit.json",
            )
            result = audit(args)
            self.assertTrue(result["passed"], result["errors"])

            child = kwargs["matrix_root"] / "ttt/4k/summary.json"
            changed = json.loads(child.read_text())
            first = sorted(changed["task_scores"])[0]
            changed["task_scores"][first] += 1.0
            write(child, changed)
            failed = audit(args)
            self.assertFalse(failed["passed"])
            self.assertIn("independent recomputation failed", failed["errors"][0])


if __name__ == "__main__":
    unittest.main()

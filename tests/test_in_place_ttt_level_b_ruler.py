import csv
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import scripts.audit_in_place_ttt_level_b_ruler as single_audit_module
from scripts.audit_in_place_ttt_level_b_ruler import audit
from scripts.audit_in_place_ttt_level_b_ruler_matrix import audit as audit_matrix
from scripts.summarize_in_place_ttt_level_b_ruler import (
    TASK_PREFIXES,
    evaluation_input_sha256,
    summarize,
)
from scripts.summarize_in_place_ttt_level_b_ruler_matrix import (
    CONTEXTS,
    summarize as summarize_matrix,
)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_raw_run(root: pathlib.Path, arm="no_ttt", context=4096, score=50.0) -> None:
    suffix = {4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k",
              65536: "64k", 131072: "128k", 262144: "256k"}[context]
    model = "model"
    result_dir = root / "opencompass/20261018/results" / model
    prediction_dir = root / "opencompass/20261018/predictions" / model
    details = {"type": "GEN"} | {str(index): {"prediction": "x"} for index in range(100)}
    predictions = {
        str(index): {
            "origin_prompt": f"prompt {index}",
            "references": f"answer {index}",
            "prediction": "x",
        }
        for index in range(100)
    }
    for prefix in TASK_PREFIXES:
        name = f"{prefix}_{suffix}"
        write_json(result_dir / f"{name}.json", {"score": score, "details": details})
        write_json(prediction_dir / f"{name}.json", predictions)
    summary_dir = root / "opencompass/20261018/summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary_20261018.txt").write_text(
        f"ruler_{suffix}: {{'score': {score}, 'naive_average': {score}}}\n",
        encoding="utf-8",
    )
    write_json(root / "run_metadata.json", {
        "protocol": "in_place_ttt_level_b_ruler_run_v1",
        "arm": arm,
        "context_length": context,
        "num_samples_per_task": 100,
    })
    (root / "eval.log").write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45\n", encoding="utf-8"
    )
    (root / "exit_code.txt").write_text("0\n", encoding="utf-8")
    with (root / "gpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "index", "memory_used_mib", "memory_total_mib", "utilization_gpu"])
        writer.writerow([1.0, 3, 12345, 81920, 50])


class RulerSingleTests(unittest.TestCase):
    def test_frozen_addendum_binds_current_implementation_and_sources(self):
        addendum = pathlib.Path(
            "IN_PLACE_TTT_BASELINE_LEVEL_B_RULER_ADDENDUM.md"
        ).read_text(encoding="utf-8")
        files = (
            "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py",
            "scripts/run_in_place_ttt_level_b_ruler_single.sh",
            "scripts/summarize_in_place_ttt_level_b_ruler.py",
            "scripts/audit_in_place_ttt_level_b_ruler.py",
            "scripts/run_in_place_ttt_level_b_ruler_matrix.sh",
            "scripts/summarize_in_place_ttt_level_b_ruler_matrix.py",
            "scripts/audit_in_place_ttt_level_b_ruler_matrix.py",
            "scripts/run_in_place_ttt_level_b_training.sh",
            "scripts/audit_in_place_ttt_level_b_training.py",
        )
        for filename in files:
            self.assertIn(f"`{filename}`", addendum)
            self.assertIn(f"`{sha(pathlib.Path(filename))}`", addendum)
        for digest in single_audit_module.EXPECTED_GENERATOR_SHA256.values():
            self.assertIn(f"`{digest}`", addendum)
        for digest in single_audit_module.EXPECTED_SOURCE_SHA256.values():
            self.assertIn(f"`{digest}`", addendum)

    def test_registered_config_is_seven_context_and_formal_100(self):
        source = pathlib.Path(
            "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py"
        ).read_text(encoding="utf-8")
        for context in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            self.assertIn(f"{context}:", source)
        self.assertIn('num_samples != 100', source)
        self.assertIn('fix_mistral_regex=False', source)

    def test_single_summary_requires_exact_13_by_100(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            make_raw_run(root)
            result = summarize(root)
            self.assertTrue(result["completed"])
            self.assertEqual(result["task_count"], 13)
            self.assertEqual(result["macro_accuracy"], 50.0)
            prediction = next((root / "opencompass").glob("*/predictions/*/*.json"))
            payload = json.loads(prediction.read_text())
            payload.pop("99")
            write_json(prediction, payload)
            self.assertFalse(summarize(root)["completed"])

    def test_independent_single_audit_recomputes_raw_results(self):
        with tempfile.TemporaryDirectory() as temp:
            base = pathlib.Path(temp)
            root = base / "run"
            root.mkdir()
            make_raw_run(root)
            plan = base / "plan.md"
            addendum = base / "addendum.md"
            plan.write_text("plan\n")
            addendum.write_text("addendum\n")
            model = base / "model"
            tokenizer = base / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            write_json(model / "config.json", {
                "ttt_mode": False, "max_position_embeddings": 131072,
            })
            (model / "tokenizer.json").write_text("tokenizer\n")
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("tokenizer\n")
            write_json(tokenizer / "tokenizer_config.json", {"model_max_length": 131072})
            training = base / "training_audit.json"
            write_json(training, {
                "passed": True, "stage": "stage2", "arm": "no_ttt",
                "final_hf_path": str(model),
            })
            generator_modules = {
                f"opencompass.{prefix}.ruler_{task}{suffix}"
                for prefix, suffix in (
                    ("configs.datasets.ruler", "_gen"),
                    ("datasets.ruler", ""),
                )
                for task in ("qa", "niah", "vt", "fwe", "cwe")
            }
            generator_files = {}
            for index, module in enumerate(sorted(generator_modules)):
                path = base / "dependencies" / f"generator_{index}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(module + "\n", encoding="utf-8")
                generator_files[module] = {"path": str(path), "sha256": sha(path)}
            source_files = {}
            for relative in (
                "data/ruler/dev-v2.0.json",
                "data/ruler/hotpotqa.json",
                "data/ruler/PaulGrahamEssays.jsonl",
            ):
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n", encoding="utf-8")
                source_files[relative] = {"path": str(path), "sha256": sha(path)}
            metadata = json.loads((root / "run_metadata.json").read_text())
            metadata.update({
                "upstream_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1",
                "upstream_clean": True,
                "tool_clean": True,
                "plan_sha256": sha(plan),
                "evaluation_addendum_sha256": sha(addendum),
                "training_audit_path": str(training),
                "training_audit_sha256": sha(training),
                "model_path": str(model),
                "model_config_sha256": sha(model / "config.json"),
                "model_files": {
                    str(path.relative_to(model)): {
                        "size_bytes": path.stat().st_size, "sha256": sha(path),
                    }
                    for path in sorted(model.iterdir()) if path.is_file()
                },
                "tokenizer_path": str(tokenizer),
                "tokenizer_json_sha256": sha(tokenizer / "tokenizer.json"),
                "tokenizer_config_sha256": sha(tokenizer / "tokenizer_config.json"),
                "physical_gpu": 3,
                "ruler_random_seed": 42,
                "context_suffix": "4k",
                "extrapolation": False,
                "tokenizer_policy": "qwen_original_tokenizer_mistral_fix_false_v1",
                "fix_mistral_regex": False,
                "config_sha256": sha(pathlib.Path(
                    "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py"
                )),
                "tool_artifacts": {
                    relative: sha(pathlib.Path(relative))
                    for relative in (
                        "eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py",
                        "scripts/summarize_in_place_ttt_level_b_ruler.py",
                        "scripts/audit_in_place_ttt_level_b_ruler.py",
                    )
                },
                "dependency_versions": {
                    "opencompass": "0.5.4",
                    "transformers": "4.57.3",
                    "datasets": "2.21.0",
                    "torch": "2.8.0+cu128",
                },
                "ruler_generator_files": generator_files,
                "ruler_source_files": source_files,
            })
            write_json(root / "run_metadata.json", metadata)
            write_json(root / "summary.json", summarize(root))
            expected_generators = {
                module: record["sha256"] for module, record in generator_files.items()
            }
            expected_sources = {
                relative: record["sha256"] for relative, record in source_files.items()
            }
            patches = (
                mock.patch.object(
                    single_audit_module, "EXPECTED_GENERATOR_SHA256", expected_generators
                ),
                mock.patch.object(
                    single_audit_module, "EXPECTED_SOURCE_SHA256", expected_sources
                ),
            )
            patches[0].start()
            patches[1].start()
            self.addCleanup(patches[1].stop)
            self.addCleanup(patches[0].stop)
            passed = audit(root, plan, addendum)
            self.assertTrue(passed["passed"], passed["errors"])
            result = next((root / "opencompass").glob("*/results/*/*.json"))
            payload = json.loads(result.read_text())
            payload["score"] = 49.0
            write_json(result, payload)
            failed = audit(root, plan, addendum)
            self.assertFalse(failed["passed"])
            self.assertIn("summary macro does not reproduce raw task scores", failed["errors"])


class RulerMatrixTests(unittest.TestCase):
    def _fixture(self, root: pathlib.Path) -> None:
        for index, (context, suffix) in enumerate(CONTEXTS):
            for arm, offset in (("no_ttt", 0.0), ("ttt", 1.0)):
                run = root / arm / suffix
                score = 40.0 + index + offset
                task_scores = {f"{prefix}_{suffix}": score for prefix in TASK_PREFIXES}
                write_json(run / "run_metadata.json", {"arm": arm, "context_length": context})
                result_dir = run / "opencompass/20261018/results/model"
                prediction_dir = run / "opencompass/20261018/predictions/model"
                input_hashes = {}
                for prefix in TASK_PREFIXES:
                    name = f"{prefix}_{suffix}.json"
                    write_json(result_dir / name, {"score": score})
                    prediction_payload = {
                        str(sample): {
                            "origin_prompt": f"prompt {sample}",
                            "references": f"answer {sample}",
                            "prediction": arm,
                        }
                        for sample in range(100)
                    }
                    write_json(prediction_dir / name, prediction_payload)
                    input_hashes[name[:-5]] = evaluation_input_sha256(prediction_payload)
                write_json(run / "summary.json", {
                    "completed": True, "arm": arm, "context_length": context,
                    "macro_accuracy": score, "task_scores": task_scores,
                    "evaluation_input_sha256": input_hashes,
                })
                write_json(run / "independent_audit.json", {
                    "passed": True, "arm": arm, "context_length": context,
                    "bindings": {
                        "raw_result_sha256": {
                            path.name: sha(path) for path in sorted(result_dir.glob("*.json"))
                        },
                        "raw_prediction_sha256": {
                            path.name: sha(path) for path in sorted(prediction_dir.glob("*.json"))
                        },
                    },
                })

    def test_matrix_summary_and_independent_recompute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self._fixture(root)
            matrix = summarize_matrix(root)
            self.assertTrue(matrix["completed"], matrix["errors"])
            self.assertEqual(matrix["context_scores"]["256k"]["ttt_minus_no_ttt"], 1.0)
            self.assertEqual(
                matrix["registered_4k_to_128k_mean"]["ttt_minus_no_ttt"], 1.0
            )
            write_json(root / "matrix_summary.json", matrix)
            checked = audit_matrix(root)
            self.assertTrue(checked["passed"], checked["errors"])
            raw = next((root / "ttt/64k/opencompass").glob("*/results/*/*.json"))
            payload = json.loads(raw.read_text())
            payload["score"] += 5
            write_json(raw, payload)
            failed = audit_matrix(root)
            self.assertFalse(failed["passed"])

    def test_paired_input_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self._fixture(root)
            summary_path = root / "ttt/32k/summary.json"
            summary = json.loads(summary_path.read_text())
            task = next(iter(summary["evaluation_input_sha256"]))
            summary["evaluation_input_sha256"][task] = "0" * 64
            write_json(summary_path, summary)
            matrix = summarize_matrix(root)
            self.assertFalse(matrix["completed"])
            self.assertIn("paired RULER inputs differ: 32k", matrix["errors"])


if __name__ == "__main__":
    unittest.main()
